import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

from synthetic_time_series_clean.evaluation import sliced_wd
from synthetic_time_series_clean.inference.parsing import parse_full_text


ROOT = Path(__file__).resolve().parents[1]
HAS_MEMORIZATION_DEPS = (
    importlib.util.find_spec("omegaconf") is not None
    and importlib.util.find_spec("transformers") is not None
)
if HAS_MEMORIZATION_DEPS:
    from synthetic_time_series_clean.inference import generate_memorization


class SlicedWdTests(unittest.TestCase):
    def test_default_synthetic_views_match_old_run_wd_contract(self):
        views = sliced_wd.default_text_views("synthetic")
        self.assertEqual(
            [(view.name, view.reference_field, view.generated_field) for view in views],
            [
                ("event_csv", "original_events", "synthetic_events"),
                ("static_event", "original_full_text", "synthetic_full_text"),
            ],
        )
        self.assertEqual(
            views[0].generated_repair_fields,
            (
                "synthetic_continuation_raw",
                "synthetic_raw_full_text",
                "synthetic_full_text",
                "synthetic_events",
            ),
        )

    def test_default_memorization_view_matches_old_run_wd_contract(self):
        views = sliced_wd.default_text_views("memorization")
        self.assertEqual(len(views), 1)
        view = views[0]
        self.assertEqual(view.name, "event_csv")
        self.assertEqual(view.reference_field, "target_continuation_event_csv")
        self.assertEqual(view.generated_field, "generated_continuation_event_csv")
        self.assertIn("generated_raw_continuation", view.generated_repair_fields)

    def test_no_strict_or_repaired_mode_in_sliced_wd_source(self):
        source = (ROOT / "evaluation" / "sliced_wd.py").read_text(encoding="utf-8")
        self.assertNotIn("strict", source)
        self.assertNotIn("repaired mode", source)
        self.assertNotIn("VALID_WD_MODES", source)

    def test_sliced_wd_metric_fields_only_expose_sliced_wasserstein(self):
        self.assertIn("metric", sliced_wd.METRIC_FIELDS)
        row = {
            "metric": "sliced_wasserstein",
            "value": 1.0,
            "reference_count": 2,
            "generated_count": 2,
            "task_type": "synthetic",
            "run_name": "run",
            "source_label": "",
            "temperature": "",
            "temp_tag": "",
            "fraction": "",
            "fraction_tag": "",
            "text_view": "event_csv",
            "embedding_name": "bge_m3",
            "embedding_model": "/tmp/model",
            "embedding_max_length": 8192,
            "source_path": "/tmp/source.jsonl",
            "valid_count": 2,
            "complete_count": 2,
            "truncated_count": 0,
            "invalid_malformed_count": 0,
            "good_count": 2,
            "repaired_count": 0,
            "included_count": 2,
            "repair_failed_count": 0,
        }
        self.assertEqual(set(row), set(sliced_wd.METRIC_FIELDS))
        self.assertEqual(row["metric"], "sliced_wasserstein")

    def test_write_grouped_outputs_keeps_validation_when_no_metric_rows(self):
        validation = {field: "" for field in sliced_wd.VALIDATION_FIELDS}
        validation.update(
            {
                "task_type": "synthetic",
                "run_name": "run_a",
                "source_path": "/tmp/missing.jsonl",
                "text_view": "event_csv",
                "total_rows": 1,
                "included_count": 0,
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            sliced_wd.write_grouped_outputs(output_root, [], [validation])
            metric_lines = (output_root / "combined_distribution_metrics.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(metric_lines), 1)
            self.assertTrue((output_root / "run_a" / "input_validation.csv").exists())

    def test_no_public_extra_eval_metric_modules_remain(self):
        removed = {
            "ppl.py",
            "mauve_eval.py",
            "memorization_metrics.py",
            "checkpoint_validation.py",
            "build_analysis_dataset.py",
            "wd.py",
        }
        present = {path.name for path in (ROOT / "evaluation").glob("*.py")}
        self.assertTrue(removed.isdisjoint(present))

    def test_event_csv_repair_returns_plain_csv_without_tags(self):
        text = "time,code,numeric_value\n2024-01-01T00:00:00.000000,a,1"
        repaired = sliced_wd.repair_event_text(text, preserve_static=False)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(repaired.status, "good")
        self.assertEqual(repaired.text, text)
        self.assertNotIn("</event>", repaired.text)

    def test_event_csv_repair_strips_tags_from_fallback_text(self):
        text = "<event>time,code,numeric_value\n2024-01-01T00:00:00.000000,a,1</event>"
        repaired = sliced_wd.repair_event_text(text, preserve_static=False)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(repaired.text, "time,code,numeric_value\n2024-01-01T00:00:00.000000,a,1")

    def test_static_event_repair_preserves_wrapping_tags(self):
        text = "<static>sex==F</static><event>time,code,numeric_value\n2024-01-01T00:00:00.000000,a,1"
        repaired = sliced_wd.repair_event_text(text, preserve_static=True)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertTrue(repaired.text.startswith("<static>sex==F</static><event>"))
        self.assertTrue(repaired.text.endswith("</event>"))

    def test_prepare_view_inputs_builds_validation_once_per_input_view(self):
        source_path = Path("/tmp/source.jsonl")
        event_csv = "time,code,numeric_value\n2024-01-01T00:00:00.000000,a,1"
        row = {
            "valid": True,
            "status": "complete",
            "original_events": event_csv,
            "synthetic_events": event_csv,
            "original_full_text": f"<static>sex==F</static><event>{event_csv}</event>",
            "synthetic_full_text": f"<static>sex==F</static><event>{event_csv}</event>",
        }
        item = sliced_wd.InputItem(metadata={"run_name": "run", "source_path": str(source_path)}, source_path=source_path)
        views = sliced_wd.default_text_views("synthetic")

        prepared, validation_rows = sliced_wd.prepare_view_inputs(
            rows_by_path={source_path: [row]},
            items=[item],
            task_type="synthetic",
            text_views=views,
            valid_field="valid",
            status_field="status",
            repair_event_text_enabled=True,
        )

        self.assertEqual(len(validation_rows), len(views))
        self.assertEqual(set(prepared), {(source_path, view.name) for view in views})
        self.assertEqual(validation_rows[0]["total_rows"], 1)


class SchemaTests(unittest.TestCase):
    def _dict_literal_keys_in_file(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
        return keys

    def test_memorization_output_schema_contains_wd_required_fields(self):
        keys = self._dict_literal_keys_in_file(ROOT / "inference" / "generate_memorization.py")
        required = {
            "target_continuation_event_csv",
            "target_continuation_normalized_text",
            "generated_raw_continuation",
            "generated_raw_continuation_before_restore",
            "generated_raw_full_text",
            "generated_event_csv",
            "generated_continuation_event_csv",
            "generated_continuation_normalized_text",
            "train_event_count",
            "seeded_event_count",
            "target_continuation_event_count",
            "generated_continuation_event_count",
            "exact_match",
            "status",
            "valid",
            "is_complete",
            "data_split_used",
        }
        self.assertTrue(required.issubset(keys))

    def test_synthetic_vllm_output_schema_contains_wd_required_fields(self):
        keys = self._dict_literal_keys_in_file(ROOT / "inference" / "generate_vllm.py")
        required = {
            "original_events",
            "original_full_text",
            "synthetic_continuation_raw",
            "synthetic_continuation",
            "synthetic_raw_full_text",
            "synthetic_events",
            "synthetic_full_text",
            "prompt_prefix",
            "status",
            "valid",
            "is_complete",
            "synthetic_finish_reason",
            "synthetic_stop_reason",
            "synthetic_raw_contains_closing_event_tag",
            "synthetic_event_count",
            "original_event_count",
        }
        self.assertTrue(required.issubset(keys))

    def test_inference_parser_rejects_bad_event_header(self):
        parsed = parse_full_text(
            "<static>x</static><event>foo,bar,baz\n2024-01-01T00:00:00.000000,a,1</event>",
            allow_truncated=False,
        )
        self.assertFalse(parsed["valid"])
        self.assertEqual(parsed["status"], "invalid_malformed")

    def test_inference_parser_rejects_bad_event_timestamp(self):
        parsed = parse_full_text(
            "<static>x</static><event>time,code,numeric_value\nnot-a-time,a,1</event>",
            allow_truncated=False,
        )
        self.assertFalse(parsed["valid"])
        self.assertEqual(parsed["status"], "invalid_malformed")


@unittest.skipIf(not HAS_MEMORIZATION_DEPS, "omegaconf/transformers are not installed")
class MemorizationStage4ATests(unittest.TestCase):
    def test_fraction_based_candidate_uses_floor_seeded_event_count(self):
        row = {
            "id_column": "subject_id",
            "id_value": 1,
            "train_static": "sex==F",
            "train_events": (
                "time,code,numeric_value\n"
                "2024-01-01T00:00:00.000000,a,1\n"
                "2024-01-02T00:00:00.000000,b,2\n"
                "2024-01-03T00:00:00.000000,c,3\n"
                "2024-01-04T00:00:00.000000,d,4"
            ),
            "train_full_text": "",
            "train_event_count": 4,
        }
        candidate = generate_memorization.build_memorization_candidate(
            row=row,
            row_index=0,
            candidate_index=0,
            fraction=0.5,
        )
        self.assertEqual(candidate["strategy_name"], "fraction_50pct")
        self.assertEqual(candidate["seeded_event_count"], 2)
        self.assertEqual(candidate["target_continuation_event_count"], 2)
        self.assertIn("a,1", candidate["prompt"])
        self.assertIn("b,2", candidate["prompt"])
        self.assertNotIn("c,3", candidate["prompt"])
        self.assertIn("c,3", candidate["target_continuation_event_csv"])


@unittest.skipIf(importlib.util.find_spec("numpy") is None, "numpy is not installed")
class SlicedWdMathTests(unittest.TestCase):
    def test_sliced_wd_zero_for_identical_embeddings(self):
        import numpy as np

        from synthetic_time_series_clean.evaluation.embedding_metrics import sliced_wasserstein_distance

        values = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        distance = sliced_wasserstein_distance(values, values, n_projections=8, seed=42)
        self.assertAlmostEqual(distance, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
