import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from synthetic_time_series_clean.common.config import configured_split_name
from synthetic_time_series_clean.data.splits import check_split_overlap

HAS_POLARS = importlib.util.find_spec("polars") is not None
if HAS_POLARS:
    import polars as pl

    from synthetic_time_series_clean.data.rendering import (
        MIMICShardSource,
        load_split_sources,
        render_mimic_patient_components_from_df,
        render_patient_text,
    )


class RenderingAndSplitTests(unittest.TestCase):
    def test_renderer_uses_configured_time_column_and_stable_tie_breaker(self):
        if not HAS_POLARS:
            self.skipTest("polars is not installed in this Python environment")
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = MIMICShardSource(
                data_path=Path(tmp_dir) / "part.parquet",
                id_column="subject_id",
                time_column="chart_time",
                identifiers={1},
            )
            df = pl.DataFrame(
                {
                    "subject_id": [1, 1, 1],
                    "chart_time": [None, "2024-01-01T00:00:00.000000", "2024-01-01T00:00:00.000000"],
                    "code": ["sex", "b_code", "a_code"],
                    "numeric_value": ["F", "2", "1"],
                }
            ).with_row_index("_source_row_index")
            rendered = render_mimic_patient_components_from_df(
                filtered_df=df,
                source=source,
                id_value=1,
                max_rows=10,
            )
            self.assertIn("time,code,numeric_value", rendered.event_csv)
            self.assertTrue(rendered.event_csv.splitlines()[1].endswith("b_code,2"))
            self.assertTrue(rendered.event_csv.splitlines()[2].endswith("a_code,1"))

    def test_split_overlap_detects_shared_identifier(self):
        overlaps = check_split_overlap(
            split_ids={
                "train": [("subject_id", 1), ("subject_id", 2)],
                "tuning": [("subject_id", 3)],
                "held_out": [("subject_id", 2)],
            }
        )
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0].left, "train")
        self.assertEqual(overlaps[0].right, "held_out")
        self.assertEqual(overlaps[0].count, 1)

    def test_configured_split_name_uses_cfg_splits(self):
        cfg = {"splits": {"train_dataset": "custom_train"}}
        self.assertEqual(configured_split_name(cfg, "train"), "custom_train")
        self.assertEqual(configured_split_name(cfg, "held_out"), "held_out")

    def test_render_patient_text_merges_rows_across_shards(self):
        if not HAS_POLARS:
            self.skipTest("polars is not installed in this Python environment")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "data" / "train"
            split_dir.mkdir(parents=True)
            pl.DataFrame(
                {
                    "subject_id": [1],
                    "time": ["2024-01-01T00:00:00.000000"],
                    "code": ["a"],
                    "numeric_value": ["1"],
                }
            ).write_parquet(split_dir / "part0.parquet")
            pl.DataFrame(
                {
                    "subject_id": [1],
                    "time": ["2024-01-02T00:00:00.000000"],
                    "code": ["b"],
                    "numeric_value": ["2"],
                }
            ).write_parquet(split_dir / "part1.parquet")
            cfg = SimpleNamespace(
                data=SimpleNamespace(
                    data_items=[
                        {
                            "path": str(root),
                            "identifier_column": "subject_id",
                            "time_column": "time",
                        }
                    ]
                )
            )
            sources, identifiers = load_split_sources(cfg=cfg, split="train")
            self.assertEqual(identifiers, [("subject_id", 1)])
            rendered = render_patient_text(
                data_sources=sources,
                id_column="subject_id",
                id_value=1,
                max_rows=10,
            )
            self.assertIn("2024-01-01T00:00:00.000000,a,1", rendered)
            self.assertIn("2024-01-02T00:00:00.000000,b,2", rendered)


if __name__ == "__main__":
    unittest.main()
