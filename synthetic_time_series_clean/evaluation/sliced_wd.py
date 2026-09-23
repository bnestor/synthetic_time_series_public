"""Single Sliced Wasserstein Distance evaluator for synthetic and memorization outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthetic_time_series_clean.common.io import read_csv, read_jsonl, write_csv, write_json
from synthetic_time_series_clean.common.runtime import safe_path_part


EVENT_HEADER = "time,code,numeric_value"
EVENT_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$")

METRIC_FIELDS = [
    "metric",
    "value",
    "reference_count",
    "generated_count",
    "task_type",
    "run_name",
    "source_label",
    "temperature",
    "temp_tag",
    "fraction",
    "fraction_tag",
    "text_view",
    "embedding_name",
    "embedding_model",
    "embedding_max_length",
    "source_path",
    "valid_count",
    "complete_count",
    "truncated_count",
    "invalid_malformed_count",
    "good_count",
    "repaired_count",
    "included_count",
    "repair_failed_count",
]

VALIDATION_FIELDS = [
    "task_type",
    "run_name",
    "source_label",
    "temperature",
    "temp_tag",
    "fraction",
    "fraction_tag",
    "source_path",
    "text_view",
    "total_rows",
    "valid_count",
    "complete_count",
    "truncated_count",
    "invalid_malformed_count",
    "good_count",
    "repaired_count",
    "included_count",
    "repair_failed_count",
    "missing_closing_event_tag_count",
    "dropped_tail_count",
    "repair_dropped_tail_line_count",
    "status_counts",
]


@dataclass(frozen=True)
class TextView:
    name: str
    reference_field: str
    generated_field: str
    reference_repair_fields: tuple[str, ...]
    generated_repair_fields: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    model_path: Path
    max_length: int = 512
    batch_size: int = 32


@dataclass(frozen=True)
class RepairedText:
    text: str
    source_field: str
    status: str
    event_count: int
    dropped_tail_lines: int
    header_found: bool
    had_closing_event_tag: bool
    missing_closing_event_tag: bool


@dataclass(frozen=True)
class InputItem:
    metadata: dict[str, Any]
    source_path: Path

    @property
    def run_name(self) -> str:
        return str(self.metadata.get("run_name") or "run")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def resolve_path(path_like: str | Path, work_dir: Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (work_dir / path).resolve()


def parse_csv_list(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def cfg_get(cfg: Any, *path: str, default: Any = None) -> Any:
    current = cfg
    for key in path:
        if current is None:
            return default
        try:
            current = current[key]
        except Exception:
            try:
                current = getattr(current, key)
            except Exception:
                return default
    return default if current is None else current


def choose(cli_value: Any, cfg: Any, *path: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else cfg_get(cfg, *path, default=default)


def parse_text_view(value: str) -> TextView:
    parts = value.split(":")
    if len(parts) not in {3, 5} or not all(part.strip() for part in parts[:3]):
        raise ValueError(
            "--text-view must have format name:reference_json_field:generated_json_field "
            "or name:reference:generated:reference_repair_fields:generated_repair_fields"
        )
    reference_field = parts[1].strip()
    generated_field = parts[2].strip()
    if len(parts) == 5:
        reference_repair_fields = tuple(item.strip() for item in parts[3].split("|") if item.strip())
        generated_repair_fields = tuple(item.strip() for item in parts[4].split("|") if item.strip())
    else:
        reference_repair_fields = (reference_field,)
        generated_repair_fields = (generated_field,)
    if reference_field not in reference_repair_fields:
        reference_repair_fields = (reference_field, *reference_repair_fields)
    if generated_field not in generated_repair_fields:
        generated_repair_fields = (generated_field, *generated_repair_fields)
    return TextView(
        name=parts[0].strip(),
        reference_field=reference_field,
        generated_field=generated_field,
        reference_repair_fields=reference_repair_fields,
        generated_repair_fields=generated_repair_fields,
    )


def default_text_views(task_type: str) -> list[TextView]:
    if task_type == "synthetic":
        return [
            parse_text_view(
                "event_csv:original_events:synthetic_events:"
                "original_events|original_full_text:"
                "synthetic_continuation_raw|synthetic_raw_full_text|synthetic_full_text|synthetic_events"
            ),
            parse_text_view(
                "static_event:original_full_text:synthetic_full_text:"
                "original_full_text|original_events:"
                "synthetic_raw_full_text|synthetic_full_text|synthetic_continuation_raw|synthetic_events"
            ),
        ]
    if task_type == "memorization":
        return [
            parse_text_view(
                "event_csv:target_continuation_event_csv:generated_continuation_event_csv:"
                "target_continuation_event_csv|target_continuation_normalized_text:"
                "generated_raw_continuation|generated_continuation_normalized_text|generated_continuation_event_csv"
            )
        ]
    raise ValueError("task_type must be synthetic or memorization")


def parse_manifest_filters(values: list[str] | None) -> dict[str, set[str]]:
    filters: dict[str, set[str]] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--filter must have format column=value[,value2]; got {value!r}")
        column, raw_values = value.split("=", 1)
        allowed = {item.strip() for item in raw_values.split(",") if item.strip()}
        if not column.strip() or not allowed:
            raise ValueError(f"--filter must have format column=value[,value2]; got {value!r}")
        filters.setdefault(column.strip(), set()).update(allowed)
    return filters


def load_input_manifest(path: Path, work_dir: Path, source_path_column: str) -> list[InputItem]:
    items: list[InputItem] = []
    for row in read_csv(path):
        source_raw = str(row.get(source_path_column) or "").strip()
        if not source_raw:
            raise ValueError(f"Input manifest row is missing {source_path_column!r}: {row}")
        metadata = dict(row)
        metadata["source_path"] = str(resolve_path(source_raw, work_dir))
        items.append(InputItem(metadata=metadata, source_path=Path(metadata["source_path"])))
    return items


def apply_manifest_filters(items: list[InputItem], filters: dict[str, set[str]]) -> list[InputItem]:
    if not filters:
        return items
    selected: list[InputItem] = []
    for item in items:
        keep = True
        for column, allowed in filters.items():
            if str(item.metadata.get(column, "")) not in allowed:
                keep = False
                break
        if keep:
            selected.append(item)
    return selected


def load_embedding_specs(path: Path, work_dir: Path, names_csv: str | None) -> list[EmbeddingSpec]:
    selected = set(parse_csv_list(names_csv))
    specs: list[EmbeddingSpec] = []
    for row in read_csv(path):
        name = str(row.get("embedding_name") or row.get("name") or "").strip()
        if not name:
            raise ValueError(f"Embedding manifest row is missing embedding_name: {row}")
        if selected and name not in selected:
            continue
        model_path = str(row.get("model_path") or "").strip()
        if not model_path:
            raise ValueError(f"Embedding manifest row is missing model_path: {row}")
        specs.append(
            EmbeddingSpec(
                name=name,
                model_path=resolve_path(model_path, work_dir),
                max_length=int(row.get("max_length") or 512),
                batch_size=int(row.get("batch_size") or 32),
            )
        )
    if not specs:
        raise ValueError(f"No embedding specs selected from {path}; requested={names_csv!r}")
    return specs


def parse_csv_event_line(line: str) -> list[str] | None:
    try:
        parsed = next(csv.reader([line]))
    except Exception:
        return None
    if len(parsed) != 3:
        return None
    return [part.strip() for part in parsed]


def is_event_header(line: str) -> bool:
    parsed = parse_csv_event_line(line.strip())
    return parsed is not None and [part.lower() for part in parsed] == ["time", "code", "numeric_value"]


def is_valid_event_row(line: str) -> bool:
    parsed = parse_csv_event_line(line.strip())
    if parsed is None:
        return False
    timestamp, code, _numeric_value = parsed
    return bool(EVENT_TIME_RE.match(timestamp)) and bool(code)


def static_prefix(text: str) -> str:
    start = text.find("<static>")
    end = text.find("</static>")
    if start >= 0 and end >= start:
        return text[start : end + len("</static>")]
    return ""


def event_payload(text: str) -> tuple[str, bool]:
    payload = text.strip()
    start = payload.find("<event>")
    if start >= 0:
        payload = payload[start + len("<event>") :]
    end = payload.find("</event>")
    had_closing = end >= 0
    if end >= 0:
        payload = payload[:end]
    return payload.replace("<event>", "").replace("</event>", "").strip(), had_closing


def parse_event_payload(payload: str) -> tuple[list[str], bool, int]:
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    header_found = any(is_event_header(line) for line in lines)
    event_lines: list[str] = []
    seen_header = False
    dropping_tail = False
    dropped_tail_lines = 0
    for line in lines:
        if dropping_tail:
            dropped_tail_lines += 1
            continue
        if is_event_header(line):
            seen_header = True
            continue
        if is_valid_event_row(line):
            event_lines.append(line)
            continue
        if seen_header or event_lines:
            dropping_tail = True
            dropped_tail_lines += 1
    return event_lines, header_found, dropped_tail_lines


def preserve_static_for_view(view: TextView) -> bool:
    names = (view.name, view.reference_field, view.generated_field)
    return any("static" in name or "full_text" in name for name in names)


def repair_event_text(text: str, *, preserve_static: bool) -> RepairedText | None:
    raw = str(text or "")
    if not raw.strip():
        return None
    static = static_prefix(raw) if preserve_static else ""
    payload, had_closing = event_payload(raw)
    event_lines, header_found, dropped_tail_lines = parse_event_payload(payload)
    if not event_lines:
        return None
    is_good = header_found and dropped_tail_lines == 0 and (had_closing or not preserve_static)
    if is_good:
        text_value = raw.strip() if preserve_static else EVENT_HEADER + "\n" + "\n".join(event_lines)
        return RepairedText(
            text=text_value,
            source_field="",
            status="good",
            event_count=len(event_lines),
            dropped_tail_lines=0,
            header_found=header_found,
            had_closing_event_tag=had_closing,
            missing_closing_event_tag=False,
        )
    body = EVENT_HEADER + "\n" + "\n".join(event_lines)
    repaired = f"{static}<event>{body}</event>" if preserve_static else body
    return RepairedText(
        text=repaired,
        source_field="",
        status="repaired",
        event_count=len(event_lines),
        dropped_tail_lines=dropped_tail_lines,
        header_found=header_found,
        had_closing_event_tag=had_closing,
        missing_closing_event_tag=not had_closing,
    )


def text_from_fields(
    row: dict[str, Any],
    fields: tuple[str, ...],
    *,
    preserve_static: bool,
    repair_event_text_enabled: bool,
) -> RepairedText | None:
    for field in fields:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        if repair_event_text_enabled:
            repaired = repair_event_text(value, preserve_static=preserve_static)
            if repaired is None:
                continue
            return RepairedText(
                text=repaired.text,
                source_field=field,
                status=repaired.status,
                event_count=repaired.event_count,
                dropped_tail_lines=repaired.dropped_tail_lines,
                header_found=repaired.header_found,
                had_closing_event_tag=repaired.had_closing_event_tag,
                missing_closing_event_tag=repaired.missing_closing_event_tag,
            )
        text_value = value.strip()
        return RepairedText(
            text=text_value,
            source_field=field,
            status="raw",
            event_count=max(0, len(text_value.splitlines()) - 1),
            dropped_tail_lines=0,
            header_found=False,
            had_closing_event_tag="</event>" in text_value,
            missing_closing_event_tag="</event>" not in text_value,
        )
    return None


def paired_texts(
    *,
    row: dict[str, Any],
    view: TextView,
    repair_event_text_enabled: bool,
) -> tuple[RepairedText | None, RepairedText | None]:
    preserve_static = preserve_static_for_view(view)
    return (
        text_from_fields(
            row,
            view.reference_repair_fields,
            preserve_static=preserve_static,
            repair_event_text_enabled=repair_event_text_enabled,
        ),
        text_from_fields(
            row,
            view.generated_repair_fields,
            preserve_static=preserve_static,
            repair_event_text_enabled=repair_event_text_enabled,
        ),
    )


def status_counts(rows: list[dict[str, Any]], status_field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get(status_field, "missing_status") or "missing_status")
        counts[status] = counts.get(status, 0) + 1
    return counts


def validation_for_view(
    *,
    rows: list[dict[str, Any]],
    item: InputItem,
    task_type: str,
    view: TextView,
    valid_field: str,
    status_field: str,
    repair_event_text_enabled: bool,
) -> dict[str, Any]:
    valid_count = sum(1 for row in rows if bool_value(row.get(valid_field)))
    counts = status_counts(rows, status_field)
    included = 0
    good_count = 0
    repaired_count = 0
    repair_failed = 0
    missing_closing = 0
    dropped_tail = 0
    dropped_tail_lines = 0
    for row in rows:
        reference, generated = paired_texts(
            row=row,
            view=view,
            repair_event_text_enabled=repair_event_text_enabled,
        )
        if reference is None or generated is None:
            repair_failed += 1
            continue
        included += 1
        if generated.status == "good":
            good_count += 1
        elif generated.status == "repaired":
            repaired_count += 1
        if generated.missing_closing_event_tag:
            missing_closing += 1
        if reference.dropped_tail_lines or generated.dropped_tail_lines:
            dropped_tail += 1
            dropped_tail_lines += reference.dropped_tail_lines + generated.dropped_tail_lines
    return {
        "task_type": task_type,
        "run_name": item.metadata.get("run_name", ""),
        "source_label": item.metadata.get("source_label", ""),
        "temperature": item.metadata.get("temperature", ""),
        "temp_tag": item.metadata.get("temp_tag", ""),
        "fraction": item.metadata.get("fraction", ""),
        "fraction_tag": item.metadata.get("fraction_tag", ""),
        "source_path": str(item.source_path),
        "text_view": view.name,
        "total_rows": len(rows),
        "valid_count": valid_count,
        "complete_count": counts.get("complete", 0),
        "truncated_count": counts.get("truncated", 0),
        "invalid_malformed_count": counts.get("invalid_malformed", 0),
        "good_count": good_count,
        "repaired_count": repaired_count,
        "included_count": included,
        "repair_failed_count": repair_failed,
        "missing_closing_event_tag_count": missing_closing,
        "dropped_tail_count": dropped_tail,
        "repair_dropped_tail_line_count": dropped_tail_lines,
        "status_counts": json.dumps(counts, sort_keys=True),
    }


def eligible_texts(
    *,
    rows: list[dict[str, Any]],
    view: TextView,
    repair_event_text_enabled: bool,
) -> tuple[list[str], list[str]]:
    references: list[str] = []
    generated: list[str] = []
    for row in rows:
        reference, synthetic = paired_texts(
            row=row,
            view=view,
            repair_event_text_enabled=repair_event_text_enabled,
        )
        if reference is None or synthetic is None:
            continue
        references.append(reference.text)
        generated.append(synthetic.text)
    return references, generated


def prepare_view_inputs(
    *,
    rows_by_path: dict[Path, list[dict[str, Any]]],
    items: list[InputItem],
    task_type: str,
    text_views: list[TextView],
    valid_field: str,
    status_field: str,
    repair_event_text_enabled: bool,
) -> tuple[
    dict[tuple[Path, str], tuple[dict[str, Any], list[str], list[str]]],
    list[dict[str, Any]],
]:
    prepared: dict[tuple[Path, str], tuple[dict[str, Any], list[str], list[str]]] = {}
    validation_rows: list[dict[str, Any]] = []
    for item in items:
        rows = rows_by_path[item.source_path]
        for view in text_views:
            validation = validation_for_view(
                rows=rows,
                item=item,
                task_type=task_type,
                view=view,
                valid_field=valid_field,
                status_field=status_field,
                repair_event_text_enabled=repair_event_text_enabled,
            )
            references, generated = eligible_texts(
                rows=rows,
                view=view,
                repair_event_text_enabled=repair_event_text_enabled,
            )
            prepared[(item.source_path, view.name)] = (validation, references, generated)
            validation_rows.append(validation)
    return prepared, validation_rows


def metric_row(
    *,
    item: InputItem,
    task_type: str,
    view: TextView,
    spec: EmbeddingSpec,
    validation: dict[str, Any],
    value: float,
    reference_count: int,
    generated_count: int,
) -> dict[str, Any]:
    return {
        "metric": "sliced_wasserstein",
        "value": value,
        "reference_count": int(reference_count),
        "generated_count": int(generated_count),
        "task_type": task_type,
        "run_name": item.metadata.get("run_name", ""),
        "source_label": item.metadata.get("source_label", ""),
        "temperature": item.metadata.get("temperature", ""),
        "temp_tag": item.metadata.get("temp_tag", ""),
        "fraction": item.metadata.get("fraction", ""),
        "fraction_tag": item.metadata.get("fraction_tag", ""),
        "text_view": view.name,
        "embedding_name": spec.name,
        "embedding_model": str(spec.model_path),
        "embedding_max_length": int(spec.max_length),
        "source_path": str(item.source_path),
        "valid_count": int(validation["valid_count"]),
        "complete_count": int(validation["complete_count"]),
        "truncated_count": int(validation["truncated_count"]),
        "invalid_malformed_count": int(validation["invalid_malformed_count"]),
        "good_count": int(validation["good_count"]),
        "repaired_count": int(validation["repaired_count"]),
        "included_count": int(validation["included_count"]),
        "repair_failed_count": int(validation["repair_failed_count"]),
    }


def write_grouped_outputs(output_root: Path, rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> None:
    write_csv(output_root / "combined_distribution_metrics.csv", rows, METRIC_FIELDS)
    write_csv(output_root / "combined_sliced_wd_metrics.csv", rows, METRIC_FIELDS)
    write_csv(output_root / "input_validation.csv", validation_rows, VALIDATION_FIELDS)
    run_names = {str(row.get("run_name") or "run") for row in rows}
    run_names.update(str(row.get("run_name") or "run") for row in validation_rows)
    for run_name in sorted(run_names):
        run_root = output_root / safe_path_part(run_name)
        run_rows = [row for row in rows if str(row.get("run_name") or "run") == run_name]
        run_validation = [row for row in validation_rows if str(row.get("run_name") or "run") == run_name]
        write_csv(run_root / "combined_distribution_metrics.csv", run_rows, METRIC_FIELDS)
        write_csv(run_root / "combined_sliced_wd_metrics.csv", run_rows, METRIC_FIELDS)
        write_csv(run_root / "input_validation.csv", run_validation, VALIDATION_FIELDS)
        write_json(
            run_root / "run_summary.json",
            {
                "run_name": run_name,
                "metric": "sliced_wasserstein",
                "metric_rows": len(run_rows),
                "validation_rows": len(run_validation),
                "total_included_count": sum(int(row["included_count"]) for row in run_validation),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--task-type", choices=["synthetic", "memorization"], default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--embedding-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--embedding-names", type=str, default=None)
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--source-path-column", type=str, default="source_path")
    parser.add_argument("--valid-field", type=str, default="valid")
    parser.add_argument("--status-field", type=str, default="status")
    parser.add_argument("--repair-event-text", action="store_true", default=None)
    parser.add_argument("--no-repair-event-text", action="store_false", dest="repair_event_text")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--expected-input-count", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-projections", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.config is None:
        cfg = {}
    else:
        from synthetic_time_series_clean.common.config import load_config

        cfg = load_config(args.config)
    started_at = time.perf_counter()
    work_dir = args.work_dir.resolve()
    task_type = choose(args.task_type, cfg, "sliced_wd", "task_type")
    manifest = choose(args.manifest, cfg, "input", "manifest")
    embedding_manifest = choose(args.embedding_manifest, cfg, "input", "embedding_manifest")
    output_root_value = choose(args.output_root, cfg, "output", "output_root")
    embedding_names = choose(args.embedding_names, cfg, "sliced_wd", "embedding_names")
    repair_event_text_enabled = bool(
        choose(args.repair_event_text, cfg, "sliced_wd", "repair_event_text", default=True)
    )
    device_arg = str(choose(args.device, cfg, "runtime", "device", default="auto"))
    max_rows = choose(args.max_rows, cfg, "sliced_wd", "max_rows")
    n_projections = int(choose(args.n_projections, cfg, "sliced_wd", "n_projections", default=128))
    seed = int(choose(args.seed, cfg, "sliced_wd", "seed", default=42))
    if task_type not in {"synthetic", "memorization"}:
        raise ValueError("--task-type is required and must be synthetic or memorization")
    if manifest is None:
        raise ValueError("--manifest is required, either on the CLI or in input.manifest")
    if embedding_manifest is None:
        raise ValueError("--embedding-manifest is required, either on the CLI or in input.embedding_manifest")
    if output_root_value is None:
        raise ValueError("--output-root is required, either on the CLI or in output.output_root")
    output_root = resolve_path(output_root_value, work_dir)
    manifest_path = resolve_path(manifest, work_dir)
    embedding_manifest_path = resolve_path(embedding_manifest, work_dir)
    if not embedding_manifest_path.exists():
        raise FileNotFoundError(
            f"Embedding manifest is required: {embedding_manifest_path}. "
            "The clean WD script never auto-creates this file."
        )

    all_items = load_input_manifest(manifest_path, work_dir, args.source_path_column)
    filters = parse_manifest_filters(args.filter)
    items = apply_manifest_filters(all_items, filters)
    if args.expected_input_count is not None and len(items) != int(args.expected_input_count):
        raise RuntimeError(f"Expected {args.expected_input_count} inputs, found {len(items)}")
    if not items:
        raise RuntimeError("No manifest inputs selected.")
    specs = load_embedding_specs(embedding_manifest_path, work_dir, embedding_names)
    text_views = default_text_views(str(task_type))

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "selected_inputs.csv", [item.metadata for item in items])
    setup = {
        "metric": "sliced_wasserstein",
        "task_type": task_type,
        "work_dir": str(work_dir),
        "manifest": str(manifest_path),
        "embedding_manifest": str(embedding_manifest_path),
        "output_root": str(output_root),
        "input_files": len(items),
        "manifest_input_files_before_filter": len(all_items),
        "manifest_filters": {key: sorted(values) for key, values in filters.items()},
        "repair_event_text": repair_event_text_enabled,
        "embedding_names": [spec.name for spec in specs],
        "text_views": [view.__dict__ for view in text_views],
        "n_projections": n_projections,
        "seed": seed,
        "max_rows": max_rows,
        "dry_run": bool(args.dry_run),
    }
    write_json(output_root / "setup.json", setup)
    if args.dry_run:
        write_json(output_root / "run_summary.json", {**setup, "is_complete": False})
        print(json.dumps({**setup, "is_complete": False}, indent=2, default=str))
        return

    metric_rows: list[dict[str, Any]] = []
    import numpy as np

    from synthetic_time_series_clean.evaluation.embedding_metrics import (
        embed_texts,
        load_embedding_model,
        sliced_wasserstein_distance,
    )

    rows_by_path = {
        item.source_path: read_jsonl(item.source_path, max_rows=max_rows)
        for item in items
    }
    prepared_view_inputs, validation_rows = prepare_view_inputs(
        rows_by_path=rows_by_path,
        items=items,
        task_type=str(task_type),
        text_views=text_views,
        valid_field=args.valid_field,
        status_field=args.status_field,
        repair_event_text_enabled=repair_event_text_enabled,
    )

    for spec in specs:
        tokenizer, model, device = load_embedding_model(spec, device_arg)
        try:
            for item in items:
                for view in text_views:
                    validation, references, generated = prepared_view_inputs[(item.source_path, view.name)]
                    if not references:
                        continue
                    reference_embeddings = embed_texts(
                        texts=references,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        batch_size=spec.batch_size,
                        max_length=spec.max_length,
                        desc=f"{spec.name} {item.run_name}/{view.name}/reference",
                    )
                    generated_embeddings = embed_texts(
                        texts=generated,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        batch_size=spec.batch_size,
                        max_length=spec.max_length,
                        desc=f"{spec.name} {item.run_name}/{view.name}/generated",
                    )
                    value = sliced_wasserstein_distance(
                        reference_embeddings,
                        generated_embeddings,
                        n_projections=n_projections,
                        seed=seed,
                    )
                    metric_rows.append(
                        metric_row(
                            item=item,
                            task_type=str(task_type),
                            view=view,
                            spec=spec,
                            validation=validation,
                            value=value,
                            reference_count=len(references),
                            generated_count=len(generated),
                        )
                    )
                    if args.save_embeddings:
                        out_dir = (
                            output_root
                            / safe_path_part(item.run_name)
                            / safe_path_part(spec.name)
                            / safe_path_part(view.name)
                        )
                        out_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            out_dir / "embeddings.npz",
                            reference_embeddings=reference_embeddings.astype(np.float32, copy=False),
                            generated_embeddings=generated_embeddings.astype(np.float32, copy=False),
                        )
        finally:
            del model
            del tokenizer
            try:
                import torch

                if getattr(device, "type", "") == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

    write_grouped_outputs(output_root, metric_rows, validation_rows)
    summary = {
        **setup,
        "is_complete": True,
        "metric_rows": len(metric_rows),
        "validation_rows": len(validation_rows),
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    write_json(output_root / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
