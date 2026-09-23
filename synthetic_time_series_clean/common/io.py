"""Small JSON/CSV I/O helpers used by all pipeline stages."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from synthetic_time_series_clean.common.paths import ensure_dir


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = ordered_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(row.get(key)) for key in fieldnames})


def ordered_fieldnames(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    names: list[str] = []
    for key in preferred or []:
        if key not in names:
            names.append(key)
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def csv_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=json_default)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value
