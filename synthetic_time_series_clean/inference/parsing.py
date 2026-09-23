"""Canonical parsing for rendered and generated samples."""

from __future__ import annotations

import csv
import io
import re
from typing import Any

from synthetic_time_series_clean.common.constants import CLOSING_EVENT_TAG, EVENT_CSV_HEADER


EVENT_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$")


def _is_standard_event_header(row: list[str]) -> bool:
    return [item.strip().lower() for item in row] == EVENT_CSV_HEADER.split(",")


def _is_standard_event_row(row: list[str]) -> bool:
    if len(row) != 3:
        return False
    timestamp, code, _numeric_value = [item.strip() for item in row]
    return bool(EVENT_TIME_RE.match(timestamp)) and bool(code)


def trim_to_closing_tag(text: str, prompt_prefix: str, closing_tag: str = CLOSING_EVENT_TAG) -> str:
    if prompt_prefix and prompt_prefix in text:
        text = text[text.index(prompt_prefix) :]
    else:
        static_index = text.find("<static>")
        if static_index != -1:
            text = text[static_index:]
    closing_index = text.find(closing_tag)
    if closing_index != -1:
        return text[: closing_index + len(closing_tag)]
    return text


def _truncate_csv_prefix_to_complete_rows(csv_text: str) -> str:
    normalized = str(csv_text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("CSV block is empty.")
    lines = normalized.split("\n")
    try:
        header = next(csv.reader([lines[0]]))
    except Exception as exc:
        raise ValueError("CSV header is malformed.") from exc
    if not header:
        raise ValueError("CSV header is empty.")
    if not _is_standard_event_header(header):
        raise ValueError(f"CSV header must be {EVENT_CSV_HEADER!r}.")
    kept = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            row = next(csv.reader([line]))
        except Exception:
            break
        if not _is_standard_event_row(row):
            break
        kept.append(line)
    if len(kept) < 2:
        raise ValueError("Truncated sample does not contain a complete event row.")
    return "\n".join(kept)


def extract_static_and_event_with_status(text: str) -> tuple[str, str, str]:
    complete = re.fullmatch(
        r"\s*<static>(.*?)</static>\s*<event>(.*)</event>\s*",
        str(text),
        flags=re.DOTALL,
    )
    if complete is not None:
        return complete.group(1).strip(), complete.group(2).strip(), "complete"

    truncated = re.fullmatch(
        r"\s*<static>(.*?)</static>\s*<event>(.*)\s*",
        str(text),
        flags=re.DOTALL,
    )
    if truncated is None:
        raise ValueError("Sample must contain one <static>...</static><event>...</event> block.")
    static_body = truncated.group(1).strip()
    event_csv = _truncate_csv_prefix_to_complete_rows(truncated.group(2).strip())
    return static_body, event_csv, "truncated"


def canonicalize_csv_text_with_event_count(csv_text: str) -> tuple[str, int]:
    reader = csv.reader(io.StringIO(str(csv_text)))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV block is empty.")
    header = rows[0]
    if not header:
        raise ValueError("CSV header is empty.")
    if not _is_standard_event_header(header):
        raise ValueError(f"CSV header must be {EVENT_CSV_HEADER!r}.")
    for row_index, row in enumerate(rows[1:], start=1):
        if not _is_standard_event_row(row):
            raise ValueError(
                f"CSV row {row_index} must have a timestamp, non-empty code, and numeric_value."
            )
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().rstrip("\n"), max(0, len(rows) - 1)


def parse_full_text(text: str, allow_truncated: bool) -> dict[str, Any]:
    try:
        static_body, event_csv, status = extract_static_and_event_with_status(text)
        if status == "truncated" and not allow_truncated:
            raise ValueError("Sample ended before </event>.")
        canonical_event_csv, event_count = canonicalize_csv_text_with_event_count(event_csv)
        if status == "complete":
            full_text = f"<static>{static_body}</static><event>{canonical_event_csv}</event>"
        else:
            full_text = f"<static>{static_body}</static><event>{canonical_event_csv}"
        return {
            "valid": True,
            "status": status,
            "is_complete": status == "complete",
            "error": None,
            "static": static_body,
            "events": canonical_event_csv,
            "full_text": full_text,
            "prompt_prefix": f"<static>{static_body}</static><event>",
            "event_count": int(event_count),
        }
    except Exception as exc:
        return {
            "valid": False,
            "status": "invalid_malformed",
            "is_complete": False,
            "error": str(exc),
            "static": "",
            "events": "",
            "full_text": "",
            "prompt_prefix": "",
            "event_count": 0,
        }


def build_rendered_row(*, id_column: str, id_value: Any, original_full_text: str) -> dict[str, Any]:
    parsed = parse_full_text(original_full_text, allow_truncated=False)
    if not parsed["valid"] or not parsed["is_complete"]:
        raise ValueError(
            f"Original sample should be complete but got status={parsed['status']}, "
            f"id_column={id_column}, id_value={id_value}, error={parsed['error']}"
        )
    return {
        "id_column": id_column,
        "id_value": id_value,
        "original_static": parsed["static"],
        "original_events": parsed["events"],
        "original_full_text": parsed["full_text"],
        "original_event_count": int(parsed["event_count"]),
        "prompt_prefix": parsed["prompt_prefix"],
    }


def restore_stop_token_text(
    *,
    continuation: str,
    stop_reason: Any,
    event_end_token_id: int,
    closing_tag: str = CLOSING_EVENT_TAG,
) -> tuple[str, bool]:
    if str(stop_reason) == str(event_end_token_id) and closing_tag not in continuation:
        return continuation + closing_tag, True
    return continuation, False
