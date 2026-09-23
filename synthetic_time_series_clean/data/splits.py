"""Split overlap sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synthetic_time_series_clean.common.config import configured_split_name

@dataclass(frozen=True)
class SplitOverlap:
    left: str
    right: str
    count: int
    examples: list[str]


def _identifier_keys(identifiers: list[tuple[str, Any]]) -> set[str]:
    return {f"{id_column}:{id_value}" for id_column, id_value in identifiers}


def check_split_overlap(
    *,
    split_ids: dict[str, list[tuple[str, Any]]],
    max_examples: int = 10,
) -> list[SplitOverlap]:
    names = list(split_ids)
    overlaps: list[SplitOverlap] = []
    key_sets = {name: _identifier_keys(ids) for name, ids in split_ids.items()}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(key_sets[left] & key_sets[right])
            if overlap:
                overlaps.append(
                    SplitOverlap(
                        left=left,
                        right=right,
                        count=len(overlap),
                        examples=overlap[:max_examples],
                    )
                )
    return overlaps


def assert_no_split_overlap_from_cfg(
    cfg: Any,
    splits: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    from synthetic_time_series_clean.data.rendering import load_split_sources

    logical_splits = splits or ("train", "tuning", "held_out")
    split_ids: dict[str, list[tuple[str, Any]]] = {}
    split_names: dict[str, str] = {}
    for split in logical_splits:
        split_name = configured_split_name(cfg, split)
        _, identifiers = load_split_sources(cfg=cfg, split=split, max_patients=None)
        split_ids[split] = identifiers
        split_names[split] = split_name
    overlaps = check_split_overlap(split_ids=split_ids)
    if overlaps:
        details = [
            {
                "left": item.left,
                "right": item.right,
                "count": item.count,
                "examples": item.examples,
            }
            for item in overlaps
        ]
        raise RuntimeError(f"Split overlap detected: {details}")
    return {
        "splits_checked": list(logical_splits),
        "split_names": split_names,
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "overlap_count": 0,
    }
