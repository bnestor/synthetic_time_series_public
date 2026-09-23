"""MEDS/MIMIC loading and deterministic text rendering.

The renderer deliberately uses the configured ``time_column`` everywhere. It
also keeps a stable source-row index and sorts event rows by
``[time_column, _source_shard_index, _source_row_index]`` when patients span
multiple shards, so rows with the same timestamp have a deterministic order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import polars as pl

from synthetic_time_series_clean.common.config import configured_split_name
from synthetic_time_series_clean.common.paths import resolve_path


_NUMERIC_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
SOURCE_ROW_INDEX = "_source_row_index"
SOURCE_SHARD_INDEX = "_source_shard_index"


@dataclass
class MIMICShardSource:
    data_path: Path
    id_column: str
    time_column: str
    identifiers: set[Any]
    codes: pl.DataFrame | None = None

    def contains(self, id_value: Any) -> bool:
        return id_value in self.identifiers


@dataclass(frozen=True)
class RenderedPatient:
    static_body: str
    event_csv: str
    normalized_text: str
    prompt_prefix: str
    event_count: int


def _normalize_rendered_text_cell(value: Any) -> str | None:
    if value is None:
        return None
    return (
        str(value)
        .replace("\r\n", "<newline>")
        .replace("\n", "<newline>")
        .replace("\r", "<newline>")
        .replace(",", "<comma>")
    )


def _normalize_rendered_numeric_cell(value: Any) -> str | None:
    if value is None:
        return None
    text = _NUMERIC_THOUSANDS_COMMA_RE.sub("", str(value))
    return (
        text.replace("\r\n", "<newline>")
        .replace("\n", "<newline>")
        .replace("\r", "<newline>")
        .replace(",", "<comma>")
    )


def looks_like_mimic_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    for relative in ("data/train", "merge_to_MEDS_cohort/train", "train"):
        if (path / relative).exists():
            return True
    return False


def data_item_from_cfg(cfg: Any) -> Any:
    items = list(cfg.data.data_items)
    if len(items) != 1 or "path" not in items[0]:
        raise ValueError("Configure exactly one MEDS/MIMIC dataset root in data.data_items[0].path.")
    return items[0]


def resolve_mimic_root_and_columns(cfg: Any) -> tuple[Path, str, str]:
    item = data_item_from_cfg(cfg)
    root = resolve_path(str(item.path))
    if not looks_like_mimic_root(root):
        raise ValueError(f"Configured path does not look like a MEDS/MIMIC root: {root}")
    id_column = str(item.get("identifier_column", "subject_id"))
    time_column = str(item.get("time_column", "time"))
    return root, id_column, time_column


def resolve_mimic_split_dir(root: Path, split: str) -> Path:
    split = str(split).strip()
    for relative_dir in ("data", "merge_to_MEDS_cohort"):
        candidate = root / relative_dir / split
        if candidate.exists():
            return candidate
    candidate = root / split
    if candidate.exists():
        return candidate
    raise ValueError(
        f"Could not find split {split!r} under {root}; expected data/{split}, "
        f"merge_to_MEDS_cohort/{split}, or flat {split}/."
    )


def resolve_mimic_codes_path(root: Path) -> Path | None:
    for relative_path in ("metadata/codes.parquet", "extract_code_metadata/codes.parquet"):
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None


def _load_split_sources(
    *,
    root: Path,
    id_column: str,
    time_column: str,
    split: str,
    codes: pl.DataFrame | None,
) -> tuple[list[MIMICShardSource], list[tuple[str, Any]]]:
    split_dir = resolve_mimic_split_dir(root, split)
    parquet_paths = sorted(split_dir.glob("*.parquet"), key=lambda item: item.name)
    sources: list[MIMICShardSource] = []
    identifiers: list[tuple[str, Any]] = []
    for path in parquet_paths:
        lazy = pl.scan_parquet(str(path))
        columns = lazy.collect_schema().names()
        if id_column not in columns:
            raise RuntimeError(f"{path} is missing configured id_column={id_column!r}.")
        if time_column not in columns:
            raise RuntimeError(f"{path} is missing configured time_column={time_column!r}.")
        shard_ids = (
            lazy.select(pl.col(id_column))
            .drop_nulls()
            .unique()
            .collect()
            .to_series()
            .to_list()
        )
        if not shard_ids:
            continue
        source = MIMICShardSource(
            data_path=path,
            id_column=id_column,
            time_column=time_column,
            identifiers=set(shard_ids),
            codes=codes,
        )
        sources.append(source)
        identifiers.extend((id_column, value) for value in shard_ids)
    unique_ids = sorted(set(identifiers), key=lambda pair: (pair[0], str(pair[1])))
    return sources, unique_ids


def cap_identifiers(identifiers: list[tuple[str, Any]], limit: int | None) -> list[tuple[str, Any]]:
    if limit is None:
        return identifiers
    return identifiers[: max(0, int(limit))]


def load_split_sources(
    *,
    cfg: Any,
    split: str,
    max_patients: int | None = None,
) -> tuple[list[MIMICShardSource], list[tuple[str, Any]]]:
    root, id_column, time_column = resolve_mimic_root_and_columns(cfg)
    codes_path = resolve_mimic_codes_path(root)
    codes = pl.read_parquet(str(codes_path)) if codes_path is not None else None
    split_name = configured_split_name(cfg, split)
    sources, ids = _load_split_sources(
        root=root,
        id_column=id_column,
        time_column=time_column,
        split=split_name,
        codes=codes,
    )
    return sources, cap_identifiers(ids, max_patients)


def load_dataset_partitions(
    cfg: Any,
    eval_split: str = "tuning",
) -> tuple[
    list[MIMICShardSource],
    list[tuple[str, Any]],
    list[MIMICShardSource],
    list[tuple[str, Any]],
]:
    root, id_column, time_column = resolve_mimic_root_and_columns(cfg)
    codes_path = resolve_mimic_codes_path(root)
    codes = pl.read_parquet(str(codes_path)) if codes_path is not None else None
    train_split = configured_split_name(cfg, "train")
    eval_split_name = configured_split_name(cfg, eval_split)
    train_sources, train_ids = _load_split_sources(
        root=root,
        id_column=id_column,
        time_column=time_column,
        split=train_split,
        codes=codes,
    )
    eval_sources, eval_ids = _load_split_sources(
        root=root,
        id_column=id_column,
        time_column=time_column,
        split=eval_split_name,
        codes=codes,
    )
    training_cfg = cfg.get("training", {})
    return (
        train_sources,
        cap_identifiers(train_ids, training_cfg.get("max_train_patients")),
        eval_sources,
        cap_identifiers(eval_ids, training_cfg.get("max_eval_patients")),
    )


def find_source(
    sources: list[MIMICShardSource],
    id_column: str,
    id_value: Any,
) -> MIMICShardSource | None:
    for source in sources:
        if source.id_column == id_column and source.contains(id_value):
            return source
    return None


def find_sources(
    sources: list[MIMICShardSource],
    id_column: str,
    id_value: Any,
) -> list[MIMICShardSource]:
    return [source for source in sources if source.id_column == id_column and source.contains(id_value)]


def prepare_mimic_filtered_df(filtered_df: pl.DataFrame, source: MIMICShardSource) -> pl.DataFrame:
    if SOURCE_ROW_INDEX not in filtered_df.columns:
        filtered_df = filtered_df.with_row_index(SOURCE_ROW_INDEX)
    if "code" in filtered_df.columns:
        filtered_df = filtered_df.filter(pl.col("code").is_not_null())
    if filtered_df.height == 0:
        return filtered_df
    if "numeric_value" not in filtered_df.columns:
        filtered_df = filtered_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("numeric_value"))
    elif "unit" in filtered_df.columns:
        filtered_df = filtered_df.with_columns(
            pl.when(pl.col("unit").is_not_null() & pl.col("numeric_value").is_not_null())
            .then(pl.col("numeric_value").cast(pl.Utf8) + " " + pl.col("unit").cast(pl.Utf8))
            .otherwise(pl.col("numeric_value").cast(pl.Utf8))
            .alias("numeric_value")
        )
    else:
        filtered_df = filtered_df.with_columns(pl.col("numeric_value").cast(pl.Utf8).alias("numeric_value"))

    if source.codes is not None and "code" in filtered_df.columns and "code" in source.codes.columns:
        filtered_df = filtered_df.join(source.codes, on="code", how="left")
        if "description" in filtered_df.columns:
            filtered_df = filtered_df.with_columns(
                pl.when(pl.col("description").is_not_null() & pl.col("code").is_not_null())
                .then(pl.col("description").cast(pl.Utf8))
                .otherwise(pl.col("code").cast(pl.Utf8))
                .alias("code")
            )
    if "code" not in filtered_df.columns:
        filtered_df = filtered_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("code"))
    if "ordercategorydescription" in filtered_df.columns:
        filtered_df = filtered_df.with_columns(
            pl.when(pl.col("ordercategorydescription").is_not_null() & pl.col("code").is_not_null())
            .then(pl.col("ordercategorydescription").cast(pl.Utf8) + ": " + pl.col("code").cast(pl.Utf8))
            .otherwise(pl.col("code").cast(pl.Utf8))
            .alias("code")
        )
    if "statusdescription" in filtered_df.columns:
        filtered_df = filtered_df.with_columns(
            pl.when(pl.col("statusdescription").is_not_null() & pl.col("code").is_not_null())
            .then(pl.col("code").cast(pl.Utf8) + " (" + pl.col("statusdescription").cast(pl.Utf8) + ")")
            .otherwise(pl.col("code").cast(pl.Utf8))
            .alias("code")
        )
    if source.time_column not in filtered_df.columns:
        raise RuntimeError(
            f"MEDS/MIMIC shard {source.data_path} is missing configured time column "
            f"{source.time_column!r}."
        )
    return filtered_df


def render_mimic_patient_components_from_df(
    *,
    filtered_df: pl.DataFrame,
    source: MIMICShardSource,
    id_value: Any,
    max_rows: int,
) -> RenderedPatient:
    del id_value
    if filtered_df.height == 0:
        raise RuntimeError(f"No rows found in {source.data_path}")
    if SOURCE_ROW_INDEX not in filtered_df.columns:
        filtered_df = filtered_df.with_row_index(SOURCE_ROW_INDEX)
    time_column = source.time_column
    if "code" in filtered_df.columns:
        filtered_df = filtered_df.with_columns(
            pl.col("code").map_elements(_normalize_rendered_text_cell, return_dtype=pl.Utf8).alias("code")
        )
    if "numeric_value" in filtered_df.columns:
        filtered_df = filtered_df.with_columns(
            pl.col("numeric_value")
            .map_elements(_normalize_rendered_numeric_cell, return_dtype=pl.Utf8)
            .alias("numeric_value")
        )

    static_df = filtered_df.filter(pl.col(time_column).is_null())
    description = ""
    if static_df.height > 0:
        static_sort_columns: list[str] = []
        if SOURCE_SHARD_INDEX in static_df.columns:
            static_sort_columns.append(SOURCE_SHARD_INDEX)
        if SOURCE_ROW_INDEX in static_df.columns:
            static_sort_columns.append(SOURCE_ROW_INDEX)
        if static_sort_columns:
            static_df = static_df.sort(static_sort_columns)
        static_df = static_df.with_columns(
            pl.when(pl.col("code").is_not_null() & pl.col("numeric_value").is_not_null())
            .then(pl.col("code").cast(pl.Utf8) + "==" + pl.col("numeric_value").cast(pl.Utf8))
            .otherwise(pl.col("code").cast(pl.Utf8))
            .alias("description")
        ).select("description")
        description = " ".join(value for value in static_df["description"].to_list() if value is not None)

    sort_columns = [time_column]
    select_columns = [time_column, "code", "numeric_value", SOURCE_ROW_INDEX]
    if SOURCE_SHARD_INDEX in filtered_df.columns:
        sort_columns.append(SOURCE_SHARD_INDEX)
        select_columns.append(SOURCE_SHARD_INDEX)
    sort_columns.append(SOURCE_ROW_INDEX)
    timeseries_df = (
        filtered_df.select(select_columns)
        .filter(pl.col(time_column).is_not_null())
        .sort(sort_columns)
        .rename({time_column: "time"})
        .select(["time", "code", "numeric_value"])
    )
    if timeseries_df.height == 0:
        raise RuntimeError(f"No time-series rows found for {source.id_column}.")
    if timeseries_df.height > int(max_rows):
        timeseries_df = timeseries_df.slice(0, int(max_rows))
    with StringIO() as buffer:
        timeseries_df.write_csv(buffer)
        events_text = buffer.getvalue().strip()
    prompt_prefix = f"<static>{description}</static><event>"
    return RenderedPatient(
        static_body=description,
        event_csv=events_text,
        normalized_text=f"{prompt_prefix}{events_text}</event>",
        prompt_prefix=prompt_prefix,
        event_count=int(timeseries_df.height),
    )


def render_mimic_patient_text_for_source(
    *,
    source: MIMICShardSource,
    id_value: Any,
    max_rows: int,
) -> str:
    filtered_df = (
        pl.scan_parquet(str(source.data_path))
        .with_row_index(SOURCE_ROW_INDEX)
        .filter(pl.col(source.id_column) == id_value)
        .collect()
    )
    filtered_df = prepare_mimic_filtered_df(filtered_df, source)
    rendered = render_mimic_patient_components_from_df(
        filtered_df=filtered_df,
        source=source,
        id_value=id_value,
        max_rows=max_rows,
    )
    return rendered.normalized_text


def render_mimic_patient_text_for_sources(
    *,
    sources: list[MIMICShardSource],
    id_value: Any,
    max_rows: int,
) -> str:
    if not sources:
        raise RuntimeError(f"No MEDS/MIMIC shard found for patient {id_value!r}")
    frames: list[pl.DataFrame] = []
    for shard_index, source in enumerate(sources):
        filtered_df = (
            pl.scan_parquet(str(source.data_path))
            .with_row_index(SOURCE_ROW_INDEX)
            .filter(pl.col(source.id_column) == id_value)
            .collect()
        )
        if filtered_df.height == 0:
            continue
        filtered_df = filtered_df.with_columns(pl.lit(int(shard_index)).alias(SOURCE_SHARD_INDEX))
        filtered_df = prepare_mimic_filtered_df(filtered_df, source)
        if filtered_df.height > 0:
            frames.append(filtered_df)
    if not frames:
        first = sources[0]
        raise RuntimeError(f"No rows found in any MEDS/MIMIC shard for {first.id_column}={id_value!r}")
    combined_df = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
    rendered = render_mimic_patient_components_from_df(
        filtered_df=combined_df,
        source=sources[0],
        id_value=id_value,
        max_rows=max_rows,
    )
    return rendered.normalized_text


def render_patient_text(
    *,
    data_sources: list[MIMICShardSource],
    id_column: str,
    id_value: Any,
    max_rows: int,
) -> str:
    sources = find_sources(data_sources, id_column=id_column, id_value=id_value)
    if not sources:
        raise RuntimeError(f"No MEDS/MIMIC shard found for {id_column}={id_value!r}")
    if len(sources) == 1:
        return render_mimic_patient_text_for_source(source=sources[0], id_value=id_value, max_rows=max_rows)
    return render_mimic_patient_text_for_sources(sources=sources, id_value=id_value, max_rows=max_rows)


def render_many_patient_rows(
    *,
    sources: list[MIMICShardSource],
    identifiers: list[tuple[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_index, (id_column, id_value) in enumerate(identifiers):
        text = render_patient_text(
            data_sources=sources,
            id_column=id_column,
            id_value=id_value,
            max_rows=max_rows,
        )
        rows.append(
            {
                "pair_index": int(pair_index),
                "id_column": id_column,
                "id_value": id_value,
                "original_full_text": text,
            }
        )
    return rows
