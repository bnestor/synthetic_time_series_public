"""Shared generation helpers for the clean vLLM generation path."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import polars as pl
from omegaconf import OmegaConf
from tqdm import tqdm

from synthetic_time_series_clean.common.config import require_eval_dataset
from synthetic_time_series_clean.common.io import write_json, write_jsonl
from synthetic_time_series_clean.common.paths import ensure_dir
from synthetic_time_series_clean.data.rendering import (
    SOURCE_ROW_INDEX,
    load_dataset_partitions,
    prepare_mimic_filtered_df,
    render_mimic_patient_components_from_df,
    render_patient_text,
)
from synthetic_time_series_clean.data.splits import assert_no_split_overlap_from_cfg
from synthetic_time_series_clean.inference.parsing import build_rendered_row


def register_hydra_resolver_for_saved_config(resolved_config_path: Path) -> None:
    try:
        OmegaConf.register_new_resolver(
            "hydra",
            lambda key: str(resolved_config_path.parent) if key == "run.dir" else "",
            replace=True,
        )
    except Exception:
        pass


def clone_cfg_for_eval(cfg: Any, max_eval_patients: int | None, max_train_patients: int | None) -> Any:
    cloned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "training" not in cloned:
        cloned.training = {}
    cloned.training.max_train_patients = int(max_train_patients) if max_train_patients is not None else None
    cloned.training.max_eval_patients = int(max_eval_patients) if max_eval_patients is not None else None
    return cloned


def load_real_eval_rows(
    *,
    resolved_config_path: Path,
    eval_dataset: str,
    max_eval_patients: int | None,
    max_train_patients: int | None,
    random_sample_eval: bool = False,
    seed: int = 42,
    run_split_sanity: bool = True,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    eval_dataset = require_eval_dataset(eval_dataset)
    register_hydra_resolver_for_saved_config(resolved_config_path)
    cfg = OmegaConf.load(resolved_config_path)
    OmegaConf.resolve(cfg)
    if run_split_sanity:
        assert_no_split_overlap_from_cfg(cfg)
    cfg = clone_cfg_for_eval(
        cfg,
        max_eval_patients=None,
        max_train_patients=max_train_patients,
    )
    _, _, eval_sources, eval_ids = load_dataset_partitions(cfg, eval_split=eval_dataset)

    eval_ids = list(eval_ids)
    total_eval_id_count = len(eval_ids)
    if max_eval_patients is not None:
        sample_size = min(max(0, int(max_eval_patients)), len(eval_ids))
        if random_sample_eval:
            rng = random.Random(int(seed))
            eval_ids = rng.sample(eval_ids, sample_size)
            eval_selection_order = "seeded_random_eval_ids_from_full_split"
        else:
            eval_ids = eval_ids[:sample_size]
            eval_selection_order = "head_eval_ids_from_full_split"
    elif random_sample_eval:
        rng = random.Random(int(seed))
        rng.shuffle(eval_ids)
        eval_selection_order = "seeded_random_all_eval_ids_from_full_split"
    else:
        eval_selection_order = "all_eval_ids_in_split_order"
    if not eval_ids:
        raise RuntimeError(f"Eval split is empty for {resolved_config_path}")

    rows_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    grouped_requests: list[tuple[Any, list[Any]]] = []
    source_membership_counts = {key: 0 for key in eval_ids}
    for source in eval_sources:
        source_ids = [
            id_value
            for id_column, id_value in eval_ids
            if id_column == source.id_column and source.contains(id_value)
        ]
        if source_ids:
            grouped_requests.append((source, source_ids))
            for id_value in source_ids:
                source_membership_counts[(source.id_column, id_value)] += 1

    has_cross_shard_patients = any(count > 1 for count in source_membership_counts.values())
    if grouped_requests and not has_cross_shard_patients:
        shard_bar = tqdm(grouped_requests, desc=f"render {eval_dataset} shards")
        total_bar = tqdm(total=len(eval_ids), desc=f"render {eval_dataset} patients")
        for source, source_ids in shard_bar:
            shard_bar.set_postfix(shard=source.data_path.name, patients=len(source_ids))
            filtered_df = (
                pl.scan_parquet(str(source.data_path))
                .with_row_index(SOURCE_ROW_INDEX)
                .filter(pl.col(source.id_column).is_in(source_ids))
                .collect()
            )
            filtered_df = prepare_mimic_filtered_df(filtered_df, source)
            if filtered_df.height == 0:
                raise RuntimeError(f"No rows found in {source.data_path} for requested eval patients.")
            for patient_df in filtered_df.partition_by(source.id_column, maintain_order=True):
                patient_id = patient_df[source.id_column][0]
                rendered = render_mimic_patient_components_from_df(
                    filtered_df=patient_df,
                    source=source,
                    id_value=patient_id,
                    max_rows=int(cfg.max_rows),
                )
                rows_by_key[(source.id_column, patient_id)] = build_rendered_row(
                    id_column=source.id_column,
                    id_value=patient_id,
                    original_full_text=rendered.normalized_text,
                )
                total_bar.update(1)
        total_bar.close()
    else:
        for id_column, id_value in tqdm(eval_ids, desc=f"render {eval_dataset} patients"):
            text = render_patient_text(
                data_sources=eval_sources,
                id_column=id_column,
                id_value=id_value,
                max_rows=int(cfg.max_rows),
            )
            rows_by_key[(id_column, id_value)] = build_rendered_row(
                id_column=id_column,
                id_value=id_value,
                original_full_text=text,
            )

    ordered_rows: list[dict[str, Any]] = []
    missing: list[tuple[str, Any]] = []
    for key in eval_ids:
        if key in rows_by_key:
            ordered_rows.append(rows_by_key[key])
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(f"Missing rendered eval rows, first few: {missing[:5]}")
    metadata = {
        "eval_dataset": eval_dataset,
        "total_eval_id_count": int(total_eval_id_count),
        "selected_eval_id_count": int(len(eval_ids)),
        "max_eval_patients": int(max_eval_patients) if max_eval_patients is not None else None,
        "random_sample_eval": bool(random_sample_eval),
        "eval_selection_order": eval_selection_order,
        "eval_sampling_seed": int(seed),
    }
    return cfg, ordered_rows, metadata


def generation_rates(status_counts: dict[str, int], total_attempts: int) -> dict[str, Any]:
    complete = int(status_counts.get("complete", 0))
    truncated = int(status_counts.get("truncated", 0))
    invalid = int(status_counts.get("invalid_malformed", 0))
    parseable = complete + truncated
    denominator = int(total_attempts)
    return {
        "complete_generations": complete,
        "truncated_generations": truncated,
        "parseable_generations": parseable,
        "invalid_malformed_generations": invalid,
        "complete_generation_rate": complete / denominator if denominator else math.nan,
        "parseable_generation_rate": parseable / denominator if denominator else math.nan,
        "truncated_generation_rate": truncated / denominator if denominator else math.nan,
        "invalid_generation_rate": invalid / denominator if denominator else math.nan,
    }


def save_generation_outputs(
    *,
    output_dir: Path,
    samples_filename: str,
    args: Any,
    run_spec: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    ensure_dir(output_dir)
    sample_rows = sorted(
        sample_rows,
        key=lambda row: (
            str(row.get("mode", "")),
            str(row.get("run_id", "")),
            int(row.get("pair_index", 0)),
            int(row.get("sample_index", 0)),
        ),
    )
    samples_path = output_dir / samples_filename
    examples_path = output_dir / samples_filename.replace(".jsonl", "_examples.jsonl")
    write_jsonl(samples_path, sample_rows)
    write_jsonl(examples_path, sample_rows[:5])
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "args.json", vars(args))
    write_json(output_dir / "run_spec.json", {key: str(value) for key, value in run_spec.items()})
