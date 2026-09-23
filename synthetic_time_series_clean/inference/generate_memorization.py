"""vLLM Stage 4A memorization-risk generation from the training split."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from synthetic_time_series_clean.common.config import load_config
from synthetic_time_series_clean.common.constants import CLOSING_EVENT_TAG
from synthetic_time_series_clean.common.io import write_json, write_jsonl
from synthetic_time_series_clean.common.model_io import resolve_run_spec
from synthetic_time_series_clean.common.runtime import file_sha256, set_seed
from synthetic_time_series_clean.common.tokens import stop_token_ids
from synthetic_time_series_clean.data.rendering import load_split_sources, render_patient_text
from synthetic_time_series_clean.data.splits import assert_no_split_overlap_from_cfg
from synthetic_time_series_clean.inference.parsing import (
    parse_full_text,
    restore_stop_token_text,
    trim_to_closing_tag,
)


def _cfg(cfg: Any, section: str, key: str, default: Any = None) -> Any:
    if section in cfg and key in cfg[section]:
        return cfg[section][key]
    return default


def _choose(value: Any, cfg: Any, section: str, key: str, default: Any = None) -> Any:
    return value if value is not None else _cfg(cfg, section, key, default)


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "-1", "0"}:
        return None
    return int(value)


def parse_top_k(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "-1"}:
        return None
    return int(value)


def parse_fractions(text: str) -> list[float]:
    fractions: list[float] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if value < 0.0 or value >= 1.0:
            raise ValueError(f"Memorization fraction must be in [0, 1): {value}")
        fractions.append(value)
    if not fractions:
        raise ValueError("At least one memorization fraction is required.")
    return fractions


def strategy_name_from_fraction(fraction: float) -> str:
    return f"fraction_{int(round(float(fraction) * 100.0))}pct"


def temp_tag_from_temperature(temperature: float) -> str:
    return "0" if abs(float(temperature)) < 1e-9 else f"{int(round(float(temperature) * 10)):02d}"


def split_event_csv(event_csv: str) -> tuple[str, list[str]]:
    lines = str(event_csv).split("\n")
    if not lines or not lines[0]:
        raise ValueError("Event CSV header is empty.")
    rows = [str(line) for line in lines[1:] if str(line).strip()]
    if not rows:
        raise ValueError("No event rows available.")
    return str(lines[0]), rows


def resolve_seeded_event_count(total_events: int, fraction: float) -> int:
    if total_events <= 1:
        raise ValueError("Need at least two events to split prefix and continuation.")
    seeded = int(math.floor(int(total_events) * float(fraction)))
    seeded = max(0, seeded)
    return min(seeded, int(total_events) - 1)


def build_train_row(*, id_column: str, id_value: Any, train_full_text: str) -> dict[str, Any]:
    parsed = parse_full_text(train_full_text, allow_truncated=False)
    if not parsed["valid"] or not parsed["is_complete"]:
        raise ValueError(
            f"Train sample should be complete but got status={parsed['status']}, "
            f"id_column={id_column}, id_value={id_value}, error={parsed['error']}"
        )
    return {
        "id_column": id_column,
        "id_value": id_value,
        "train_static": parsed["static"],
        "train_events": parsed["events"],
        "train_full_text": parsed["full_text"],
        "train_event_count": int(parsed["event_count"]),
        "prompt_prefix": parsed["prompt_prefix"],
    }


def load_train_rows(
    *,
    cfg: Any,
    max_train_patients: int | None,
    random_sample_train: bool,
    train_selection_mode: str,
    seed: int,
    min_events: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_sources, train_ids = load_split_sources(cfg=cfg, split="train", max_patients=None)
    train_ids = list(train_ids)
    total_train_id_count = len(train_ids)
    if not train_ids:
        raise RuntimeError("Train split is empty for memorization generation.")

    mode = str(train_selection_mode).strip().lower()
    if mode not in {"post_filter_shuffle", "head_ids", "random_ids"}:
        raise ValueError("train_selection_mode must be post_filter_shuffle, head_ids, or random_ids")

    requested_train_ids = list(train_ids)
    if mode == "head_ids" and max_train_patients is not None:
        requested_train_ids = requested_train_ids[: max(0, int(max_train_patients))]
    elif mode == "random_ids" and max_train_patients is not None:
        rng = random.Random(int(seed))
        requested_train_ids = rng.sample(
            requested_train_ids,
            min(max(0, int(max_train_patients)), len(requested_train_ids)),
        )
    elif mode == "random_ids":
        rng = random.Random(int(seed))
        rng.shuffle(requested_train_ids)

    rows_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    render_error_count = 0
    render_error_examples: list[dict[str, Any]] = []
    for id_column, id_value in tqdm(requested_train_ids, desc="render train memorization rows"):
        try:
            train_full_text = render_patient_text(
                data_sources=train_sources,
                id_column=id_column,
                id_value=id_value,
                max_rows=int(cfg.max_rows),
            )
            row = build_train_row(id_column=id_column, id_value=id_value, train_full_text=train_full_text)
        except Exception as exc:
            render_error_count += 1
            if len(render_error_examples) < 10:
                render_error_examples.append(
                    {
                        "id_column": id_column,
                        "id_value": id_value,
                        "error": str(exc),
                    }
                )
            continue
        if int(row["train_event_count"]) >= int(min_events):
            rows_by_key[(id_column, id_value)] = row

    ordered_rows = [rows_by_key[key] for key in requested_train_ids if key in rows_by_key]
    filtered_count_before_sampling = len(ordered_rows)

    if mode == "post_filter_shuffle" and bool(random_sample_train):
        rng = random.Random(int(seed))
        rng.shuffle(ordered_rows)
    if mode == "post_filter_shuffle" and max_train_patients is not None:
        ordered_rows = ordered_rows[: max(0, int(max_train_patients))]

    if not ordered_rows:
        raise RuntimeError(
            f"No train rows left after min_events filter. min_events={min_events}, "
            f"requested={len(requested_train_ids)}"
        )

    if mode == "head_ids":
        selection_order = "head_train_ids_then_render_then_min_events_filter"
        random_effective = False
    elif mode == "random_ids":
        selection_order = "seeded_random_train_ids_then_render_then_min_events_filter"
        random_effective = True
    else:
        selection_order = "render_full_train_then_min_events_filter_then_seeded_shuffle_then_cap"
        random_effective = bool(random_sample_train)

    return ordered_rows, {
        "train_selection_mode": mode,
        "train_selection_order": selection_order,
        "total_train_id_count": int(total_train_id_count),
        "render_requested_train_id_count": int(len(requested_train_ids)),
        "render_error_count": int(render_error_count),
        "render_error_examples": render_error_examples,
        "train_rows_after_min_events_filter": int(filtered_count_before_sampling),
        "selected_train_row_count": int(len(ordered_rows)),
        "max_train_patients": int(max_train_patients) if max_train_patients is not None else None,
        "random_sample_train_requested": bool(random_sample_train),
        "random_sample_train_effective": bool(random_effective),
        "seed": int(seed),
    }


def build_selected_train_rows(
    train_rows: list[dict[str, Any]],
    *,
    run_name: str,
    temperature: float,
    temp_tag: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(train_rows):
        rows.append(
            {
                "stage": "stage_4a_memorization_risk_data_generation",
                "data_split_used": "train",
                "row_index": int(row_index),
                "run_name": run_name,
                "temperature": float(temperature),
                "temp_tag": temp_tag,
                "id_column": row["id_column"],
                "id_value": row["id_value"],
                "train_event_count": int(row["train_event_count"]),
                "train_static": row["train_static"],
                "train_events": row["train_events"],
                "train_full_text": row["train_full_text"],
            }
        )
    return rows


def build_memorization_candidate(
    *,
    row: dict[str, Any],
    row_index: int,
    candidate_index: int,
    fraction: float,
) -> dict[str, Any]:
    header, event_rows = split_event_csv(str(row["train_events"]))
    seeded = resolve_seeded_event_count(len(event_rows), float(fraction))
    prefix_rows = event_rows[:seeded]
    continuation_rows = event_rows[seeded:]
    if not continuation_rows:
        raise ValueError("Continuation split is empty while building memorization candidate.")

    seeded_block = "\n".join([header, *prefix_rows])
    continuation_event_csv = "\n".join([header, *continuation_rows])
    prompt = f"<static>{row['train_static']}</static><event>{seeded_block}\n"
    return {
        "stage": "stage_4a_memorization_risk_data_generation",
        "data_split_used": "train",
        "row_index": int(row_index),
        "candidate_index": int(candidate_index),
        "strategy_name": strategy_name_from_fraction(float(fraction)),
        "fraction": float(fraction),
        "id_column": row["id_column"],
        "id_value": row["id_value"],
        "train_event_count": int(row["train_event_count"]),
        "seeded_event_count": int(seeded),
        "target_continuation_event_count": int(len(continuation_rows)),
        "full_event_header": header,
        "prompt": prompt,
        "target_continuation_event_csv": continuation_event_csv,
        "target_continuation_normalized_text": f"<event>{continuation_event_csv}</event>",
        "train_static": row["train_static"],
        "train_events": row["train_events"],
        "train_full_text": row["train_full_text"],
    }


def build_all_candidates(*, train_rows: list[dict[str, Any]], fractions: list[float]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for fraction in fractions:
        candidate_index = 0
        for row_index, row in enumerate(tqdm(train_rows, desc=f"build candidates {strategy_name_from_fraction(fraction)}")):
            candidates.append(
                build_memorization_candidate(
                    row=row,
                    row_index=row_index,
                    candidate_index=candidate_index,
                    fraction=float(fraction),
                )
            )
            candidate_index += 1
    return candidates


def attach_run_metadata(
    rows: list[dict[str, Any]],
    *,
    run_name: str,
    temperature: float,
    temp_tag: str,
) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["run_name"] = run_name
        enriched["temperature"] = float(temperature)
        enriched["temp_tag"] = temp_tag
        enriched_rows.append(enriched)
    return enriched_rows


def compute_prompt_token_info(tokenizer: Any, prompts: list[str]) -> tuple[list[int], int]:
    if not prompts:
        return [], 0
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    finally:
        tokenizer.padding_side = original_padding_side
    counts = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    return counts, int(encoded["input_ids"].shape[1])


def resolve_effective_max_new_tokens(
    *,
    max_new_tokens: int,
    total_max_length: int | None,
    padded_prompt_token_count: int,
) -> int:
    if total_max_length is None:
        return int(max_new_tokens)
    return min(int(max_new_tokens), max(0, int(total_max_length) - int(padded_prompt_token_count)))


def continuation_from_generated_events(event_csv: str, seeded_event_count: int) -> tuple[str, int]:
    header, event_rows = split_event_csv(event_csv)
    continuation_rows = event_rows[int(seeded_event_count) :]
    return "\n".join([header, *continuation_rows]), max(0, len(continuation_rows))


def extract_generated_continuation_from_full_text(
    *,
    generated_full_text: str,
    seeded_event_count: int,
) -> dict[str, Any]:
    parsed = parse_full_text(generated_full_text, allow_truncated=True)
    if not parsed["valid"]:
        return {
            "valid": False,
            "status": "invalid_malformed",
            "is_complete": False,
            "error": parsed["error"],
            "generated_event_csv": "",
            "generated_continuation_event_csv": "",
            "generated_continuation_normalized_text": "",
            "generated_event_count": 0,
            "generated_continuation_event_count": 0,
        }
    try:
        continuation_event_csv, continuation_event_count = continuation_from_generated_events(
            str(parsed["events"]),
            int(seeded_event_count),
        )
    except ValueError as exc:
        return {
            "valid": False,
            "status": "invalid_malformed",
            "is_complete": False,
            "error": str(exc),
            "generated_event_csv": "",
            "generated_continuation_event_csv": "",
            "generated_continuation_normalized_text": "",
            "generated_event_count": 0,
            "generated_continuation_event_count": 0,
        }
    return {
        "valid": True,
        "status": parsed["status"],
        "is_complete": bool(parsed["is_complete"]),
        "error": None,
        "generated_event_csv": parsed["events"],
        "generated_continuation_event_csv": continuation_event_csv,
        "generated_continuation_normalized_text": f"<event>{continuation_event_csv}</event>",
        "generated_event_count": int(parsed["event_count"]),
        "generated_continuation_event_count": int(continuation_event_count),
    }


def build_sampling_params(
    *,
    sampling_params_cls: Any,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int | None,
    repetition_penalty: float,
    seed: int,
    stop_ids: list[int],
) -> Any:
    kwargs: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature) if bool(do_sample) else 0.0,
        "stop_token_ids": [int(token_id) for token_id in stop_ids],
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "seed": int(seed),
    }
    if bool(do_sample):
        kwargs.update(
            {
                "top_p": float(top_p),
                "top_k": int(top_k) if top_k is not None else 0,
                "repetition_penalty": float(repetition_penalty),
            }
        )
    else:
        kwargs.update({"top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0})
    return sampling_params_cls(**kwargs)


def run_memorization_generation_vllm(
    *,
    candidates: list[dict[str, Any]],
    llm: Any,
    tokenizer: Any,
    sampling_params_cls: Any,
    run_spec: dict[str, Any],
    batch_size: int,
    max_new_tokens: int,
    total_max_length: int | None,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int | None,
    repetition_penalty: float,
    seed: int,
    event_end_token_id: int,
    stop_ids: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(candidates), max(1, int(batch_size))), desc="memorization vllm generate"):
        batch = candidates[start : start + max(1, int(batch_size))]
        prompts = [str(item["prompt"]) for item in batch]
        prompt_token_counts, padded_prompt_token_count = compute_prompt_token_info(tokenizer, prompts)
        effective_max_new_tokens_by_prompt = [
            resolve_effective_max_new_tokens(
                max_new_tokens=max_new_tokens,
                total_max_length=total_max_length,
                padded_prompt_token_count=prompt_token_count,
            )
            for prompt_token_count in prompt_token_counts
        ]

        generation_records: list[dict[str, Any] | None] = [None] * len(batch)
        runnable_by_max_tokens: dict[int, list[int]] = {}
        for batch_index, (prompt_token_count, effective_max_new_tokens) in enumerate(
            zip(prompt_token_counts, effective_max_new_tokens_by_prompt)
        ):
            if effective_max_new_tokens <= 0:
                error = (
                    "Prompt is too long for total_max_length: "
                    f"prompt_token_count={prompt_token_count}, total_max_length={total_max_length}"
                )
                generation_records[batch_index] = {
                    "generated_full_text": "",
                    "generated_raw_continuation_before_restore": "",
                    "vllm_finish_reason": None,
                    "vllm_stop_reason": None,
                    "vllm_restored_event_end_token": False,
                    "prompt_token_count": int(prompt_token_count),
                    "batch_padded_prompt_token_count": int(padded_prompt_token_count),
                    "effective_max_new_tokens": int(effective_max_new_tokens),
                    "generation_error": error,
                    "skipped_prompt_too_long": True,
                }
            else:
                runnable_by_max_tokens.setdefault(int(effective_max_new_tokens), []).append(batch_index)

        for effective_max_new_tokens, batch_indices in sorted(runnable_by_max_tokens.items()):
            group_prompts = [prompts[batch_index] for batch_index in batch_indices]
            sampling_params = build_sampling_params(
                sampling_params_cls=sampling_params_cls,
                max_tokens=effective_max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_ids=stop_ids,
            )
            outputs = llm.generate(group_prompts, sampling_params, use_tqdm=False)
            for batch_index, output in zip(batch_indices, outputs):
                prompt = prompts[batch_index]
                prompt_token_count = prompt_token_counts[batch_index]
                generated = output.outputs[0]
                continuation_raw = str(generated.text)
                stop_reason = getattr(generated, "stop_reason", None)
                continuation, restored = restore_stop_token_text(
                    continuation=continuation_raw,
                    stop_reason=stop_reason,
                    event_end_token_id=event_end_token_id,
                )
                generated_full_text = trim_to_closing_tag(prompt + continuation, prompt)
                generation_records[batch_index] = {
                    "generated_full_text": generated_full_text,
                    "generated_raw_continuation_before_restore": continuation_raw,
                    "vllm_finish_reason": getattr(generated, "finish_reason", None),
                    "vllm_stop_reason": stop_reason,
                    "vllm_restored_event_end_token": bool(restored),
                    "prompt_token_count": int(prompt_token_count),
                    "batch_padded_prompt_token_count": int(padded_prompt_token_count),
                    "effective_max_new_tokens": int(effective_max_new_tokens),
                    "generation_error": None,
                    "skipped_prompt_too_long": False,
                }

        for batch_index, record in enumerate(generation_records):
            if record is not None:
                continue
            error = (
                "Prompt generation record was not created; this indicates an internal batching bug."
            )
            generation_records[batch_index] = {
                "generated_full_text": "",
                "generated_raw_continuation_before_restore": "",
                "vllm_finish_reason": None,
                "vllm_stop_reason": None,
                "vllm_restored_event_end_token": False,
                "prompt_token_count": int(prompt_token_counts[batch_index]),
                "batch_padded_prompt_token_count": int(padded_prompt_token_count),
                "effective_max_new_tokens": int(effective_max_new_tokens_by_prompt[batch_index]),
                "generation_error": error,
                "skipped_prompt_too_long": True,
            }

        for candidate, record in zip(batch, generation_records):
            assert record is not None
            if bool(record["skipped_prompt_too_long"]):
                extracted = {
                    "valid": False,
                    "status": "skipped_prompt_too_long",
                    "is_complete": False,
                    "error": record["generation_error"],
                    "generated_event_csv": "",
                    "generated_continuation_event_csv": "",
                    "generated_continuation_normalized_text": "",
                    "generated_event_count": 0,
                    "generated_continuation_event_count": 0,
                }
            else:
                extracted = extract_generated_continuation_from_full_text(
                    generated_full_text=str(record["generated_full_text"]),
                    seeded_event_count=int(candidate["seeded_event_count"]),
                )

            generated_continuation = str(record["generated_full_text"])
            prompt = str(candidate["prompt"])
            if prompt and generated_continuation.startswith(prompt):
                generated_continuation = generated_continuation[len(prompt) :]

            exact_match = (
                bool(extracted["valid"])
                and str(extracted["generated_continuation_event_csv"]) == str(candidate["target_continuation_event_csv"])
            )
            rows.append(
                {
                    **candidate,
                    "mode": run_spec["mode"],
                    "run_id": run_spec["run_id"],
                    "run_dir": str(run_spec["run_dir"]),
                    "model_dir": str(run_spec["model_dir"]),
                    "tokenizer_dir": str(run_spec["tokenizer_dir"]),
                    "resolved_config": str(run_spec["resolved_config"]),
                    "generated_raw_full_text": str(record["generated_full_text"]),
                    "generated_raw_continuation": generated_continuation,
                    "generated_raw_continuation_before_restore": record["generated_raw_continuation_before_restore"],
                    "generated_raw_contains_closing_event_tag": CLOSING_EVENT_TAG in str(record["generated_full_text"]),
                    "generated_event_csv": extracted["generated_event_csv"],
                    "generated_continuation_event_csv": extracted["generated_continuation_event_csv"],
                    "generated_continuation_normalized_text": extracted["generated_continuation_normalized_text"],
                    "generated_event_count": int(extracted["generated_event_count"]),
                    "generated_continuation_event_count": int(extracted["generated_continuation_event_count"]),
                    "valid": bool(extracted["valid"]),
                    "status": extracted["status"],
                    "is_complete": bool(extracted["is_complete"]),
                    "error": extracted["error"],
                    "exact_match": bool(exact_match),
                    "decoding_backend": "vllm",
                    "decoding_do_sample": bool(do_sample),
                    "decoding_temperature": float(temperature),
                    "decoding_top_p": float(top_p),
                    "decoding_top_k": top_k,
                    "decoding_repetition_penalty": float(repetition_penalty),
                    "decoding_max_new_tokens": int(max_new_tokens),
                    "decoding_total_max_length": int(total_max_length) if total_max_length is not None else None,
                    "decoding_effective_max_new_tokens": int(record["effective_max_new_tokens"]),
                    "prompt_token_count": int(record["prompt_token_count"]),
                    "batch_padded_prompt_token_count": int(record["batch_padded_prompt_token_count"]),
                    "generation_error": record["generation_error"],
                    "skipped_prompt_too_long": bool(record["skipped_prompt_too_long"]),
                    "decoding_seed": int(seed),
                    "event_end_token_id": int(event_end_token_id),
                    "eos_token_ids": [int(token_id) for token_id in stop_ids],
                    "stop_token_ids": [int(token_id) for token_id in stop_ids],
                    "vllm_finish_reason": record["vllm_finish_reason"],
                    "vllm_stop_reason": record["vllm_stop_reason"],
                    "vllm_restored_event_end_token": bool(record["vllm_restored_event_end_token"]),
                }
            )
    return rows


def summarize_generation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "missing_status") for row in rows)
    total = len(rows)
    complete = int(status_counts.get("complete", 0))
    truncated = int(status_counts.get("truncated", 0))
    parseable = sum(1 for row in rows if bool(row.get("valid")))
    exact = sum(1 for row in rows if bool(row.get("exact_match")))
    restored = sum(1 for row in rows if bool(row.get("vllm_restored_event_end_token")))
    raw_end = sum(1 for row in rows if bool(row.get("generated_raw_contains_closing_event_tag")))
    length_finish = sum(1 for row in rows if str(row.get("vllm_finish_reason")).lower() == "length")
    return {
        "total_generation_attempts": int(total),
        "complete_generations": complete,
        "truncated_generations": truncated,
        "parseable_generations": int(parseable),
        "invalid_malformed_generations": int(status_counts.get("invalid_malformed", 0)),
        "skipped_prompt_too_long_generations": int(status_counts.get("skipped_prompt_too_long", 0)),
        "complete_generation_rate": complete / total if total else math.nan,
        "parseable_generation_rate": parseable / total if total else math.nan,
        "exact_match_count": int(exact),
        "exact_match_rate": exact / total if total else math.nan,
        "raw_contains_closing_event_tag_generations": int(raw_end),
        "vllm_restored_event_end_token_generations": int(restored),
        "length_finish_generations": int(length_finish),
        "status_counts": dict(status_counts),
        "finish_reason_counts": dict(Counter(str(row.get("vllm_finish_reason")) for row in rows)),
        "stop_reason_counts": dict(Counter(str(row.get("vllm_stop_reason")) for row in rows)),
    }


def write_readme(output_dir: Path) -> None:
    text = """Stage 4A memorization risk data-generation output

This clean vLLM path reads only the train split and creates fraction-based
memorization probes compatible with the old Stage 4A WD inputs.

Main files:
  selected_train_rows.jsonl
  memorization_samples.jsonl
  phases/fraction_*pct/train_memorization_candidates.jsonl
  phases/fraction_*pct/train_memorization_generations.jsonl
  phases/fraction_*pct/summary.json
  stage_4a_summary.json
"""
    (output_dir / "README_stage4_memorization_risk.txt").write_text(text, encoding="utf-8")


def main() -> None:
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--temp-tag", type=str, default=None)
    parser.add_argument("--samples-filename", type=str, default=None)
    parser.add_argument("--max-train-patients", type=int, default=None)
    parser.add_argument("--random-sample-train", action="store_true", default=None)
    parser.add_argument("--no-random-sample-train", action="store_false", dest="random_sample_train")
    parser.add_argument(
        "--train-selection-mode",
        type=str,
        default=None,
        choices=["post_filter_shuffle", "head_ids", "random_ids"],
    )
    parser.add_argument("--min-events", type=int, default=None)
    parser.add_argument("--fractions", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--total-max-length", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true", default=None)
    parser.add_argument("--no-sample", action="store_false", dest="do_sample")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=str, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--max-model-len", type=str, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--no-split-sanity", action="store_true")
    args = parser.parse_args()

    cfg_args = load_config(args.config)
    started_at = time.perf_counter()
    run_dir_value = _choose(args.run_dir, cfg_args, "model", "run_dir")
    output_dir_value = _choose(args.output_dir, cfg_args, "output", "output_dir")
    if run_dir_value is None:
        raise ValueError("--run-dir is required, either on the CLI or in model.run_dir")
    if output_dir_value is None:
        raise ValueError("--output-dir is required, either on the CLI or in output.output_dir")

    run_dir = Path(run_dir_value)
    output_dir = Path(output_dir_value)
    checkpoint = _choose(args.checkpoint, cfg_args, "model", "checkpoint")
    samples_filename = str(
        _choose(args.samples_filename, cfg_args, "output", "memorization_samples_filename", "memorization_samples.jsonl")
    )
    max_train_patients = _choose(args.max_train_patients, cfg_args, "memorization", "max_train_patients", 1000)
    random_sample_train = bool(_choose(args.random_sample_train, cfg_args, "memorization", "random_sample_train", True))
    train_selection_mode = str(_choose(args.train_selection_mode, cfg_args, "memorization", "train_selection_mode", "random_ids"))
    min_events = int(_choose(args.min_events, cfg_args, "memorization", "min_events", 8))
    fractions = parse_fractions(str(_choose(args.fractions, cfg_args, "memorization", "fractions", "0,0.25,0.5,0.75")))
    max_new_tokens = int(_choose(args.max_new_tokens, cfg_args, "memorization", "max_new_tokens", 16384))
    total_max_length = int_or_none(_choose(args.total_max_length, cfg_args, "memorization", "total_max_length", 16384))
    batch_size = int(_choose(args.batch_size, cfg_args, "memorization", "batch_size", 4))
    do_sample = bool(_choose(args.do_sample, cfg_args, "memorization", "do_sample", True))
    temperature = float(_choose(args.temperature, cfg_args, "memorization", "temperature", 0.8))
    top_p = float(_choose(args.top_p, cfg_args, "memorization", "top_p", 0.9))
    top_k = parse_top_k(_choose(args.top_k, cfg_args, "memorization", "top_k", 10))
    repetition_penalty = float(_choose(args.repetition_penalty, cfg_args, "memorization", "repetition_penalty", 1.1))
    seed = int(_choose(args.seed, cfg_args, "memorization", "seed", 42))
    dtype = str(_choose(args.dtype, cfg_args, "runtime", "dtype", "bfloat16"))
    max_model_len = int_or_none(_choose(args.max_model_len, cfg_args, "runtime", "max_model_len", 16384))
    gpu_memory_utilization = float(_choose(args.gpu_memory_utilization, cfg_args, "runtime", "gpu_memory_utilization", 0.85))
    trust_remote_code = bool(_choose(args.trust_remote_code, cfg_args, "model", "trust_remote_code", True))
    set_seed(seed)

    run_spec = resolve_run_spec(run_dir, checkpoint=checkpoint)
    run_name = str(_choose(args.run_name, cfg_args, "memorization", "run_name", "") or "").strip()
    if not run_name:
        run_name = run_dir.resolve().parent.parent.name if len(run_dir.resolve().parents) >= 2 else run_dir.name
    temp_tag = str(_choose(args.temp_tag, cfg_args, "memorization", "temp_tag", "") or "").strip()
    if not temp_tag:
        temp_tag = temp_tag_from_temperature(temperature)

    training_config = Path(args.training_config) if args.training_config else run_spec["resolved_config"]
    cfg = OmegaConf.load(training_config)
    OmegaConf.resolve(cfg)
    if not args.no_split_sanity:
        assert_no_split_overlap_from_cfg(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "args.json", vars(args))
    write_json(output_dir / "run_spec.json", {key: str(value) for key, value in run_spec.items()})
    write_readme(output_dir)

    train_rows, train_selection_metadata = load_train_rows(
        cfg=cfg,
        max_train_patients=int(max_train_patients) if max_train_patients is not None else None,
        random_sample_train=random_sample_train,
        train_selection_mode=train_selection_mode,
        seed=seed,
        min_events=min_events,
    )
    selected_train_rows = build_selected_train_rows(
        train_rows,
        run_name=run_name,
        temperature=temperature,
        temp_tag=temp_tag,
    )
    selected_train_rows_path = output_dir / "selected_train_rows.jsonl"
    write_jsonl(selected_train_rows_path, selected_train_rows)

    tokenizer = AutoTokenizer.from_pretrained(
        str(run_spec["tokenizer_dir"]),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    eos_ids = stop_token_ids(tokenizer, CLOSING_EVENT_TAG)
    event_end_token_id = eos_ids[0]

    llm_kwargs: dict[str, Any] = {
        "model": str(run_spec["model_dir"]),
        "tokenizer": str(run_spec["tokenizer_dir"]),
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "seed": seed,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = int(max_model_len)
    llm = LLM(**llm_kwargs)

    phase_summaries: list[dict[str, Any]] = []
    all_generation_rows: list[dict[str, Any]] = []
    phases_dir = output_dir / "phases"
    stage_summary_path = output_dir / "stage_4a_summary.json"

    def stage_summary(is_complete: bool) -> dict[str, Any]:
        return {
            "stage": "stage_4a_memorization_risk_data_generation",
            "backend": "vllm",
            "meaning": "TRAINING SET memorization probe data generation.",
            "data_split_used": "train",
            "run_name": run_name,
            "temperature": float(temperature),
            "temp_tag": temp_tag,
            "mode": run_spec["mode"],
            "run_id": run_spec["run_id"],
            "run_dir": str(run_spec["run_dir"]),
            "model_dir": str(run_spec["model_dir"]),
            "tokenizer_dir": str(run_spec["tokenizer_dir"]),
            "resolved_config": str(run_spec["resolved_config"]),
            "training_config": str(training_config),
            "training_config_sha256": file_sha256(training_config),
            "output_dir": str(output_dir),
            "is_complete": bool(is_complete),
            "completed_phase_count": int(len(phase_summaries)),
            "expected_phase_count": int(len(fractions)),
            "max_rows": int(cfg.max_rows),
            "max_text_length": int(cfg.max_text_length),
            "min_events": int(min_events),
            "fractions": [float(fraction) for fraction in fractions],
            "strategies": [strategy_name_from_fraction(fraction) for fraction in fractions],
            **train_selection_metadata,
            "candidate_count": int(sum(int(phase["candidate_count"]) for phase in phase_summaries)),
            "decoding_backend": "vllm",
            "decoding_batch_size": int(batch_size),
            "decoding_max_new_tokens": int(max_new_tokens),
            "decoding_total_max_length": int(total_max_length) if total_max_length is not None else None,
            "decoding_do_sample": bool(do_sample),
            "decoding_temperature": float(temperature),
            "decoding_top_p": float(top_p),
            "decoding_top_k": top_k,
            "decoding_repetition_penalty": float(repetition_penalty),
            "decoding_seed": int(seed),
            "dtype": dtype,
            "max_model_len": int(max_model_len) if max_model_len is not None else None,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "event_end_token_id": int(event_end_token_id),
            "eos_token_ids": [int(token_id) for token_id in eos_ids],
            "stop_token_ids": [int(token_id) for token_id in eos_ids],
            "generation_summary": summarize_generation_rows(all_generation_rows),
            "phase_summaries": phase_summaries,
            "elapsed_seconds": float(time.perf_counter() - started_at),
            "artifacts": {
                "selected_train_rows_jsonl": str(selected_train_rows_path),
                "combined_generations_jsonl": str(output_dir / samples_filename),
                "stage_4a_summary_json": str(stage_summary_path),
                "summary_json": str(output_dir / "summary.json"),
                "args_json": str(output_dir / "args.json"),
                "run_spec_json": str(output_dir / "run_spec.json"),
                "phases_dir": str(phases_dir),
            },
        }

    for fraction in fractions:
        strategy_name = strategy_name_from_fraction(fraction)
        phase_dir = phases_dir / strategy_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        phase_started = time.perf_counter()

        candidates = build_all_candidates(train_rows=train_rows, fractions=[float(fraction)])
        candidates = attach_run_metadata(
            candidates,
            run_name=run_name,
            temperature=temperature,
            temp_tag=temp_tag,
        )
        if not candidates:
            raise RuntimeError(f"No memorization candidates were built for {strategy_name}.")

        candidates_path = phase_dir / "train_memorization_candidates.jsonl"
        write_jsonl(candidates_path, candidates)

        generation_rows = run_memorization_generation_vllm(
            candidates=candidates,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=SamplingParams,
            run_spec=run_spec,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            total_max_length=total_max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            event_end_token_id=event_end_token_id,
            stop_ids=[int(token_id) for token_id in eos_ids],
        )
        generations_path = phase_dir / "train_memorization_generations.jsonl"
        write_jsonl(generations_path, generation_rows)
        all_generation_rows.extend(generation_rows)

        phase_summary = {
            "stage": "stage_4a_memorization_risk_data_generation",
            "backend": "vllm",
            "data_split_used": "train",
            "run_name": run_name,
            "temperature": float(temperature),
            "temp_tag": temp_tag,
            "strategy_name": strategy_name,
            "fraction": float(fraction),
            "phase_dir": str(phase_dir),
            "candidate_count": int(len(candidates)),
            "selected_train_row_count": int(len(train_rows)),
            "generation_summary": summarize_generation_rows(generation_rows),
            "elapsed_seconds": float(time.perf_counter() - phase_started),
            "artifacts": {
                "candidates_jsonl": str(candidates_path),
                "generations_jsonl": str(generations_path),
                "summary_json": str(phase_dir / "summary.json"),
            },
        }
        write_json(phase_dir / "summary.json", phase_summary)
        phase_summaries.append(phase_summary)
        write_json(stage_summary_path, stage_summary(is_complete=False))

    samples_path = output_dir / samples_filename
    write_jsonl(samples_path, all_generation_rows)
    final_summary = stage_summary(is_complete=True)
    final_summary["samples_jsonl"] = str(samples_path)
    final_summary["candidate_count"] = int(len(all_generation_rows))
    final_summary["written_rows"] = int(len(all_generation_rows))
    write_json(stage_summary_path, final_summary)
    write_json(output_dir / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
