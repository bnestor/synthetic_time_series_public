"""vLLM generation-only stage for final or validation splits."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from synthetic_time_series_clean.common.config import load_config, require_eval_dataset
from synthetic_time_series_clean.common.constants import CLOSING_EVENT_TAG
from synthetic_time_series_clean.common.model_io import resolve_run_spec
from synthetic_time_series_clean.common.runtime import set_seed
from synthetic_time_series_clean.common.tokens import stop_token_ids
from synthetic_time_series_clean.inference.generation_common import (
    generation_rates,
    load_real_eval_rows,
    save_generation_outputs,
)
from synthetic_time_series_clean.inference.parsing import (
    parse_full_text,
    restore_stop_token_text,
    trim_to_closing_tag,
)


def _cfg(cfg: DictConfig, section: str, key: str, default: Any = None) -> Any:
    if section in cfg and key in cfg[section]:
        return cfg[section][key]
    return default


def _choose(value: Any, cfg: DictConfig, section: str, key: str, default: Any = None) -> Any:
    return value if value is not None else _cfg(cfg, section, key, default)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples-filename", type=str, default=None)
    parser.add_argument("--eval-dataset", type=str, default=None, choices=["train", "tuning", "held_out"])
    parser.add_argument("--max-eval-patients", type=int, default=None)
    parser.add_argument("--max-train-patients", type=int, default=None)
    parser.add_argument("--random-sample-eval", action="store_true", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--synthetic-samples-per-eval", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true", default=None)
    parser.add_argument("--no-sample", action="store_false", dest="do_sample")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--no-split-sanity", action="store_true")
    return parser


def main() -> None:
    from vllm import LLM, SamplingParams

    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    started_at = time.perf_counter()

    run_dir = Path(_choose(args.run_dir, cfg, "model", "run_dir"))
    output_dir = Path(_choose(args.output_dir, cfg, "output", "output_dir"))
    eval_dataset = require_eval_dataset(_choose(args.eval_dataset, cfg, "generation", "eval_dataset"))
    samples_filename = str(_choose(args.samples_filename, cfg, "output", "samples_filename", "samples.jsonl"))
    checkpoint = _choose(args.checkpoint, cfg, "model", "checkpoint")
    seed = int(_choose(args.seed, cfg, "generation", "seed", 42))
    set_seed(seed)

    batch_size = max(1, int(_choose(args.batch_size, cfg, "generation", "batch_size", 128)))
    samples_per_eval = max(1, int(_choose(args.synthetic_samples_per_eval, cfg, "generation", "synthetic_samples_per_eval", 1)))
    max_new_tokens = int(_choose(args.max_new_tokens, cfg, "generation", "max_new_tokens", 1024))
    do_sample = bool(_choose(args.do_sample, cfg, "generation", "do_sample", True))
    temperature = float(_choose(args.temperature, cfg, "generation", "temperature", 0.8))
    top_p = float(_choose(args.top_p, cfg, "generation", "top_p", 0.9))
    top_k = int(_choose(args.top_k, cfg, "generation", "top_k", 10))
    repetition_penalty = float(_choose(args.repetition_penalty, cfg, "generation", "repetition_penalty", 1.1))
    max_eval_patients = _choose(args.max_eval_patients, cfg, "generation", "max_eval_patients")
    max_train_patients = _choose(args.max_train_patients, cfg, "generation", "max_train_patients")
    random_sample_eval = bool(_choose(args.random_sample_eval, cfg, "generation", "random_sample_eval", False))
    dtype = str(_choose(args.dtype, cfg, "runtime", "dtype", "bfloat16"))
    max_model_len = _choose(args.max_model_len, cfg, "runtime", "max_model_len")
    gpu_memory_utilization = float(_choose(args.gpu_memory_utilization, cfg, "runtime", "gpu_memory_utilization", 0.85))
    trust_remote_code = bool(_choose(args.trust_remote_code, cfg, "model", "trust_remote_code", True))

    run_spec = resolve_run_spec(run_dir, checkpoint=checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        str(run_spec["tokenizer_dir"]),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    eos_ids = stop_token_ids(tokenizer, CLOSING_EVENT_TAG)
    event_end_token_id = eos_ids[0]

    _, real_eval_rows, eval_metadata = load_real_eval_rows(
        resolved_config_path=run_spec["resolved_config"],
        eval_dataset=eval_dataset,
        max_eval_patients=max_eval_patients,
        max_train_patients=max_train_patients,
        random_sample_eval=random_sample_eval,
        seed=seed,
        run_split_sanity=not args.no_split_sanity,
    )

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

    sampling_kwargs: dict[str, Any] = {
        "max_tokens": max_new_tokens,
        "temperature": temperature if do_sample else 0.0,
        "stop_token_ids": [int(token_id) for token_id in eos_ids],
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "seed": seed,
    }
    if do_sample:
        sampling_kwargs.update(
            {
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
        )
    else:
        sampling_kwargs.update({"top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0})
    sampling_params = SamplingParams(**sampling_kwargs)

    requests: list[dict[str, Any]] = []
    for pair_index, real_row in enumerate(real_eval_rows):
        for sample_index in range(samples_per_eval):
            requests.append(
                {
                    "pair_index": pair_index,
                    "sample_index": sample_index,
                    "real_row": real_row,
                    "prompt": real_row["prompt_prefix"],
                }
            )

    status_counts = {"complete": 0, "truncated": 0, "invalid_malformed": 0}
    finish_reason_counts: dict[str, int] = {}
    stop_reason_counts: dict[str, int] = {}
    raw_contains_closing = 0
    restored_event_end = 0
    length_finish = 0
    sample_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(requests), batch_size), desc=f"vllm generate {eval_dataset}"):
        batch = requests[start : start + batch_size]
        outputs = llm.generate([item["prompt"] for item in batch], sampling_params, use_tqdm=False)
        for req, output in zip(batch, outputs):
            real_row = req["real_row"]
            prompt_prefix = str(req["prompt"])
            generated = output.outputs[0]
            continuation_raw = generated.text
            finish_reason = getattr(generated, "finish_reason", None)
            stop_reason = getattr(generated, "stop_reason", None)
            finish_reason_counts[str(finish_reason)] = finish_reason_counts.get(str(finish_reason), 0) + 1
            stop_reason_counts[str(stop_reason)] = stop_reason_counts.get(str(stop_reason), 0) + 1
            if str(finish_reason).lower() == "length":
                length_finish += 1
            continuation_for_parse, restored = restore_stop_token_text(
                continuation=continuation_raw,
                stop_reason=stop_reason,
                event_end_token_id=event_end_token_id,
            )
            restored_event_end += int(restored)
            synthetic_raw_full_text = trim_to_closing_tag(prompt_prefix + continuation_for_parse, prompt_prefix)
            contains_closing = CLOSING_EVENT_TAG in synthetic_raw_full_text
            raw_contains_closing += int(contains_closing)
            parsed = parse_full_text(synthetic_raw_full_text, allow_truncated=True)
            status = str(parsed["status"])
            if status not in status_counts:
                status = "invalid_malformed"
            status_counts[status] += 1
            sample_rows.append(
                {
                    "mode": run_spec["mode"],
                    "run_id": run_spec["run_id"],
                    "pair_index": int(req["pair_index"]),
                    "sample_index": int(req["sample_index"]),
                    "eval_dataset": eval_dataset,
                    "id_column": real_row["id_column"],
                    "id_value": real_row["id_value"],
                    "original_static": real_row["original_static"],
                    "original_events": real_row["original_events"],
                    "original_full_text": real_row["original_full_text"],
                    "synthetic_continuation_raw": continuation_raw,
                    "synthetic_continuation": continuation_for_parse,
                    "synthetic_finish_reason": finish_reason,
                    "synthetic_stop_reason": stop_reason,
                    "vllm_restored_event_end_token": bool(restored),
                    "synthetic_raw_full_text": synthetic_raw_full_text,
                    "synthetic_raw_contains_closing_event_tag": bool(contains_closing),
                    "synthetic_static": parsed["static"],
                    "synthetic_events": parsed["events"],
                    "synthetic_full_text": parsed["full_text"],
                    "prompt_prefix": prompt_prefix,
                    "synthetic_decoding_backend": "vllm",
                    "synthetic_decoding_do_sample": do_sample,
                    "synthetic_decoding_temperature": temperature,
                    "synthetic_decoding_top_p": top_p,
                    "synthetic_decoding_top_k": top_k,
                    "synthetic_decoding_repetition_penalty": repetition_penalty,
                    "synthetic_decoding_max_new_tokens": max_new_tokens,
                    "synthetic_decoding_seed": seed,
                    "status": status,
                    "valid": bool(parsed["valid"]),
                    "is_complete": bool(parsed["is_complete"]),
                    "error": parsed["error"],
                    "original_event_count": int(real_row["original_event_count"]),
                    "synthetic_event_count": int(parsed["event_count"]),
                }
            )

    total_attempts = len(requests)
    rates = generation_rates(status_counts, total_attempts)
    summary = {
        "backend": "vllm",
        "mode": run_spec["mode"],
        "run_id": run_spec["run_id"],
        "run_dir": str(run_spec["run_dir"]),
        "model_dir": str(run_spec["model_dir"]),
        "tokenizer_dir": str(run_spec["tokenizer_dir"]),
        "resolved_config": str(run_spec["resolved_config"]),
        "output_dir": str(output_dir),
        **eval_metadata,
        "eval_count": int(len(real_eval_rows)),
        "synthetic_samples_per_eval": int(samples_per_eval),
        "total_generation_attempts": int(total_attempts),
        "batch_size": int(batch_size),
        **rates,
        "raw_contains_closing_event_tag_generations": int(raw_contains_closing),
        "raw_contains_closing_event_tag_rate": raw_contains_closing / total_attempts if total_attempts else math.nan,
        "vllm_restored_event_end_token_generations": int(restored_event_end),
        "vllm_restored_event_end_token_rate": restored_event_end / total_attempts if total_attempts else math.nan,
        "length_finish_generations": int(length_finish),
        "length_finish_rate": length_finish / total_attempts if total_attempts else math.nan,
        "finish_reason_counts": finish_reason_counts,
        "stop_reason_counts": stop_reason_counts,
        "stop_token_restore_note": (
            "vLLM may omit the stop token text. Only samples with stop_reason == "
            "event_end_token_id have </event> restored before parsing."
        ),
        "stop_string": CLOSING_EVENT_TAG,
        "event_end_token_id": int(event_end_token_id),
        "stop_token_ids": [int(token_id) for token_id in eos_ids],
        "dtype": dtype,
        "max_model_len": int(max_model_len) if max_model_len is not None else None,
        "gpu_memory_utilization": gpu_memory_utilization,
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    save_generation_outputs(
        output_dir=output_dir,
        samples_filename=samples_filename,
        args=args,
        run_spec=run_spec,
        sample_rows=sample_rows,
        summary=summary,
    )


if __name__ == "__main__":
    main()
