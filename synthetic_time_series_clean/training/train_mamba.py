"""Clean Mamba causal language model training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from synthetic_time_series_clean.common.config import default_config_path, load_config
from synthetic_time_series_clean.common.constants import SPECIAL_TOKENS
from synthetic_time_series_clean.common.model_io import resolve_model_source
from synthetic_time_series_clean.common.paths import allocate_indexed_output_dir, resolve_path
from synthetic_time_series_clean.common.runtime import (
    disable_hub_kernels_if_requested,
    normalize_training_args_keys,
    set_seed,
)
from synthetic_time_series_clean.common.tokens import ensure_special_tokens
from synthetic_time_series_clean.data.rendering import (
    MIMICShardSource,
    load_dataset_partitions,
    render_mimic_patient_text_for_sources,
)
from synthetic_time_series_clean.data.splits import assert_no_split_overlap_from_cfg


os.environ.setdefault("USE_HUB_KERNELS", "0")


def tokenize_text(tokenizer: Any, text: str, max_text_length: int) -> dict[str, torch.Tensor]:
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=int(max_text_length),
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].squeeze(0)
    attention_mask = tokenized["attention_mask"].squeeze(0)
    labels = input_ids.clone().masked_fill(attention_mask == 0, -100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class MIMICTimeSeriesDataset(Dataset):
    def __init__(
        self,
        *,
        identifiers: list[tuple[str, Any]],
        data_sources: list[MIMICShardSource],
        tokenizer: Any,
        max_rows: int,
        max_text_length: int,
    ) -> None:
        self.identifiers = identifiers
        self.tokenizer = tokenizer
        self.max_rows = int(max_rows)
        self.max_text_length = int(max_text_length)
        self.sources_by_identifier: dict[tuple[str, Any], list[MIMICShardSource]] = {}
        for source in data_sources:
            for identifier in source.identifiers:
                self.sources_by_identifier.setdefault((source.id_column, identifier), []).append(source)

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        id_column, id_value = self.identifiers[index]
        sources = self.sources_by_identifier.get((id_column, id_value))
        if not sources:
            raise RuntimeError(f"No MEDS/MIMIC shard found for {id_column}={id_value!r}")
        text = render_mimic_patient_text_for_sources(
            sources=sources,
            id_value=id_value,
            max_rows=self.max_rows,
        )
        return tokenize_text(self.tokenizer, text, self.max_text_length)


def _resolve_numerical_normalization_flag(cfg: DictConfig) -> bool:
    data_cfg = cfg.get("data")
    if data_cfg is not None and "numerical_normalization" in data_cfg:
        return bool(data_cfg.get("numerical_normalization"))
    return bool(cfg.get("numerical_normalization", False))


def save_loss_history(log_history: list[dict[str, Any]], output_dir: Path) -> None:
    rows_by_step: dict[int, dict[str, Any]] = {}
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        step = int(step)
        row = rows_by_step.setdefault(step, {"step": step, "train_loss": "", "eval_loss": ""})
        if "loss" in entry:
            row["train_loss"] = float(entry["loss"])
        if "eval_loss" in entry:
            row["eval_loss"] = float(entry["eval_loss"])
    rows = [rows_by_step[key] for key in sorted(rows_by_step)]
    with (output_dir / "loss_history.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (output_dir / "loss_history.csv").open("w", encoding="utf-8") as handle:
        handle.write("step,train_loss,eval_loss\n")
        for row in rows:
            handle.write(f"{row['step']},{row['train_loss']},{row['eval_loss']}\n")


def train_from_config(cfg: DictConfig) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    set_seed(int(cfg.training.seed))
    disable_hub_kernels_if_requested()
    if bool(cfg.get("sanity", {}).get("check_split_overlap", True)):
        split_report = assert_no_split_overlap_from_cfg(cfg)
    else:
        split_report = {"skipped": True}

    model_source, trust_remote_code, local_files_only = resolve_model_source(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    added_tokens = ensure_special_tokens(tokenizer, SPECIAL_TOKENS)
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    train_sources, train_ids, eval_sources, eval_ids = load_dataset_partitions(cfg, eval_split="tuning")
    if not train_ids:
        raise ValueError("No training identifiers found in train split.")
    if not eval_ids:
        raise ValueError("No validation identifiers found in tuning split.")

    train_dataset = MIMICTimeSeriesDataset(
        identifiers=train_ids,
        data_sources=train_sources,
        tokenizer=tokenizer,
        max_rows=int(cfg.max_rows),
        max_text_length=int(cfg.max_text_length),
    )
    eval_dataset = MIMICTimeSeriesDataset(
        identifiers=eval_ids,
        data_sources=eval_sources,
        tokenizer=tokenizer,
        max_rows=int(cfg.max_rows),
        max_text_length=int(cfg.max_text_length),
    )

    trainer_args_dict = OmegaConf.to_container(cfg.training.trainer_args, resolve=True)
    assert isinstance(trainer_args_dict, dict)
    if "output_dir" not in trainer_args_dict or not trainer_args_dict["output_dir"]:
        raise ValueError("training.trainer_args.output_dir must be set.")
    base_output_dir = resolve_path(str(trainer_args_dict["output_dir"]))
    mode_name = "normalized" if _resolve_numerical_normalization_flag(cfg) else "unnormalized"
    output_dir = allocate_indexed_output_dir(base_output_dir, mode_name)
    trainer_args_dict["output_dir"] = str(output_dir)
    trainer_args_dict["logging_dir"] = str(output_dir / "logs")
    trainer_args_dict = normalize_training_args_keys(trainer_args_dict)

    training_args = TrainingArguments(**trainer_args_dict)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    resume_from_checkpoint = cfg.training.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    with (output_dir / "final_tuning_eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_eval_metrics, handle, indent=2)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    save_loss_history(trainer.state.log_history, output_dir)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        handle.write(OmegaConf.to_yaml(cfg))
    summary = {
        "global_step": int(trainer.state.global_step),
        "train_examples": len(train_dataset),
        "tuning_examples": len(eval_dataset),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "output_dir": str(output_dir),
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "split_sanity": split_report,
        "added_tokens": int(added_tokens),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path("training", "mamba_train.yaml"),
        help="Clean training config. This replaces the old bad Hydra default name.",
    )
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--model-local-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.resume_from_checkpoint is not None:
        cfg.training.resume_from_checkpoint = str(args.resume_from_checkpoint)
    if args.model_local_path is not None:
        cfg.model.load_from_local = True
        cfg.model.local_path = str(args.model_local_path)
    if args.output_dir is not None:
        cfg.training.trainer_args.output_dir = str(args.output_dir)
    train_from_config(cfg)


if __name__ == "__main__":
    main()
