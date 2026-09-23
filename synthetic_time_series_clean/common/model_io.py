"""Model/checkpoint location and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from synthetic_time_series_clean.common.paths import resolve_path
from synthetic_time_series_clean.common.runtime import (
    choose_device,
    disable_hub_kernels_if_requested,
    resolve_dtype,
)


def resolve_model_source(cfg: DictConfig) -> tuple[str, bool, bool]:
    model_cfg = cfg.model
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    local_files_only = bool(model_cfg.get("local_files_only", False))
    if bool(model_cfg.get("load_from_local", False)):
        local_path = model_cfg.get("local_path")
        if not local_path:
            raise ValueError("model.local_path is required when model.load_from_local=true")
        return str(resolve_path(str(local_path))), trust_remote_code, local_files_only
    name = model_cfg.get("name")
    if not name:
        raise ValueError("model.name is required when model.load_from_local=false")
    return str(name), trust_remote_code, local_files_only


def resolve_run_spec(run_dir: Path, checkpoint: str | None = None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    resolved_config = run_dir / "resolved_config.yaml"
    if not resolved_config.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found: {resolved_config}")
    if checkpoint is not None:
        model_dir = run_dir / checkpoint
    elif (run_dir / "model.safetensors").exists() or (run_dir / "pytorch_model.bin").exists():
        model_dir = run_dir
    else:
        checkpoints = sorted(
            [child for child in run_dir.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")],
            key=lambda item: int(item.name.split("-", 1)[1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No model.safetensors/pytorch_model.bin or checkpoint-* under {run_dir}")
        model_dir = checkpoints[-1]
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
    tokenizer_dir = run_dir / "tokenizer" if (run_dir / "tokenizer").exists() else run_dir
    mode = run_dir.parent.name
    run_id: int | str = int(run_dir.name) if run_dir.name.isdigit() else run_dir.name
    return {
        "run_dir": run_dir,
        "model_dir": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "resolved_config": resolved_config,
        "mode": mode,
        "run_id": run_id,
    }


def load_causal_lm_and_tokenizer(
    *,
    model_dir: Path,
    tokenizer_dir: Path,
    device_arg: str = "auto",
    dtype_arg: str = "auto",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    disable_hub_kernels_if_requested()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        trust_remote_code=bool(trust_remote_code),
        local_files_only=bool(local_files_only),
    )
    device = choose_device(device_arg)
    dtype = resolve_dtype(device, dtype_arg)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=bool(trust_remote_code),
        local_files_only=bool(local_files_only),
    )
    if dtype is None:
        model = model.to(device=device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer, device

