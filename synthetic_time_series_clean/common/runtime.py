"""Runtime helpers that are safe for training, inference, and evaluation."""

from __future__ import annotations

import inspect
import os
import random
from pathlib import Path
from typing import Any

_ENV_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def set_seed(seed: int) -> None:
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def disable_hub_kernels_if_requested() -> None:
    use_hub_kernels = os.environ.get("USE_HUB_KERNELS", "YES").strip().upper()
    if use_hub_kernels in _ENV_TRUE_VALUES:
        return
    try:
        from transformers.integrations import hub_kernels
    except Exception:
        return
    hub_kernels._kernels_enabled = False
    hub_kernels._kernels_available = False


def choose_device(device_arg: str):
    import torch

    value = str(device_arg).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device: {device_arg}")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(value)


def cuda_supports_bf16(device: Any) -> bool:
    import torch

    if getattr(device, "type", "") != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8 and torch.cuda.is_bf16_supported()


def resolve_dtype(device: Any, dtype_arg: str):
    import torch

    value = str(dtype_arg).strip().lower()
    if getattr(device, "type", "") != "cuda":
        return None
    if value == "auto":
        return torch.bfloat16 if cuda_supports_bf16(device) else torch.float16
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_arg}")
    resolved = mapping[value]
    if resolved is torch.bfloat16 and not cuda_supports_bf16(device):
        major, minor = torch.cuda.get_device_capability(device)
        raise ValueError(
            f"bfloat16 requested, but GPU does not support native BF16 "
            f"(compute capability {major}.{minor})."
        )
    return resolved


def normalize_training_args_keys(trainer_args_dict: dict[str, Any]) -> dict[str, Any]:
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    if (
        "evaluation_strategy" in trainer_args_dict
        and "evaluation_strategy" not in params
        and "eval_strategy" in params
    ):
        trainer_args_dict["eval_strategy"] = trainer_args_dict.pop("evaluation_strategy")
    if (
        "eval_strategy" in trainer_args_dict
        and "eval_strategy" not in params
        and "evaluation_strategy" in params
    ):
        trainer_args_dict["evaluation_strategy"] = trainer_args_dict.pop("eval_strategy")
    return trainer_args_dict


def safe_path_part(value: Any, default: str = "missing") -> str:
    text = str(value or default).strip() or default
    return "".join(char if char.isalnum() or char in "._=-" else "_" for char in text)


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
