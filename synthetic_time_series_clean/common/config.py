"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from synthetic_time_series_clean.common.constants import VALID_EVAL_DATASETS
from synthetic_time_series_clean.common.paths import PROJECT_ROOT, resolve_path

if TYPE_CHECKING:
    from omegaconf import DictConfig

SPLIT_CONFIG_KEYS = {
    "train": "train_dataset",
    "tuning": "tuning_dataset",
    "held_out": "held_out_dataset",
}


def get_config_value(container: Any, key: str, default: Any = None) -> Any:
    try:
        if key in container:
            return container[key]
    except Exception:
        pass
    try:
        return getattr(container, key)
    except Exception:
        return default


def load_config(path: str | Path | None) -> DictConfig:
    from omegaconf import OmegaConf

    if path is None:
        return OmegaConf.create({})
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    cfg = OmegaConf.load(resolved)
    OmegaConf.resolve(cfg)
    return cfg


def merge_cli_overrides(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    from omegaconf import OmegaConf

    clean = {key: value for key, value in overrides.items() if value is not None}
    if not clean:
        return cfg
    return OmegaConf.merge(cfg, clean)


def require_eval_dataset(value: str) -> str:
    dataset = str(value).strip()
    if dataset not in VALID_EVAL_DATASETS:
        raise ValueError(
            f"Unsupported eval dataset {dataset!r}; expected one of {sorted(VALID_EVAL_DATASETS)}."
        )
    return dataset


def configured_split_name(cfg: Any, split: str) -> str:
    """Map a logical split name through cfg.splits, falling back to the input."""
    split_name = str(split).strip()
    config_key = SPLIT_CONFIG_KEYS.get(split_name)
    if config_key is None:
        return split_name
    splits_cfg = get_config_value(cfg, "splits", {})
    return str(get_config_value(splits_cfg, config_key, split_name)).strip() or split_name


def default_config_path(*parts: str) -> Path:
    return PROJECT_ROOT / "configs" / Path(*parts)
