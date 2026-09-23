"""Embedding helpers for the single Sliced WD evaluation path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    model_path: Path
    max_length: int
    batch_size: int


def choose_device(device_arg: str):
    import torch

    value = str(device_arg).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device: {device_arg}")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda requested but CUDA is not available.")
    return torch.device(value)


def load_embedding_model(spec: EmbeddingSpec, device_arg: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not spec.model_path.exists():
        raise FileNotFoundError(f"Embedding model path not found: {spec.model_path}")
    device = choose_device(device_arg)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        str(spec.model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return tokenizer, model, device


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def embed_texts(
    *,
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: Any,
    batch_size: int,
    max_length: int,
    desc: str,
) -> np.ndarray:
    import torch
    from tqdm import tqdm

    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), max(1, int(batch_size))), desc=desc):
        batch = texts[start : start + max(1, int(batch_size))]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def sliced_wasserstein_distance(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    n_projections: int,
    seed: int,
) -> float:
    if reference.shape[0] == 0 or generated.shape[0] == 0:
        return math.nan
    if reference.shape[1] != generated.shape[1]:
        raise ValueError("Embedding dimensions differ for sliced Wasserstein.")

    rng = np.random.default_rng(int(seed))
    dim = int(reference.shape[1])
    projections = rng.normal(size=(int(n_projections), dim))
    projections /= np.clip(np.linalg.norm(projections, axis=1, keepdims=True), EPS, None)

    ref_proj = reference @ projections.T
    gen_proj = generated @ projections.T
    n_quantiles = max(2, min(reference.shape[0], generated.shape[0]))
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    distances = []
    for idx in range(projections.shape[0]):
        ref_q = np.quantile(ref_proj[:, idx], quantiles)
        gen_q = np.quantile(gen_proj[:, idx], quantiles)
        distances.append(np.mean(np.abs(ref_q - gen_q)))
    return float(np.mean(distances))

