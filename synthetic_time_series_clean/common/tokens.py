"""Tokenizer helpers shared by training and inference."""

from __future__ import annotations

from typing import Any

from synthetic_time_series_clean.common.constants import CLOSING_EVENT_TAG, SPECIAL_TOKENS


def ensure_special_tokens(tokenizer: Any, special_tokens: list[str] | None = None) -> int:
    """Ensure pad plus rendering tokens exist and return the total newly added count."""
    special_tokens = list(special_tokens or SPECIAL_TOKENS)
    added = 0

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            added += int(tokenizer.add_special_tokens({"pad_token": "<pad>"}))

    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    missing = [token for token in special_tokens if token not in existing]
    if missing:
        added += int(tokenizer.add_special_tokens({"additional_special_tokens": missing}))
    return int(added)


def assert_single_token(tokenizer: Any, token: str = CLOSING_EVENT_TAG) -> int:
    """Return token id only if ``token`` is represented by exactly one tokenizer id."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if token_id is None or int(token_id) < 0:
        raise RuntimeError(f"Could not resolve {token!r} token id: {token_id!r}")
    if encoded != [int(token_id)]:
        raise RuntimeError(
            f"{token!r} must be a single token. "
            f"convert_tokens_to_ids={token_id}, encode(add_special_tokens=False)={encoded}"
        )
    return int(token_id)


def stop_token_ids(tokenizer: Any, closing_tag: str = CLOSING_EVENT_TAG) -> list[int]:
    event_end_id = assert_single_token(tokenizer, closing_tag)
    ids = [event_end_id]
    if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return ids

