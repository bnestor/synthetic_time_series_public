"""Path and output-directory helpers."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to ``base_dir`` or the clean project root."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is None:
        base_dir = os.environ.get("PROJECT_DIR") or PROJECT_ROOT
    return (Path(base_dir).expanduser().resolve() / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def assert_not_source_tree_output(output_path: Path, source_root: Path | None = None) -> None:
    """Prevent generated artifacts from being written into the package source tree."""
    source_root = (source_root or PROJECT_ROOT).resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(source_root)
    except ValueError:
        return
    allowed = {source_root / "outputs", source_root / "work"}
    if not any(output_path == item or item in output_path.parents for item in allowed):
        raise ValueError(
            f"Refusing to write runtime outputs inside source tree: {output_path}. "
            f"Use an explicit output/work directory outside the repo, or under {source_root / 'outputs'}."
        )


def next_numeric_run_index(parent_dir: Path) -> int:
    if not parent_dir.exists():
        return 0
    max_index = -1
    for child in parent_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            value = int(child.name)
        except ValueError:
            continue
        max_index = max(max_index, value)
    return max_index + 1


def allocate_indexed_output_dir(base_output_dir: Path, mode_name: str) -> Path:
    mode_dir = base_output_dir / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    index = next_numeric_run_index(mode_dir)
    while True:
        candidate = mode_dir / str(index)
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1

