"""
Shared helpers for TetraMod train validation scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from bonito.data import DataSettings


def resolve_train_mod_data_settings(
    *,
    directory: Path,
    output_dir: Path,
    chunks: int | None,
    valid_chunks: int | None,
    caller: str,
) -> DataSettings:
    """
    Reconstruct the same train/validation split settings used by tetramod train.
    """
    has_validation_dir = (directory / "validation").exists()
    if has_validation_dir:
        num_train_chunks = chunks if chunks is not None else valid_chunks
        num_valid_chunks = valid_chunks if valid_chunks is not None else chunks
        if num_train_chunks is None:
            num_train_chunks = 512
        if num_valid_chunks is None:
            num_valid_chunks = 512
        return DataSettings(directory, num_train_chunks, num_valid_chunks, output_dir)

    if chunks is None or valid_chunks is None:
        raise ValueError(
            f"This dataset has no validation/ directory. Pass both --chunks and --valid-chunks so the {caller} can "
            "reproduce the same train/valid split that tetramod train used."
        )
    return DataSettings(directory, chunks, valid_chunks, output_dir)


def maybe_trim_refs(refs: List[str], model) -> List[str]:
    n_pre = getattr(model, "n_pre_context_bases", 0)
    n_post = getattr(model, "n_post_context_bases", 0)
    if n_pre <= 0 and n_post <= 0:
        return refs

    trimmed = []
    for ref in refs:
        start = n_pre
        end = len(ref) - n_post if n_post else len(ref)
        trimmed.append(ref[start:end])
    return trimmed
