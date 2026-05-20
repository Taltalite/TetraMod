#!/usr/bin/env python3
"""
Create leave-one-motif-out Stage 1 datasets from a merged mAFiA dataset.

Each output dataset contains all samples except one held-out motif in the train
split.  By default, validation is also restricted to train motifs so model
selection does not see the held-out motif.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np


REQUIRED_ARRAYS = ("chunks.npy", "references.npy", "reference_lengths.npy", "mod_targets.npy")
OPTIONAL_ARRAYS = ("bag_keys.npy", "bag_targets.npy")
COPY_BLOCK_SIZE = 2048
DEFAULT_MOTIFS = ("AGACT", "GAACT", "GGACA", "GGACC", "GGACT", "TGACT")


def parse_motifs(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return list(DEFAULT_MOTIFS)
    return [item.strip().upper().replace("U", "T") for item in str(value).split(",") if item.strip()]


def safe_name(motif: str) -> str:
    return str(motif).strip().upper().replace("U", "T").replace("/", "_")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_metadata(split_dir: Path) -> dict[str, np.ndarray]:
    path = split_dir / "metadata.npz"
    require_file(path)
    loaded = np.load(path)
    metadata = {name: loaded[name] for name in loaded.files}
    if "motif_context" not in metadata:
        raise ValueError(f"{path}: missing required metadata field motif_context")
    return metadata


def split_size(split_dir: Path) -> int:
    lengths = np.load(split_dir / "reference_lengths.npy", mmap_mode="r")
    return int(lengths.shape[0])


def validate_split(split_dir: Path) -> tuple[int, dict[str, np.ndarray]]:
    for name in REQUIRED_ARRAYS:
        require_file(split_dir / name)
    metadata = load_metadata(split_dir)
    n = split_size(split_dir)
    for field, values in metadata.items():
        if int(values.shape[0]) != n:
            raise ValueError(f"{split_dir}: metadata field {field} has {values.shape[0]} rows, expected {n}")
    return n, metadata


def copy_array_subset(src_path: Path, dst_path: Path, indices: np.ndarray) -> dict[str, object]:
    src = np.load(src_path, mmap_mode="r")
    shape = (int(indices.shape[0]), *src.shape[1:])
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src.dtype, shape=shape)
    for start in range(0, int(indices.shape[0]), COPY_BLOCK_SIZE):
        end = min(start + COPY_BLOCK_SIZE, int(indices.shape[0]))
        dst[start:end] = src[indices[start:end]]
    del dst
    return {"shape": [int(dim) for dim in shape], "dtype": str(src.dtype)}


def copy_metadata_subset(src_metadata: dict[str, np.ndarray], dst_path: Path, indices: np.ndarray) -> None:
    selected = {field: np.asarray(values)[indices] for field, values in src_metadata.items()}
    np.savez(dst_path, **selected)


def select_indices(metadata: dict[str, np.ndarray], heldout_motif: str, *, keep_heldout: bool) -> np.ndarray:
    motifs = np.asarray(metadata["motif_context"]).astype(str)
    motifs = np.char.upper(np.char.replace(motifs, "U", "T"))
    heldout_motif = heldout_motif.upper().replace("U", "T")
    mask = motifs == heldout_motif if keep_heldout else motifs != heldout_motif
    return np.flatnonzero(mask).astype(np.int64)


def copy_split(
    src_dir: Path,
    dst_dir: Path,
    indices: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> dict[str, object]:
    if indices.size == 0:
        raise ValueError(f"{src_dir}: selected split is empty")

    dst_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in REQUIRED_ARRAYS:
        arrays[name] = copy_array_subset(src_dir / name, dst_dir / name, indices)
    for name in OPTIONAL_ARRAYS:
        src_path = src_dir / name
        if src_path.exists():
            arrays[name] = copy_array_subset(src_path, dst_dir / name, indices)
    copy_metadata_subset(metadata, dst_dir / "metadata.npz", indices)
    return {
        "num_samples": int(indices.shape[0]),
        "arrays": arrays,
    }


def validation_indices_for_mode(
    metadata: dict[str, np.ndarray],
    heldout_motif: str,
    mode: str,
) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "keep-all":
        return np.arange(np.asarray(metadata["motif_context"]).shape[0], dtype=np.int64)
    if mode == "heldout-motif":
        return select_indices(metadata, heldout_motif, keep_heldout=True)
    if mode == "train-motifs":
        return select_indices(metadata, heldout_motif, keep_heldout=False)
    raise ValueError(f"Unsupported validation mode: {mode}")


def create_lomo_datasets(args: argparse.Namespace) -> dict[str, object]:
    dataset_dir = Path(args.dataset_dir)
    output_root = Path(args.output_root)
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_n, train_metadata = validate_split(dataset_dir)
    valid_dir = dataset_dir / "validation"
    valid_metadata = None
    valid_n = 0
    if valid_dir.exists() and args.validation_mode != "none":
        valid_n, valid_metadata = validate_split(valid_dir)
    elif args.validation_mode != "none":
        raise FileNotFoundError(
            f"{dataset_dir}: validation split not found. Use --validation-mode none or provide a dataset with validation/."
        )

    motifs = parse_motifs(args.motifs)
    summary = {
        "source_dataset_dir": str(dataset_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "motifs": motifs,
        "validation_mode": str(args.validation_mode),
        "source_train_samples": int(train_n),
        "source_validation_samples": int(valid_n),
        "datasets": [],
    }

    for motif in motifs:
        out_dir = output_root / f"leave_{safe_name(motif)}"
        if out_dir.exists():
            if not args.force:
                raise FileExistsError(f"{out_dir} exists. Use --force to overwrite.")
            shutil.rmtree(out_dir)

        train_indices = select_indices(train_metadata, motif, keep_heldout=False)
        train_summary = copy_split(dataset_dir, out_dir, train_indices, train_metadata)
        record = {
            "heldout_motif": motif,
            "dataset_dir": str(out_dir.resolve()),
            "train": train_summary,
        }

        if valid_metadata is not None:
            valid_indices = validation_indices_for_mode(valid_metadata, motif, str(args.validation_mode))
            if valid_indices is not None:
                record["validation"] = copy_split(valid_dir, out_dir / "validation", valid_indices, valid_metadata)
        summary["datasets"].append(record)
        print(
            f"[lomo] heldout={motif} train={record['train']['num_samples']} "
            f"validation={record.get('validation', {}).get('num_samples', 0)} -> {out_dir}"
        )

    summary_path = output_root / "lomo_datasets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote LOMO dataset summary to: {summary_path}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Merged mAFiA Stage 1 dataset with metadata.npz.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    parser.add_argument(
        "--validation-mode",
        choices=["train-motifs", "heldout-motif", "keep-all", "none"],
        default="train-motifs",
        help=(
            "How to build each LOMO validation split. train-motifs keeps validation blind to the "
            "held-out motif; heldout-motif is diagnostic only."
        ),
    )
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args(argv)


def main() -> None:
    create_lomo_datasets(parse_args())


if __name__ == "__main__":
    main()
