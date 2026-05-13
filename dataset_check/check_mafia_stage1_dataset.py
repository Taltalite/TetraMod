#!/usr/bin/env python3
"""
Inspect motif/class balance in merged mAFiA Stage 1 datasets.

The script expects a dataset produced by gen_data/merge_mafia_stage1_datasets.py
or gen_data/create_mafia_synthetic_stage1_dataset.py.  By default it checks the
root split as "train" and output_dir/validation as "validation" when present.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


CANONICAL_A_LABEL = 0
M6A_LABEL = 4
IGNORE_INDEX = -100
CLASSES = ("positive", "negative", "mixed", "unlabeled")
DEFAULT_METADATA_FIELDS = (
    "source_dataset",
    "run_id",
    "oligo_ids",
    "ligation_strategy",
    "modification_status",
)


def sample_class(mod_targets: np.ndarray, length: int | None = None) -> str:
    valid = mod_targets if length is None else mod_targets[: int(length)]
    valid = valid[valid != IGNORE_INDEX]
    has_pos = bool(np.any(valid == M6A_LABEL))
    has_neg = bool(np.any(valid == CANONICAL_A_LABEL))
    if has_pos and has_neg:
        return "mixed"
    if has_pos:
        return "positive"
    if has_neg:
        return "negative"
    return "unlabeled"


def metadata_values(metadata: dict[str, np.ndarray], field: str, size: int) -> np.ndarray:
    if field not in metadata:
        return np.full((size,), "unknown", dtype=str)
    values = np.asarray(metadata[field]).astype(str, copy=False)
    if values.shape[0] != size:
        raise ValueError(f"metadata field {field!r} has {values.shape[0]} rows, expected {size}")
    return values


def load_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    mod_path = split_dir / "mod_targets.npy"
    meta_path = split_dir / "metadata.npz"
    if not mod_path.exists():
        raise FileNotFoundError(f"{split_dir}: missing mod_targets.npy")
    if not meta_path.exists():
        raise FileNotFoundError(f"{split_dir}: missing metadata.npz")

    mod_targets = np.load(mod_path, mmap_mode="r")
    lengths = None
    if (split_dir / "reference_lengths.npy").exists():
        lengths = np.load(split_dir / "reference_lengths.npy", mmap_mode="r")
        if lengths.shape[0] != mod_targets.shape[0]:
            raise ValueError(
                f"{split_dir}: reference_lengths.npy has {lengths.shape[0]} rows, "
                f"expected {mod_targets.shape[0]}"
            )

    metadata_npz = np.load(meta_path)
    metadata = {name: metadata_npz[name] for name in metadata_npz.files}
    return mod_targets, lengths, metadata


def iter_existing_splits(dataset_dir: Path, split_names: Iterable[str]) -> list[tuple[str, Path]]:
    result = []
    for split in split_names:
        split = split.strip()
        if not split:
            continue
        split_dir = dataset_dir if split in {"train", ".", "root"} else dataset_dir / split
        if split_dir.exists():
            result.append(("train" if split in {".", "root"} else split, split_dir))
    return result


def inspect_split(
    split_name: str,
    split_dir: Path,
    *,
    group_fields: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mod_targets, lengths, metadata = load_split(split_dir)
    num_samples = int(mod_targets.shape[0])
    motifs = metadata_values(metadata, "motif_context", num_samples)
    group_values = {
        field: metadata_values(metadata, field, num_samples)
        for field in group_fields
    }

    motif_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)

    for idx in range(num_samples):
        length = None if lengths is None else int(lengths[idx])
        cls = sample_class(mod_targets[idx], length)
        motif = str(motifs[idx])
        motif_counts[motif][cls] += 1
        for field, values in group_values.items():
            group_counts[(motif, cls, field, str(values[idx]))] += 1

    motif_rows = []
    for motif in sorted(motif_counts):
        counts = motif_counts[motif]
        positive = int(counts["positive"])
        negative = int(counts["negative"])
        row = {
            "split": split_name,
            "motif_context": motif,
            "positive": positive,
            "negative": negative,
            "mixed": int(counts["mixed"]),
            "unlabeled": int(counts["unlabeled"]),
            "total": int(sum(counts[cls] for cls in CLASSES)),
            "positive_negative_ratio": "" if negative == 0 else f"{positive / negative:.6g}",
        }
        motif_rows.append(row)

    group_rows = []
    for (motif, cls, field, value), count in sorted(group_counts.items()):
        group_rows.append(
            {
                "split": split_name,
                "motif_context": motif,
                "class": cls,
                "field": field,
                "value": value,
                "count": int(count),
            }
        )

    return motif_rows, group_rows


def print_motif_table(rows: list[dict[str, object]]) -> None:
    print("split\tmotif_context\tpositive\tnegative\tmixed\tunlabeled\ttotal\tpos/neg")
    for row in rows:
        print(
            "\t".join(
                str(row[key])
                for key in (
                    "split",
                    "motif_context",
                    "positive",
                    "negative",
                    "mixed",
                    "unlabeled",
                    "total",
                    "positive_negative_ratio",
                )
            )
        )


def print_group_table(rows: list[dict[str, object]]) -> None:
    print("split\tmotif_context\tclass\tfield\tvalue\tcount")
    for row in rows:
        print(
            "\t".join(
                str(row[key])
                for key in ("split", "motif_context", "class", "field", "value", "count")
            )
        )


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Merged mAFiA Stage 1 dataset directory, or a single per-run dataset directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "validation"),
        help="Splits to inspect. 'train', '.', or 'root' means dataset_dir itself.",
    )
    parser.add_argument(
        "--group-fields",
        nargs="+",
        default=DEFAULT_METADATA_FIELDS,
        help="Metadata fields to break down per motif/class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for motif_balance.tsv and motif_group_counts.tsv.",
    )
    parser.add_argument(
        "--no-group-table",
        action="store_true",
        help="Do not print the detailed motif/class/metadata breakdown to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    splits = iter_existing_splits(dataset_dir, args.splits)
    if not splits:
        raise FileNotFoundError(f"No requested split directories found under {dataset_dir}")

    all_motif_rows: list[dict[str, object]] = []
    all_group_rows: list[dict[str, object]] = []
    group_fields = tuple(str(field) for field in args.group_fields)

    for split_name, split_dir in splits:
        motif_rows, group_rows = inspect_split(split_name, split_dir, group_fields=group_fields)
        all_motif_rows.extend(motif_rows)
        all_group_rows.extend(group_rows)

    print_motif_table(all_motif_rows)
    if not args.no_group_table:
        print()
        print_group_table(all_group_rows)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        write_tsv(
            output_dir / "motif_balance.tsv",
            all_motif_rows,
            (
                "split",
                "motif_context",
                "positive",
                "negative",
                "mixed",
                "unlabeled",
                "total",
                "positive_negative_ratio",
            ),
        )
        write_tsv(
            output_dir / "motif_group_counts.tsv",
            all_group_rows,
            ("split", "motif_context", "class", "field", "value", "count"),
        )
        print(f"\nWrote TSV reports to: {output_dir}")


if __name__ == "__main__":
    main()
