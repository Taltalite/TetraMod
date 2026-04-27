#!/usr/bin/env python3
"""
Build a promoted LLP dataset from known-ratio chunk datasets.

The input directories are outputs from create_dataset_dorado_ctc_like.py plus
make_mod_targets_m6a.py. Each input directory represents one known mixture
ratio, for example 0, 25, 50, 75, or 100.

This script keeps the Bonito-compatible arrays while adding:
- bag_keys.npy: integer bag ids used by train_promote --promote-stage llp
- bag_targets.npy: per-read known bag proportion in [0, 1]
- metadata.npz: selected metadata in output order

Sampling is stratified by ratio, motif, k-mer context, run, quality bin, and
coverage bin to avoid a global random mixture that can overfit run/context
artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


COPY_BLOCK_SIZE = 2048
IGNORE_INDEX = -100
METADATA_STRING_FIELDS = (
    "record_id",
    "pod5_read_id",
    "run_id",
    "contig",
    "primary_site_key",
    "kmer_context",
    "motif_context",
)
METADATA_NUMERIC_FIELDS = {
    "ref_start": np.int64,
    "ref_end": np.int64,
    "ref_strand": np.int8,
    "chunk_start": np.int64,
    "chunk_end": np.int64,
    "primary_site_pos": np.int64,
    "mean_qscore": np.float32,
    "mapping_accuracy": np.float32,
    "mapping_coverage": np.float32,
}


@dataclass
class RatioDataset:
    ratio_label: str
    ratio: float
    directory: Path
    chunks: np.ndarray
    references: np.ndarray
    reference_lengths: np.ndarray
    mod_targets: np.ndarray
    metadata: dict[str, np.ndarray]

    @property
    def num_samples(self) -> int:
        return int(self.reference_lengths.shape[0])


def normalize_ratio(value: str) -> float:
    ratio = float(value)
    if ratio > 1.0:
        ratio /= 100.0
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"ratio must be in [0, 1] or [0, 100], got {value!r}")
    return ratio


def parse_ratio_dataset(spec: str) -> tuple[str, float, Path]:
    try:
        ratio_text, directory_text = spec.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid --ratio-dataset {spec!r}; expected <ratio>:<dataset_dir>.") from exc
    return ratio_text, normalize_ratio(ratio_text), Path(directory_text)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_ratio_dataset(spec: str) -> RatioDataset:
    ratio_label, ratio, directory = parse_ratio_dataset(spec)
    directory = directory.resolve()
    for name in ("chunks.npy", "references.npy", "reference_lengths.npy", "mod_targets.npy", "metadata.npz"):
        require_file(directory / name)

    chunks = np.load(directory / "chunks.npy", mmap_mode="r")
    references = np.load(directory / "references.npy", mmap_mode="r")
    reference_lengths = np.load(directory / "reference_lengths.npy", mmap_mode="r")
    mod_targets = np.load(directory / "mod_targets.npy", mmap_mode="r")
    metadata_file = np.load(directory / "metadata.npz")
    metadata = {name: metadata_file[name] for name in metadata_file.files}

    validate_dataset(directory, chunks, references, reference_lengths, mod_targets, metadata)
    return RatioDataset(ratio_label, ratio, directory, chunks, references, reference_lengths, mod_targets, metadata)


def validate_dataset(
    directory: Path,
    chunks: np.ndarray,
    references: np.ndarray,
    reference_lengths: np.ndarray,
    mod_targets: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> None:
    if chunks.ndim != 2:
        raise ValueError(f"{directory}: chunks.npy must be 2D, got {tuple(chunks.shape)}")
    if references.ndim != 2:
        raise ValueError(f"{directory}: references.npy must be 2D, got {tuple(references.shape)}")
    if reference_lengths.ndim != 1:
        raise ValueError(f"{directory}: reference_lengths.npy must be 1D, got {tuple(reference_lengths.shape)}")
    if mod_targets.ndim != 2:
        raise ValueError(f"{directory}: mod_targets.npy must be 2D, got {tuple(mod_targets.shape)}")
    num_samples = int(chunks.shape[0])
    if references.shape[0] != num_samples or reference_lengths.shape[0] != num_samples or mod_targets.shape[0] != num_samples:
        raise ValueError(f"{directory}: chunks/references/reference_lengths/mod_targets length mismatch")
    missing = [field for field in (*METADATA_STRING_FIELDS, *METADATA_NUMERIC_FIELDS.keys()) if field not in metadata]
    if missing:
        raise ValueError(f"{directory}: metadata.npz missing fields: {missing}")
    for field, values in metadata.items():
        if values.shape[0] != num_samples:
            raise ValueError(f"{directory}: metadata field {field} has length {values.shape[0]}, expected {num_samples}")


def parse_bins(text: str) -> np.ndarray:
    if not text:
        return np.asarray([], dtype=np.float32)
    return np.asarray([float(item) for item in text.split(",") if item.strip()], dtype=np.float32)


def load_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_masks(dataset: RatioDataset, args, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    total = dataset.num_samples
    valid_mask = np.zeros((total,), dtype=bool)
    heldout_runs = set(args.heldout_run or []) | set(load_lines(args.heldout_runs_file))
    heldout_sites = set(args.heldout_site or []) | set(load_lines(args.heldout_sites_file))

    if args.heldout_mode == "none":
        valid_mask[:] = False
    elif args.heldout_mode == "leave-run":
        if not heldout_runs:
            raise ValueError("--heldout-mode leave-run requires --heldout-run or --heldout-runs-file")
        valid_mask = np.isin(dataset.metadata["run_id"].astype(str), list(heldout_runs))
    elif args.heldout_mode == "leave-site":
        site_keys = dataset.metadata["primary_site_key"].astype(str)
        if heldout_sites:
            selected_sites = heldout_sites
        else:
            unique_sites = np.unique(site_keys[site_keys != "no_A"])
            if unique_sites.size == 0:
                raise ValueError(f"{dataset.directory}: no primary A sites available for leave-site split")
            count = max(1, int(round(unique_sites.size * float(args.leave_site_fraction))))
            selected_sites = set(rng.choice(unique_sites, size=count, replace=False).astype(str).tolist())
        valid_mask = np.isin(site_keys, list(selected_sites))
    else:
        raise ValueError(f"Unsupported heldout mode: {args.heldout_mode}")

    if args.validation_fraction > 0 and args.heldout_mode == "none":
        indices = np.arange(total)
        count = max(1, int(round(total * float(args.validation_fraction))))
        valid_mask[rng.choice(indices, size=count, replace=False)] = True

    train_mask = ~valid_mask
    return train_mask, valid_mask, {
        "heldout_mode": args.heldout_mode,
        "heldout_runs": sorted(heldout_runs),
        "heldout_sites_count": int(np.unique(dataset.metadata["primary_site_key"].astype(str)[valid_mask]).size),
        "validation_fraction": float(args.validation_fraction),
    }


def stratum_keys(dataset: RatioDataset, q_bins: np.ndarray, coverage_bins: np.ndarray) -> list[tuple]:
    q_values = np.nan_to_num(dataset.metadata["mean_qscore"].astype(np.float32), nan=-1.0)
    coverage_values = np.nan_to_num(dataset.metadata["mapping_coverage"].astype(np.float32), nan=-1.0)
    q_bin_ids = np.digitize(q_values, q_bins).astype(np.int16)
    coverage_bin_ids = np.digitize(coverage_values, coverage_bins).astype(np.int16)
    run_ids = dataset.metadata["run_id"].astype(str)
    kmers = dataset.metadata["kmer_context"].astype(str)
    motifs = dataset.metadata["motif_context"].astype(str)
    sites = dataset.metadata["primary_site_key"].astype(str)
    return [
        (motifs[idx], kmers[idx], run_ids[idx], int(q_bin_ids[idx]), int(coverage_bin_ids[idx]), sites[idx])
        for idx in range(dataset.num_samples)
    ]


def select_balanced(
    datasets: Sequence[RatioDataset],
    masks: Sequence[np.ndarray],
    q_bins: np.ndarray,
    coverage_bins: np.ndarray,
    max_per_stratum: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    per_dataset = []
    all_strata = []
    for dataset, mask in zip(datasets, masks):
        strata = stratum_keys(dataset, q_bins, coverage_bins)
        groups: dict[tuple, list[int]] = {}
        for idx, keep in enumerate(mask):
            if not keep:
                continue
            key = strata[idx]
            if key[-1] == "no_A":
                continue
            groups.setdefault(key, []).append(idx)
        per_dataset.append(groups)
        all_strata.append(set(groups))

    common_strata = set.intersection(*all_strata) if all_strata else set()
    selected: list[tuple[int, int]] = []
    for key in sorted(common_strata):
        count = min(len(groups[key]) for groups in per_dataset)
        if max_per_stratum > 0:
            count = min(count, max_per_stratum)
        if count <= 0:
            continue
        for dataset_idx, groups in enumerate(per_dataset):
            choices = np.asarray(groups[key], dtype=np.int64)
            picked = rng.choice(choices, size=count, replace=False)
            selected.extend((dataset_idx, int(idx)) for idx in picked)

    rng.shuffle(selected)
    return selected


def write_selected_split(
    output_dir: Path,
    datasets: Sequence[RatioDataset],
    selected: Sequence[tuple[int, int]],
    q_bins: np.ndarray,
    coverage_bins: np.ndarray,
    summary_name: str,
) -> dict:
    if not selected:
        raise ValueError(f"{summary_name}: no samples selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    first = datasets[0]
    chunk_width = int(first.chunks.shape[1])
    if any(int(dataset.chunks.shape[1]) != chunk_width for dataset in datasets):
        raise ValueError("All ratio datasets must have the same chunk width")
    reference_width = max(int(dataset.references.shape[1]) for dataset in datasets)
    mod_width = max(int(dataset.mod_targets.shape[1]) for dataset in datasets)
    total = len(selected)

    out_chunks = np.lib.format.open_memmap(output_dir / "chunks.npy", mode="w+", dtype=first.chunks.dtype, shape=(total, chunk_width))
    out_refs = np.lib.format.open_memmap(output_dir / "references.npy", mode="w+", dtype=first.references.dtype, shape=(total, reference_width))
    out_lens = np.lib.format.open_memmap(output_dir / "reference_lengths.npy", mode="w+", dtype=first.reference_lengths.dtype, shape=(total,))
    out_mods = np.lib.format.open_memmap(output_dir / "mod_targets.npy", mode="w+", dtype=first.mod_targets.dtype, shape=(total, mod_width))
    bag_keys = np.empty((total,), dtype=np.int64)
    bag_targets = np.empty((total,), dtype=np.float32)
    metadata_out = {field: [] for field in (*METADATA_STRING_FIELDS, *METADATA_NUMERIC_FIELDS.keys())}
    ratio_labels = []
    source_indices = np.empty((total,), dtype=np.int64)
    bag_key_to_id: dict[tuple, int] = {}

    for out_start in range(0, total, COPY_BLOCK_SIZE):
        out_end = min(out_start + COPY_BLOCK_SIZE, total)
        out_refs[out_start:out_end] = 0
        out_mods[out_start:out_end] = IGNORE_INDEX
        for pos, (dataset_idx, src_idx) in enumerate(selected[out_start:out_end], start=out_start):
            dataset = datasets[dataset_idx]
            out_chunks[pos] = dataset.chunks[src_idx]
            out_refs[pos, :dataset.references.shape[1]] = dataset.references[src_idx]
            out_lens[pos] = dataset.reference_lengths[src_idx]
            out_mods[pos, :dataset.mod_targets.shape[1]] = dataset.mod_targets[src_idx]
            for field in metadata_out:
                metadata_out[field].append(dataset.metadata[field][src_idx])

            q_value = float(np.nan_to_num(np.float32(dataset.metadata["mean_qscore"][src_idx]), nan=-1.0))
            coverage_value = float(np.nan_to_num(np.float32(dataset.metadata["mapping_coverage"][src_idx]), nan=-1.0))
            q_bin = int(np.digitize(q_value, q_bins))
            coverage_bin = int(np.digitize(coverage_value, coverage_bins))
            bag_tuple = (
                dataset.ratio_label,
                str(dataset.metadata["primary_site_key"][src_idx]),
                str(dataset.metadata["run_id"][src_idx]),
                str(dataset.metadata["kmer_context"][src_idx]),
                str(dataset.metadata["motif_context"][src_idx]),
                q_bin,
                coverage_bin,
            )
            if bag_tuple not in bag_key_to_id:
                bag_key_to_id[bag_tuple] = len(bag_key_to_id)
            bag_keys[pos] = bag_key_to_id[bag_tuple]
            bag_targets[pos] = np.float32(dataset.ratio)
            ratio_labels.append(dataset.ratio_label)
            source_indices[pos] = int(src_idx)

    out_chunks.flush()
    out_refs.flush()
    out_lens.flush()
    out_mods.flush()
    np.save(output_dir / "bag_keys.npy", bag_keys)
    np.save(output_dir / "bag_targets.npy", bag_targets)
    np.save(output_dir / "source_indices.npy", source_indices)
    np.save(output_dir / "ratio_labels.npy", np.asarray(ratio_labels, dtype=str))

    metadata_arrays = {}
    for field in METADATA_STRING_FIELDS:
        metadata_arrays[field] = np.asarray(metadata_out[field], dtype=str)
    for field, dtype in METADATA_NUMERIC_FIELDS.items():
        metadata_arrays[field] = np.asarray(metadata_out[field], dtype=dtype)
    np.savez(output_dir / "metadata.npz", **metadata_arrays)

    ratio_counts = {}
    for label in ratio_labels:
        ratio_counts[label] = ratio_counts.get(label, 0) + 1
    summary = {
        "name": summary_name,
        "num_samples": int(total),
        "num_bags": int(len(bag_key_to_id)),
        "ratio_counts": {key: int(value) for key, value in sorted(ratio_counts.items())},
        "output_shapes": {
            "chunks": [int(total), int(chunk_width)],
            "references": [int(total), int(reference_width)],
            "reference_lengths": [int(total)],
            "mod_targets": [int(total), int(mod_width)],
            "bag_keys": [int(total)],
            "bag_targets": [int(total)],
        },
    }
    (output_dir / "llp_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratio-dataset",
        action="append",
        required=True,
        help="Known-ratio dataset in the form <ratio>:<dataset_dir>; use repeatedly for 0/25/50/75/100.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-per-stratum", type=int, default=0, help="Cap selected samples per ratio per exact stratum; 0 disables the cap.")
    parser.add_argument("--qscore-bins", default="8,10,12,14,16", help="Comma-separated mean qscore bin edges.")
    parser.add_argument("--coverage-bins", default="0.85,0.9,0.95,0.98", help="Comma-separated mapping coverage bin edges.")
    parser.add_argument("--heldout-mode", choices=["none", "leave-run", "leave-site"], default="none")
    parser.add_argument("--heldout-run", action="append", default=[])
    parser.add_argument("--heldout-runs-file", type=Path, default=None)
    parser.add_argument("--heldout-site", action="append", default=[])
    parser.add_argument("--heldout-sites-file", type=Path, default=None)
    parser.add_argument("--leave-site-fraction", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.0, help="Random validation fraction only when --heldout-mode none.")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    datasets = [load_ratio_dataset(spec) for spec in args.ratio_dataset]
    if len(datasets) < 2:
        raise ValueError("LLP mixture construction expects at least two --ratio-dataset inputs.")

    q_bins = parse_bins(args.qscore_bins)
    coverage_bins = parse_bins(args.coverage_bins)

    train_masks = []
    valid_masks = []
    split_summaries = []
    for dataset in datasets:
        train_mask, valid_mask, split_summary = split_masks(dataset, args, rng)
        train_masks.append(train_mask)
        valid_masks.append(valid_mask)
        split_summaries.append({"ratio": dataset.ratio_label, **split_summary})

    train_selected = select_balanced(datasets, train_masks, q_bins, coverage_bins, int(args.max_per_stratum), rng)
    train_summary = write_selected_split(args.output_dir, datasets, train_selected, q_bins, coverage_bins, "train")

    valid_summary = None
    if any(mask.any() for mask in valid_masks):
        valid_selected = select_balanced(datasets, valid_masks, q_bins, coverage_bins, int(args.max_per_stratum), rng)
        if valid_selected:
            valid_summary = write_selected_split(args.output_dir / "validation", datasets, valid_selected, q_bins, coverage_bins, "validation")

    summary = {
        "ratio_datasets": [
            {"ratio_label": item.ratio_label, "ratio": item.ratio, "directory": str(item.directory)}
            for item in datasets
        ],
        "qscore_bins": q_bins.tolist(),
        "coverage_bins": coverage_bins.tolist(),
        "heldout": split_summaries,
        "train": train_summary,
        "validation": valid_summary,
    }
    (args.output_dir / "llp_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] LLP dataset written to: {args.output_dir}")
    print(json.dumps({"train": train_summary, "validation": valid_summary}, indent=2))


if __name__ == "__main__":
    main()
