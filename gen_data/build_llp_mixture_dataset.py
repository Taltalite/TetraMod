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

For real known-ratio IVT data, labels are sample-level proportions rather than
read/site truth. The default ratio-stratified mode therefore builds bags within
each ratio dataset and avoids requiring exact strata to exist in every ratio.
The older common-strata mode remains available for synthetic or tightly matched
controls.
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


@dataclass(frozen=True)
class SelectedSample:
    dataset_idx: int
    source_idx: int
    bag_key: int | None = None


DEFAULT_RATIO_STRATIFIED_FIELDS = "contig,kmer_context,motif_context"
COMMON_STRATA_FIELDS = "motif_context,kmer_context,run_id,q_bin,coverage_bin,primary_site_key"
ALLOWED_MATCH_FIELDS = {
    "run_id",
    "contig",
    "primary_site_key",
    "kmer_context",
    "motif_context",
    "q_bin",
    "coverage_bin",
}


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


def resolve_heldout_sites(datasets: Sequence[RatioDataset], args, rng: np.random.Generator) -> set[str]:
    heldout_sites = set(args.heldout_site or []) | set(load_lines(args.heldout_sites_file))
    if args.heldout_mode != "leave-site" or heldout_sites:
        return heldout_sites

    all_sites: set[str] = set()
    for dataset in datasets:
        sites = dataset.metadata["primary_site_key"].astype(str)
        all_sites.update(site for site in sites if site != "no_A")
    if not all_sites:
        raise ValueError("No primary A sites available for leave-site split")
    count = max(1, int(round(len(all_sites) * float(args.leave_site_fraction))))
    return set(rng.choice(np.asarray(sorted(all_sites), dtype=str), size=count, replace=False).tolist())


def split_masks(
    dataset: RatioDataset,
    args,
    rng: np.random.Generator,
    heldout_sites: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    total = dataset.num_samples
    valid_mask = np.zeros((total,), dtype=bool)
    heldout_runs = set(args.heldout_run or []) | set(load_lines(args.heldout_runs_file))
    heldout_sites = set() if heldout_sites is None else set(heldout_sites)

    if args.heldout_mode == "none":
        valid_mask[:] = False
    elif args.heldout_mode == "leave-run":
        if not heldout_runs:
            raise ValueError("--heldout-mode leave-run requires --heldout-run or --heldout-runs-file")
        valid_mask = np.isin(dataset.metadata["run_id"].astype(str), list(heldout_runs))
    elif args.heldout_mode == "leave-site":
        site_keys = dataset.metadata["primary_site_key"].astype(str)
        if not heldout_sites:
            raise ValueError("leave-site split requires resolved heldout sites")
        valid_mask = np.isin(site_keys, list(heldout_sites))
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
        "heldout_sites_count": int(len(heldout_sites)),
        "heldout_sites_present": int(np.unique(dataset.metadata["primary_site_key"].astype(str)[valid_mask]).size),
        "validation_fraction": float(args.validation_fraction),
    }


def parse_match_fields(text: str) -> tuple[str, ...]:
    fields = tuple(item.strip() for item in text.split(",") if item.strip())
    if not fields:
        raise ValueError("--match-fields must contain at least one field")
    invalid = [field for field in fields if field not in ALLOWED_MATCH_FIELDS]
    if invalid:
        raise ValueError(f"Invalid --match-fields values: {invalid}; allowed={sorted(ALLOWED_MATCH_FIELDS)}")
    return fields


def metadata_field_arrays(dataset: RatioDataset, q_bins: np.ndarray, coverage_bins: np.ndarray) -> dict[str, np.ndarray]:
    q_values = np.nan_to_num(dataset.metadata["mean_qscore"].astype(np.float32), nan=-1.0)
    coverage_values = np.nan_to_num(dataset.metadata["mapping_coverage"].astype(np.float32), nan=-1.0)
    return {
        "run_id": dataset.metadata["run_id"].astype(str),
        "contig": dataset.metadata["contig"].astype(str),
        "primary_site_key": dataset.metadata["primary_site_key"].astype(str),
        "kmer_context": dataset.metadata["kmer_context"].astype(str),
        "motif_context": dataset.metadata["motif_context"].astype(str),
        "q_bin": np.digitize(q_values, q_bins).astype(np.int16).astype(str),
        "coverage_bin": np.digitize(coverage_values, coverage_bins).astype(np.int16).astype(str),
    }


def stratum_keys(
    dataset: RatioDataset,
    q_bins: np.ndarray,
    coverage_bins: np.ndarray,
    match_fields: Sequence[str],
) -> list[tuple]:
    arrays = metadata_field_arrays(dataset, q_bins, coverage_bins)
    return [
        tuple(str(arrays[field][idx]) for field in match_fields)
        for idx in range(dataset.num_samples)
    ]


def select_common_strata(
    datasets: Sequence[RatioDataset],
    masks: Sequence[np.ndarray],
    q_bins: np.ndarray,
    coverage_bins: np.ndarray,
    match_fields: Sequence[str],
    max_per_stratum: int,
    rng: np.random.Generator,
) -> tuple[list[SelectedSample], dict]:
    per_dataset = []
    all_strata = []
    for dataset, mask in zip(datasets, masks):
        strata = stratum_keys(dataset, q_bins, coverage_bins, match_fields)
        groups: dict[tuple, list[int]] = {}
        for idx, keep in enumerate(mask):
            if not keep:
                continue
            key = strata[idx]
            if str(dataset.metadata["primary_site_key"][idx]) == "no_A":
                continue
            groups.setdefault(key, []).append(idx)
        per_dataset.append(groups)
        all_strata.append(set(groups))

    common_strata = set.intersection(*all_strata) if all_strata else set()
    selected: list[SelectedSample] = []
    bag_key = 0
    for key in sorted(common_strata):
        count = min(len(groups[key]) for groups in per_dataset)
        if max_per_stratum > 0:
            count = min(count, max_per_stratum)
        if count <= 0:
            continue
        for dataset_idx, groups in enumerate(per_dataset):
            choices = np.asarray(groups[key], dtype=np.int64)
            picked = rng.choice(choices, size=count, replace=False)
            for idx in picked:
                selected.append(SelectedSample(dataset_idx, int(idx), bag_key))
            bag_key += 1

    rng.shuffle(selected)
    return selected, {
        "mode": "common-strata",
        "match_fields": list(match_fields),
        "common_strata": int(len(common_strata)),
        "selected_reads": int(len(selected)),
        "selected_bags": int(bag_key),
    }


def build_ratio_stratified_bags(
    datasets: Sequence[RatioDataset],
    masks: Sequence[np.ndarray],
    q_bins: np.ndarray,
    coverage_bins: np.ndarray,
    match_fields: Sequence[str],
    bag_size: int,
    min_bag_size: int,
    max_bags_per_stratum: int,
    balance_ratios: bool,
    rng: np.random.Generator,
) -> tuple[list[SelectedSample], dict]:
    if bag_size <= 0:
        raise ValueError(f"--bag-size must be positive, got {bag_size}")
    if min_bag_size <= 0:
        raise ValueError(f"--min-bag-size must be positive, got {min_bag_size}")
    if min_bag_size > bag_size:
        raise ValueError(f"--min-bag-size must be <= --bag-size, got {min_bag_size}>{bag_size}")

    candidate_bags: list[list[list[int]]] = []
    per_ratio_summary = []
    for dataset, mask in zip(datasets, masks):
        strata = stratum_keys(dataset, q_bins, coverage_bins, match_fields)
        groups: dict[tuple, list[int]] = {}
        for idx, keep in enumerate(mask):
            if not keep:
                continue
            if str(dataset.metadata["primary_site_key"][idx]) == "no_A":
                continue
            groups.setdefault(strata[idx], []).append(int(idx))

        bags_for_dataset: list[list[int]] = []
        for key in sorted(groups):
            values = list(groups[key])
            rng.shuffle(values)
            stratum_bags = []
            for start in range(0, len(values), bag_size):
                bag = values[start:start + bag_size]
                if len(bag) >= min_bag_size:
                    stratum_bags.append(bag)
            if max_bags_per_stratum > 0 and len(stratum_bags) > max_bags_per_stratum:
                keep = rng.choice(np.arange(len(stratum_bags)), size=max_bags_per_stratum, replace=False)
                stratum_bags = [stratum_bags[int(idx)] for idx in sorted(keep)]
            bags_for_dataset.extend(stratum_bags)

        rng.shuffle(bags_for_dataset)
        candidate_bags.append(bags_for_dataset)
        per_ratio_summary.append({
            "ratio_label": dataset.ratio_label,
            "candidate_strata": int(len(groups)),
            "candidate_bags": int(len(bags_for_dataset)),
            "candidate_reads": int(sum(len(bag) for bag in bags_for_dataset)),
        })

    if not candidate_bags or any(len(bags) == 0 for bags in candidate_bags):
        empty = [datasets[idx].ratio_label for idx, bags in enumerate(candidate_bags) if not bags]
        raise ValueError(f"ratio-stratified bagging produced no bags for ratios: {empty}")

    max_bags = min(len(bags) for bags in candidate_bags) if balance_ratios else None
    selected: list[SelectedSample] = []
    bag_key = 0
    selected_bags_by_ratio = {}
    for dataset_idx, bags in enumerate(candidate_bags):
        selected_bags = bags
        if max_bags is not None and len(bags) > max_bags:
            keep = rng.choice(np.arange(len(bags)), size=max_bags, replace=False)
            selected_bags = [bags[int(idx)] for idx in sorted(keep)]
        selected_bags_by_ratio[datasets[dataset_idx].ratio_label] = int(len(selected_bags))
        for bag in selected_bags:
            for source_idx in bag:
                selected.append(SelectedSample(dataset_idx, int(source_idx), bag_key))
            bag_key += 1

    rng.shuffle(selected)
    return selected, {
        "mode": "ratio-stratified",
        "match_fields": list(match_fields),
        "bag_size": int(bag_size),
        "min_bag_size": int(min_bag_size),
        "max_bags_per_stratum": int(max_bags_per_stratum),
        "balance_ratios": bool(balance_ratios),
        "ratio_candidate_summary": per_ratio_summary,
        "selected_bags_by_ratio": selected_bags_by_ratio,
        "selected_reads": int(len(selected)),
        "selected_bags": int(bag_key),
    }


def write_selected_split(
    output_dir: Path,
    datasets: Sequence[RatioDataset],
    selected: Sequence[SelectedSample],
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
        for pos, item in enumerate(selected[out_start:out_end], start=out_start):
            dataset_idx = int(item.dataset_idx)
            src_idx = int(item.source_idx)
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
            if item.bag_key is not None:
                bag_keys[pos] = int(item.bag_key)
                bag_key_to_id.setdefault(("explicit", int(item.bag_key)), int(item.bag_key))
            else:
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
    parser.add_argument("--bagging-mode", choices=["ratio-stratified", "common-strata"], default="ratio-stratified")
    parser.add_argument("--match-fields", default=DEFAULT_RATIO_STRATIFIED_FIELDS)
    parser.add_argument("--bag-size", type=int, default=20)
    parser.add_argument("--min-bag-size", type=int, default=4)
    parser.add_argument("--max-bags-per-stratum", type=int, default=0)
    parser.add_argument("--no-balance-ratios", dest="balance_ratios", action="store_false")
    parser.set_defaults(balance_ratios=True)
    parser.add_argument("--max-per-stratum", type=int, default=0, help="common-strata mode only: cap selected samples per ratio per exact stratum; 0 disables the cap.")
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
    if args.bagging_mode == "common-strata" and args.match_fields == DEFAULT_RATIO_STRATIFIED_FIELDS:
        match_fields = parse_match_fields(COMMON_STRATA_FIELDS)
    else:
        match_fields = parse_match_fields(args.match_fields)

    train_masks = []
    valid_masks = []
    split_summaries = []
    heldout_sites = resolve_heldout_sites(datasets, args, rng)
    for dataset in datasets:
        train_mask, valid_mask, split_summary = split_masks(dataset, args, rng, heldout_sites=heldout_sites)
        train_masks.append(train_mask)
        valid_masks.append(valid_mask)
        split_summaries.append({"ratio": dataset.ratio_label, **split_summary})

    if args.bagging_mode == "common-strata":
        train_selected, train_selection_summary = select_common_strata(
            datasets,
            train_masks,
            q_bins,
            coverage_bins,
            match_fields,
            int(args.max_per_stratum),
            rng,
        )
    else:
        train_selected, train_selection_summary = build_ratio_stratified_bags(
            datasets,
            train_masks,
            q_bins,
            coverage_bins,
            match_fields,
            int(args.bag_size),
            int(args.min_bag_size),
            int(args.max_bags_per_stratum),
            bool(args.balance_ratios),
            rng,
        )
    train_summary = write_selected_split(args.output_dir, datasets, train_selected, q_bins, coverage_bins, "train")

    valid_summary = None
    valid_selection_summary = None
    if any(mask.any() for mask in valid_masks):
        if args.bagging_mode == "common-strata":
            valid_selected, valid_selection_summary = select_common_strata(
                datasets,
                valid_masks,
                q_bins,
                coverage_bins,
                match_fields,
                int(args.max_per_stratum),
                rng,
            )
        else:
            valid_selected, valid_selection_summary = build_ratio_stratified_bags(
                datasets,
                valid_masks,
                q_bins,
                coverage_bins,
                match_fields,
                int(args.bag_size),
                int(args.min_bag_size),
                int(args.max_bags_per_stratum),
                bool(args.balance_ratios),
                rng,
            )
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
        "selection": {
            "train": train_selection_summary,
            "validation": valid_selection_summary,
        },
        "train": train_summary,
        "validation": valid_summary,
    }
    (args.output_dir / "llp_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] LLP dataset written to: {args.output_dir}")
    print(json.dumps({"train": train_summary, "validation": valid_summary}, indent=2))


if __name__ == "__main__":
    main()
