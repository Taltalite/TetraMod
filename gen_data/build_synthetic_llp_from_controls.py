#!/usr/bin/env python3
"""
Build synthetic known-ratio LLP bags from only 100% and 0% control datasets.

This is the path to use when no physically mixed POD5 data exists yet:
1. optionally generate Bonito-style chunk datasets from Dorado BAM/POD5 controls
2. label the 100% source as full-mod and the 0% source as canonical
3. synthesize 0/25/50/75/100 bag-level mixtures from matched strata

The synthetic bag is the LLP unit:
- bag_key: unique integer per synthetic bag
- bag_target: requested modified fraction in [0, 1]
- reads in one bag are sampled from the same site/context/quality/coverage stratum
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


COPY_BLOCK_SIZE = 2048
IGNORE_INDEX = -100
SOURCE_FULL = "full_mod"
SOURCE_CANONICAL = "canonical"
DEFAULT_RATIOS = "0,25,50,75,100"
DEFAULT_MATCH_FIELDS = "primary_site_key,kmer_context,motif_context,qscore_bin,coverage_bin"
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
class ControlDataset:
    source_label: str
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
class SelectedRead:
    source_id: int
    source_index: int
    bag_key: int
    bag_target: float
    ratio_label: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def script_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def parse_ratios(text: str) -> list[tuple[str, float]]:
    ratios = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        normalized = value / 100.0 if value > 1.0 else value
        if normalized < 0.0 or normalized > 1.0:
            raise ValueError(f"ratio must be in [0, 1] or [0, 100], got {item!r}")
        ratios.append((item, normalized))
    if not ratios:
        raise ValueError("--ratios must contain at least one ratio")
    return ratios


def parse_bins(text: str) -> np.ndarray:
    if not text:
        return np.asarray([], dtype=np.float32)
    return np.asarray([float(item) for item in text.split(",") if item.strip()], dtype=np.float32)


def parse_fields(text: str) -> tuple[str, ...]:
    fields = tuple(item.strip() for item in text.split(",") if item.strip())
    allowed = {
        "primary_site_key",
        "kmer_context",
        "motif_context",
        "run_id",
        "qscore_bin",
        "coverage_bin",
    }
    invalid = [field for field in fields if field not in allowed]
    if invalid:
        raise ValueError(f"Invalid --match-fields values: {invalid}; allowed={sorted(allowed)}")
    if not fields:
        raise ValueError("--match-fields must not be empty")
    return fields


def load_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def required_dataset_files_exist(directory: Path) -> bool:
    return all(
        (directory / name).exists()
        for name in ("chunks.npy", "references.npy", "reference_lengths.npy", "mod_targets.npy", "metadata.npz")
    )


def run_command(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    print("[cmd] " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=repo_root(), check=True)


def build_source_dataset(args, source_label: str, output_dir: Path) -> None:
    if required_dataset_files_exist(output_dir) and not args.rebuild_sources:
        print(f"[source] using existing {source_label} dataset: {output_dir}")
        return

    if source_label == SOURCE_FULL:
        bam = args.full_mod_bam
        pod5 = args.full_mod_pod5_dir
        run_id = args.full_mod_run_id
        target_mode = "full-mod"
    else:
        bam = args.canonical_bam
        pod5 = args.canonical_pod5_dir
        run_id = args.canonical_run_id
        target_mode = "canonical"

    if bam is None or pod5 is None:
        raise ValueError(
            f"Missing BAM/POD5 for {source_label}. Provide paths or use --{source_label.replace('_', '-')}-dataset."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    create_cmd = [
        sys.executable,
        script_path("create_dataset_dorado_ctc_like.py"),
        "--bam-file",
        bam,
        "--pod5-dir",
        pod5,
        "--reference-fasta",
        args.reference_fasta,
        "--output-dir",
        output_dir,
        "--sample-type",
        args.sample_type,
        "--chunk-len",
        str(args.chunk_len),
        "--overlap",
        str(args.overlap),
        "--workers",
        str(args.workers),
        "--filter-preset",
        args.filter_preset,
        "--metadata-kmer",
        str(args.metadata_kmer),
        "--seed",
        str(args.seed),
        "--mm2-preset",
        args.mm2_preset,
    ]
    if run_id:
        create_cmd.extend(["--run-id", run_id])
    if args.max_label_len is not None:
        create_cmd.extend(["--max-label-len", str(args.max_label_len)])
    if args.max_records > 0:
        create_cmd.extend(["--max-records", str(args.max_records)])
    if args.max_chunks > 0:
        create_cmd.extend(["--max-chunks", str(args.max_chunks)])
    if args.min_accuracy is not None:
        create_cmd.extend(["--min-accuracy", str(args.min_accuracy)])
    if args.min_coverage is not None:
        create_cmd.extend(["--min-coverage", str(args.min_coverage)])
    if args.min_qscore is not None:
        create_cmd.extend(["--min-qscore", str(args.min_qscore)])
    if args.norm_strategy is not None:
        create_cmd.extend(["--norm-strategy", args.norm_strategy])
    if args.rna002:
        create_cmd.append("--rna002")
    if args.model_config is not None:
        create_cmd.extend(["--model-config", args.model_config])

    run_command(create_cmd, dry_run=args.dry_run)

    target_cmd = [
        sys.executable,
        script_path("make_mod_targets_m6a.py"),
        "--dataset-dir",
        output_dir,
        "--mode",
        target_mode,
        "--non-a-policy",
        "ignore",
    ]
    run_command(target_cmd, dry_run=args.dry_run)


def load_control_dataset(directory: Path, source_label: str) -> ControlDataset:
    directory = directory.resolve()
    for name in ("chunks.npy", "references.npy", "reference_lengths.npy", "mod_targets.npy", "metadata.npz"):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    chunks = np.load(directory / "chunks.npy", mmap_mode="r")
    references = np.load(directory / "references.npy", mmap_mode="r")
    reference_lengths = np.load(directory / "reference_lengths.npy", mmap_mode="r")
    mod_targets = np.load(directory / "mod_targets.npy", mmap_mode="r")
    metadata_file = np.load(directory / "metadata.npz")
    metadata = {name: metadata_file[name] for name in metadata_file.files}
    validate_control_dataset(directory, chunks, references, reference_lengths, mod_targets, metadata)
    return ControlDataset(source_label, directory, chunks, references, reference_lengths, mod_targets, metadata)


def validate_control_dataset(
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
    for field in (*METADATA_STRING_FIELDS, *METADATA_NUMERIC_FIELDS.keys()):
        if metadata[field].shape[0] != num_samples:
            raise ValueError(f"{directory}: metadata field {field} has length {metadata[field].shape[0]}, expected {num_samples}")


def split_masks(
    full: ControlDataset,
    canonical: ControlDataset,
    args,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    datasets = [full, canonical]
    train_masks = [np.ones((dataset.num_samples,), dtype=bool) for dataset in datasets]
    valid_masks = [np.zeros((dataset.num_samples,), dtype=bool) for dataset in datasets]
    heldout_runs = set(args.heldout_run or []) | set(load_lines(args.heldout_runs_file))
    heldout_sites = set(args.heldout_site or []) | set(load_lines(args.heldout_sites_file))

    if args.heldout_mode == "leave-run":
        if not heldout_runs:
            raise ValueError("--heldout-mode leave-run requires --heldout-run or --heldout-runs-file")
        for idx, dataset in enumerate(datasets):
            valid_masks[idx] = np.isin(dataset.metadata["run_id"].astype(str), list(heldout_runs))
            train_masks[idx] = ~valid_masks[idx]
    elif args.heldout_mode == "leave-site":
        if not heldout_sites:
            full_sites = set(full.metadata["primary_site_key"].astype(str))
            canonical_sites = set(canonical.metadata["primary_site_key"].astype(str))
            candidate_sites = sorted((full_sites & canonical_sites) - {"no_A"})
            if not candidate_sites:
                raise ValueError("No common primary_site_key values available for leave-site split")
            count = max(1, int(round(len(candidate_sites) * float(args.leave_site_fraction))))
            heldout_sites = set(rng.choice(np.asarray(candidate_sites, dtype=str), size=count, replace=False).tolist())
        for idx, dataset in enumerate(datasets):
            valid_masks[idx] = np.isin(dataset.metadata["primary_site_key"].astype(str), list(heldout_sites))
            train_masks[idx] = ~valid_masks[idx]
    elif args.heldout_mode == "none":
        if args.validation_fraction > 0:
            for idx, dataset in enumerate(datasets):
                indices = np.arange(dataset.num_samples)
                count = max(1, int(round(dataset.num_samples * float(args.validation_fraction))))
                valid_indices = rng.choice(indices, size=count, replace=False)
                valid_masks[idx][valid_indices] = True
                train_masks[idx] = ~valid_masks[idx]
    else:
        raise ValueError(f"Unsupported heldout mode: {args.heldout_mode}")

    return train_masks, valid_masks, {
        "heldout_mode": args.heldout_mode,
        "heldout_runs": sorted(heldout_runs),
        "heldout_sites_count": int(len(heldout_sites)),
        "validation_fraction": float(args.validation_fraction),
    }


def sample_field(dataset: ControlDataset, idx: int, field: str, q_bins: np.ndarray, coverage_bins: np.ndarray):
    if field == "qscore_bin":
        value = np.float32(dataset.metadata["mean_qscore"][idx])
        return int(np.digitize(float(np.nan_to_num(value, nan=-1.0)), q_bins))
    if field == "coverage_bin":
        value = np.float32(dataset.metadata["mapping_coverage"][idx])
        return int(np.digitize(float(np.nan_to_num(value, nan=-1.0)), coverage_bins))
    return str(dataset.metadata[field][idx])


def group_by_stratum(
    dataset: ControlDataset,
    mask: np.ndarray,
    match_fields: Sequence[str],
    q_bins: np.ndarray,
    coverage_bins: np.ndarray,
) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for idx, keep in enumerate(mask):
        if not keep:
            continue
        site_key = str(dataset.metadata["primary_site_key"][idx])
        if site_key == "no_A":
            continue
        key = tuple(sample_field(dataset, idx, field, q_bins, coverage_bins) for field in match_fields)
        groups.setdefault(key, []).append(int(idx))
    return groups


def take_indices(
    values: list[int],
    count: int,
    pointer: int,
    rng: np.random.Generator,
    allow_replacement: bool,
) -> tuple[list[int], int]:
    if count == 0:
        return [], pointer
    if allow_replacement:
        chosen = rng.choice(np.asarray(values, dtype=np.int64), size=count, replace=True)
        return [int(item) for item in chosen], pointer
    if pointer + count > len(values):
        return [], pointer
    return values[pointer:pointer + count], pointer + count


def synthesize_split(
    full: ControlDataset,
    canonical: ControlDataset,
    masks: Sequence[np.ndarray],
    ratios: Sequence[tuple[str, float]],
    match_fields: Sequence[str],
    q_bins: np.ndarray,
    coverage_bins: np.ndarray,
    bag_size: int,
    bags_per_stratum: int,
    allow_replacement: bool,
    rng: np.random.Generator,
) -> tuple[list[SelectedRead], dict]:
    full_groups = group_by_stratum(full, masks[0], match_fields, q_bins, coverage_bins)
    canonical_groups = group_by_stratum(canonical, masks[1], match_fields, q_bins, coverage_bins)
    common_keys = sorted(set(full_groups) & set(canonical_groups))
    selected: list[SelectedRead] = []
    ratio_bag_counts = {label: 0 for label, _ in ratios}
    bag_key = 0

    for key in common_keys:
        full_values = list(full_groups[key])
        canonical_values = list(canonical_groups[key])
        rng.shuffle(full_values)
        rng.shuffle(canonical_values)
        full_pointer = 0
        canonical_pointer = 0

        for ratio_label, ratio in ratios:
            full_count = int(round(bag_size * ratio))
            canonical_count = bag_size - full_count
            made_for_ratio = 0
            while True:
                if bags_per_stratum > 0 and made_for_ratio >= bags_per_stratum:
                    break
                if full_count > 0 and not allow_replacement and full_pointer + full_count > len(full_values):
                    break
                if canonical_count > 0 and not allow_replacement and canonical_pointer + canonical_count > len(canonical_values):
                    break
                if full_count > 0 and not full_values:
                    break
                if canonical_count > 0 and not canonical_values:
                    break

                full_pick, full_pointer = take_indices(
                    full_values,
                    full_count,
                    full_pointer,
                    rng,
                    allow_replacement,
                )
                canonical_pick, canonical_pointer = take_indices(
                    canonical_values,
                    canonical_count,
                    canonical_pointer,
                    rng,
                    allow_replacement,
                )
                if len(full_pick) != full_count or len(canonical_pick) != canonical_count:
                    break

                for idx in full_pick:
                    selected.append(SelectedRead(0, idx, bag_key, ratio, ratio_label))
                for idx in canonical_pick:
                    selected.append(SelectedRead(1, idx, bag_key, ratio, ratio_label))
                ratio_bag_counts[ratio_label] += 1
                bag_key += 1
                made_for_ratio += 1

                if bags_per_stratum == 0 and not allow_replacement:
                    continue
                if bags_per_stratum == 0 and allow_replacement:
                    break

    rng.shuffle(selected)
    return selected, {
        "common_strata": int(len(common_keys)),
        "ratio_bag_counts": {key: int(value) for key, value in ratio_bag_counts.items()},
        "num_synthetic_bags": int(bag_key),
        "num_selected_reads": int(len(selected)),
    }


def write_split(
    output_dir: Path,
    datasets: Sequence[ControlDataset],
    selected: Sequence[SelectedRead],
    summary_name: str,
) -> dict:
    if not selected:
        raise ValueError(f"{summary_name}: no reads selected. Check common site/context strata or use --allow-replacement.")

    output_dir.mkdir(parents=True, exist_ok=True)
    first = datasets[0]
    chunk_width = int(first.chunks.shape[1])
    if any(int(dataset.chunks.shape[1]) != chunk_width for dataset in datasets):
        raise ValueError("Control source chunk widths differ")
    reference_width = max(int(dataset.references.shape[1]) for dataset in datasets)
    mod_width = max(int(dataset.mod_targets.shape[1]) for dataset in datasets)
    total = len(selected)

    out_chunks = np.lib.format.open_memmap(output_dir / "chunks.npy", mode="w+", dtype=first.chunks.dtype, shape=(total, chunk_width))
    out_refs = np.lib.format.open_memmap(output_dir / "references.npy", mode="w+", dtype=first.references.dtype, shape=(total, reference_width))
    out_lens = np.lib.format.open_memmap(output_dir / "reference_lengths.npy", mode="w+", dtype=first.reference_lengths.dtype, shape=(total,))
    out_mods = np.lib.format.open_memmap(output_dir / "mod_targets.npy", mode="w+", dtype=first.mod_targets.dtype, shape=(total, mod_width))
    bag_keys = np.empty((total,), dtype=np.int64)
    bag_targets = np.empty((total,), dtype=np.float32)
    source_labels = []
    ratio_labels = []
    source_indices = np.empty((total,), dtype=np.int64)
    metadata_out = {field: [] for field in (*METADATA_STRING_FIELDS, *METADATA_NUMERIC_FIELDS.keys())}

    for out_start in range(0, total, COPY_BLOCK_SIZE):
        out_end = min(out_start + COPY_BLOCK_SIZE, total)
        out_refs[out_start:out_end] = 0
        out_mods[out_start:out_end] = IGNORE_INDEX
        for pos, item in enumerate(selected[out_start:out_end], start=out_start):
            dataset = datasets[item.source_id]
            src_idx = item.source_index
            out_chunks[pos] = dataset.chunks[src_idx]
            out_refs[pos, :dataset.references.shape[1]] = dataset.references[src_idx]
            out_lens[pos] = dataset.reference_lengths[src_idx]
            out_mods[pos, :dataset.mod_targets.shape[1]] = dataset.mod_targets[src_idx]
            bag_keys[pos] = int(item.bag_key)
            bag_targets[pos] = np.float32(item.bag_target)
            source_labels.append(dataset.source_label)
            ratio_labels.append(item.ratio_label)
            source_indices[pos] = int(src_idx)
            for field in metadata_out:
                metadata_out[field].append(dataset.metadata[field][src_idx])

    out_chunks.flush()
    out_refs.flush()
    out_lens.flush()
    out_mods.flush()
    np.save(output_dir / "bag_keys.npy", bag_keys)
    np.save(output_dir / "bag_targets.npy", bag_targets)
    np.save(output_dir / "source_labels.npy", np.asarray(source_labels, dtype=str))
    np.save(output_dir / "ratio_labels.npy", np.asarray(ratio_labels, dtype=str))
    np.save(output_dir / "source_indices.npy", source_indices)

    metadata_arrays = {
        "source_label": np.asarray(source_labels, dtype=str),
        "synthetic_ratio_label": np.asarray(ratio_labels, dtype=str),
    }
    for field in METADATA_STRING_FIELDS:
        metadata_arrays[field] = np.asarray(metadata_out[field], dtype=str)
    for field, dtype in METADATA_NUMERIC_FIELDS.items():
        metadata_arrays[field] = np.asarray(metadata_out[field], dtype=dtype)
    np.savez(output_dir / "metadata.npz", **metadata_arrays)

    ratio_counts = {}
    source_counts = {}
    for label in ratio_labels:
        ratio_counts[label] = ratio_counts.get(label, 0) + 1
    for label in source_labels:
        source_counts[label] = source_counts.get(label, 0) + 1
    summary = {
        "name": summary_name,
        "num_samples": int(total),
        "num_bags": int(np.unique(bag_keys).size),
        "ratio_read_counts": {key: int(value) for key, value in sorted(ratio_counts.items())},
        "source_read_counts": {key: int(value) for key, value in sorted(source_counts.items())},
        "output_shapes": {
            "chunks": [int(total), int(chunk_width)],
            "references": [int(total), int(reference_width)],
            "reference_lengths": [int(total)],
            "mod_targets": [int(total), int(mod_width)],
            "bag_keys": [int(total)],
            "bag_targets": [int(total)],
        },
    }
    (output_dir / "synthetic_llp_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-mod-bam", default=None)
    parser.add_argument("--full-mod-pod5-dir", default=None)
    parser.add_argument("--canonical-bam", default=None)
    parser.add_argument("--canonical-pod5-dir", default=None)
    parser.add_argument("--reference-fasta", default=None)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-mod-dataset", type=Path, default=None, help="Use an existing 100%% chunk dataset instead of generating it.")
    parser.add_argument("--canonical-dataset", type=Path, default=None, help="Use an existing 0%% chunk dataset instead of generating it.")
    parser.add_argument("--rebuild-sources", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--full-mod-run-id", default="full_mod")
    parser.add_argument("--canonical-run-id", default="canonical")
    parser.add_argument("--ratios", default=DEFAULT_RATIOS)
    parser.add_argument("--bag-size", type=int, default=20)
    parser.add_argument("--bags-per-stratum", type=int, default=1, help="Synthetic bags per ratio per common stratum; 0 uses as many as possible without replacement.")
    parser.add_argument("--allow-replacement", action="store_true", default=False)
    parser.add_argument("--match-fields", default=DEFAULT_MATCH_FIELDS)
    parser.add_argument("--qscore-bins", default="8,10,12,14,16")
    parser.add_argument("--coverage-bins", default="0.85,0.9,0.95,0.98")
    parser.add_argument("--heldout-mode", choices=["none", "leave-run", "leave-site"], default="none")
    parser.add_argument("--heldout-run", action="append", default=[])
    parser.add_argument("--heldout-runs-file", type=Path, default=None)
    parser.add_argument("--heldout-site", action="append", default=[])
    parser.add_argument("--heldout-sites-file", type=Path, default=None)
    parser.add_argument("--leave-site-fraction", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.0)

    parser.add_argument("--sample-type", choices=["dna", "rna"], default="rna")
    parser.add_argument("--chunk-len", type=int, default=12000)
    parser.add_argument("--overlap", type=int, default=600)
    parser.add_argument("--max-label-len", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--max-chunks", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--filter-preset", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--min-qscore", type=float, default=None)
    parser.add_argument("--rna002", action="store_true", default=False)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--norm-strategy", choices=["from-bam", "pa", "quantile", "model-config"], default=None)
    parser.add_argument("--metadata-kmer", type=int, default=5)
    parser.add_argument("--mm2-preset", default="lr:hq")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bag_size <= 0:
        raise ValueError(f"--bag-size must be positive, got {args.bag_size}")
    if args.bags_per_stratum < 0:
        raise ValueError(f"--bags-per-stratum must be >= 0, got {args.bags_per_stratum}")
    if args.metadata_kmer <= 0 or args.metadata_kmer % 2 == 0:
        raise ValueError(f"--metadata-kmer must be a positive odd integer, got {args.metadata_kmer}")
    if args.reference_fasta is None and (args.full_mod_dataset is None or args.canonical_dataset is None):
        raise ValueError("--reference-fasta is required when source datasets must be generated from BAM/POD5")

    rng = np.random.default_rng(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    full_dir = args.full_mod_dataset or (args.work_dir / "source_100_full_mod")
    canonical_dir = args.canonical_dataset or (args.work_dir / "source_0_canonical")

    if args.full_mod_dataset is None:
        build_source_dataset(args, SOURCE_FULL, full_dir)
    if args.canonical_dataset is None:
        build_source_dataset(args, SOURCE_CANONICAL, canonical_dir)
    if args.dry_run:
        return

    full = load_control_dataset(full_dir, SOURCE_FULL)
    canonical = load_control_dataset(canonical_dir, SOURCE_CANONICAL)
    ratios = parse_ratios(args.ratios)
    match_fields = parse_fields(args.match_fields)
    q_bins = parse_bins(args.qscore_bins)
    coverage_bins = parse_bins(args.coverage_bins)
    train_masks, valid_masks, split_summary = split_masks(full, canonical, args, rng)

    train_selected, train_selection_summary = synthesize_split(
        full,
        canonical,
        train_masks,
        ratios,
        match_fields,
        q_bins,
        coverage_bins,
        args.bag_size,
        args.bags_per_stratum,
        args.allow_replacement,
        rng,
    )
    train_summary = write_split(args.output_dir, (full, canonical), train_selected, "train")

    valid_summary = None
    valid_selection_summary = None
    if any(mask.any() for mask in valid_masks):
        valid_selected, valid_selection_summary = synthesize_split(
            full,
            canonical,
            valid_masks,
            ratios,
            match_fields,
            q_bins,
            coverage_bins,
            args.bag_size,
            args.bags_per_stratum,
            args.allow_replacement,
            rng,
        )
        if valid_selected:
            valid_summary = write_split(args.output_dir / "validation", (full, canonical), valid_selected, "validation")

    summary = {
        "source_datasets": {
            "full_mod": str(full.directory),
            "canonical": str(canonical.directory),
        },
        "ratios": [{"label": label, "fraction": value} for label, value in ratios],
        "bag_size": int(args.bag_size),
        "bags_per_stratum": int(args.bags_per_stratum),
        "allow_replacement": bool(args.allow_replacement),
        "match_fields": list(match_fields),
        "qscore_bins": q_bins.tolist(),
        "coverage_bins": coverage_bins.tolist(),
        "heldout": split_summary,
        "selection": {
            "train": train_selection_summary,
            "validation": valid_selection_summary,
        },
        "train": train_summary,
        "validation": valid_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "synthetic_llp_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] synthetic LLP dataset written to: {args.output_dir}")
    print(json.dumps({"train": train_summary, "validation": valid_summary}, indent=2))


if __name__ == "__main__":
    main()
