#!/usr/bin/env python3
"""
Merge per-run mAFiA synthetic datasets into one balanced Stage 1 dataset.

Each input dataset should be produced by create_mafia_synthetic_stage1_dataset.py
and already contain center-only mod_targets.npy labels.  This merger creates a
training split plus output_dir/validation with stratified validation selection
and per-motif positive/negative balancing for the training split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm


IGNORE_INDEX = -100
CANONICAL_A_LABEL = 0
M6A_LABEL = 4
NEGATIVE_CLASS = 1
POSITIVE_CLASS = 2
COPY_BLOCK_SIZE = 2048
BALANCE_MODE_MOTIF = "motif"
BALANCE_MODE_NONE = "none"
BALANCE_MODE_SOURCE_CLASS = "source-class"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    directory: Path


@dataclass
class SourceDataset:
    name: str
    directory: Path
    chunks: np.ndarray
    references: np.ndarray
    lengths: np.ndarray
    mod_targets: np.ndarray
    metadata: dict[str, np.ndarray]

    @property
    def num_samples(self) -> int:
        return int(self.lengths.shape[0])


@dataclass(frozen=True)
class SelectedSample:
    dataset_idx: int
    source_idx: int


@dataclass(frozen=True)
class ScanResult:
    dataset_idx: int
    name: str
    source_indices: np.ndarray
    class_codes: np.ndarray
    motifs: np.ndarray
    ligations: np.ndarray
    statuses: np.ndarray
    runs: np.ndarray
    class_counts: dict[str, int]

    @property
    def labeled_count(self) -> int:
        return int(self.source_indices.shape[0])


def parse_dataset_spec(text: str) -> DatasetSpec:
    if ":" in text:
        name, directory = text.split(":", 1)
        return DatasetSpec(name.strip(), Path(directory))
    path = Path(text)
    return DatasetSpec(path.name, path)


def load_dataset(spec: DatasetSpec) -> SourceDataset:
    directory = spec.directory
    required = ("chunks.npy", "references.npy", "reference_lengths.npy", "mod_targets.npy", "metadata.npz")
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing required files: {missing}")
    chunks = np.load(directory / "chunks.npy", mmap_mode="r")
    references = np.load(directory / "references.npy", mmap_mode="r")
    lengths = np.load(directory / "reference_lengths.npy", mmap_mode="r")
    mod_targets = np.load(directory / "mod_targets.npy", mmap_mode="r")
    metadata_file = np.load(directory / "metadata.npz")
    metadata = {name: metadata_file[name] for name in metadata_file.files}
    validate_dataset(spec.name, chunks, references, lengths, mod_targets, metadata)
    return SourceDataset(spec.name, directory, chunks, references, lengths, mod_targets, metadata)


def validate_dataset(name, chunks, references, lengths, mod_targets, metadata):
    num_samples = int(lengths.shape[0])
    if chunks.ndim != 2 or references.ndim != 2 or mod_targets.ndim != 2 or lengths.ndim != 1:
        raise ValueError(f"{name}: invalid array dimensions")
    if references.shape[0] != num_samples or chunks.shape[0] != num_samples or mod_targets.shape[0] != num_samples:
        raise ValueError(f"{name}: dataset arrays must share first dimension")
    for field, values in metadata.items():
        if values.shape[0] != num_samples:
            raise ValueError(f"{name}: metadata field {field} has length {values.shape[0]}, expected {num_samples}")


def sample_class(mod_targets: np.ndarray, length: int) -> str:
    valid = mod_targets[:int(length)]
    has_pos = bool((valid == M6A_LABEL).any())
    has_neg = bool((valid == CANONICAL_A_LABEL).any())
    if has_pos and has_neg:
        return "mixed"
    if has_pos:
        return "positive"
    if has_neg:
        return "negative"
    return "unlabeled"


def class_name(class_code: int) -> str:
    if int(class_code) == POSITIVE_CLASS:
        return "positive"
    if int(class_code) == NEGATIVE_CLASS:
        return "negative"
    return "unlabeled"


def metadata_array(metadata: dict[str, np.ndarray], field: str, default: str, size: int) -> np.ndarray:
    values = metadata.get(field)
    if values is None:
        return np.full((size,), default, dtype=str)
    return np.asarray(values)


def scan_dataset_labels(dataset_idx: int, spec: DatasetSpec) -> ScanResult:
    dataset = load_dataset(spec)
    mod_targets = dataset.mod_targets
    has_pos = np.any(mod_targets == M6A_LABEL, axis=1)
    has_neg = np.any(mod_targets == CANONICAL_A_LABEL, axis=1)
    positive_mask = has_pos & ~has_neg
    negative_mask = has_neg & ~has_pos
    labeled_mask = positive_mask | negative_mask
    source_indices = np.flatnonzero(labeled_mask).astype(np.int64, copy=False)
    class_codes = np.where(positive_mask[source_indices], POSITIVE_CLASS, NEGATIVE_CLASS).astype(np.uint8)

    metadata = dataset.metadata
    motifs = metadata_array(metadata, "motif_context", "unknown", dataset.num_samples)[source_indices].astype(str, copy=True)
    ligations = metadata_array(metadata, "ligation_strategy", "unknown", dataset.num_samples)[source_indices].astype(str, copy=True)
    statuses = metadata_array(metadata, "modification_status", "unknown", dataset.num_samples)[source_indices].astype(str, copy=True)
    runs = metadata_array(metadata, "run_id", dataset.name, dataset.num_samples)[source_indices].astype(str, copy=True)

    return ScanResult(
        dataset_idx=dataset_idx,
        name=dataset.name,
        source_indices=source_indices.copy(),
        class_codes=class_codes.copy(),
        motifs=motifs,
        ligations=ligations,
        statuses=statuses,
        runs=runs,
        class_counts={
            "positive": int(np.count_nonzero(positive_mask)),
            "negative": int(np.count_nonzero(negative_mask)),
        },
    )


def stratum_key(dataset: SourceDataset, idx: int, cls: str) -> tuple[str, str, str, str]:
    metadata = dataset.metadata
    motif = str(metadata.get("motif_context", np.asarray(["unknown"] * dataset.num_samples))[idx])
    ligation = str(metadata.get("ligation_strategy", np.asarray(["unknown"] * dataset.num_samples))[idx])
    status = str(metadata.get("modification_status", np.asarray(["unknown"] * dataset.num_samples))[idx])
    run = str(metadata.get("run_id", np.asarray([dataset.name] * dataset.num_samples))[idx])
    return motif, ligation, status or cls, run


def select_splits(
    datasets: Sequence[SourceDataset],
    specs: Sequence[DatasetSpec],
    *,
    valid_fraction: float,
    balance_mode: str,
    balance_validation: bool,
    source_class_cap: int | None,
    rng: np.random.Generator,
    show_progress: bool,
    scan_workers: int,
) -> tuple[list[SelectedSample], list[SelectedSample], dict]:
    train_candidates: list[tuple[SelectedSample, str, str]] = []
    valid_candidates: list[tuple[SelectedSample, str, str]] = []
    class_counts = {"positive": 0, "negative": 0}
    strata: dict[tuple, list[tuple[SelectedSample, str, str]]] = {}

    workers = max(1, int(scan_workers))
    if workers == 1 or len(specs) == 1:
        scan_results = [
            scan_dataset_labels(dataset_idx, spec)
            for dataset_idx, spec in tqdm(
                list(enumerate(specs)),
                desc="scan:datasets",
                unit="dataset",
                ascii=True,
                ncols=100,
                disable=not show_progress,
            )
        ]
    else:
        scan_results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as executor:
            futures = {
                executor.submit(scan_dataset_labels, dataset_idx, spec): spec.name
                for dataset_idx, spec in enumerate(specs)
            }
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="scan:datasets",
                unit="dataset",
                ascii=True,
                ncols=100,
                disable=not show_progress,
            )
            for future in iterator:
                scan_results.append(future.result())
        scan_results.sort(key=lambda result: result.dataset_idx)

    for result in scan_results:
        class_counts["positive"] += int(result.class_counts.get("positive", 0))
        class_counts["negative"] += int(result.class_counts.get("negative", 0))
        if show_progress:
            print(
                "      "
                f"scanned {result.name}: labeled={result.labeled_count} "
                f"positive={result.class_counts.get('positive', 0)} "
                f"negative={result.class_counts.get('negative', 0)}"
            )
        for pos in range(result.labeled_count):
            source_idx = int(result.source_indices[pos])
            cls = class_name(int(result.class_codes[pos]))
            motif = str(result.motifs[pos])
            status = str(result.statuses[pos]) or cls
            key = (motif, str(result.ligations[pos]), status, str(result.runs[pos]))
            item = (SelectedSample(result.dataset_idx, source_idx), cls, motif)
            strata.setdefault(key, []).append(item)

    stratum_items = tqdm(
        list(strata.items()),
        desc="split:strata",
        unit="stratum",
        ascii=True,
        ncols=100,
        disable=not show_progress,
    )
    for key, items in stratum_items:
        items = list(items)
        rng.shuffle(items)
        valid_count = int(round(len(items) * valid_fraction))
        if valid_fraction > 0 and len(items) > 1:
            valid_count = min(max(valid_count, 1), len(items) - 1)
        valid_candidates.extend(items[:valid_count])
        train_candidates.extend(items[valid_count:])

    def balance_source_class(
        candidates: Sequence[tuple[SelectedSample, str, str]],
        cap: int | None,
    ) -> tuple[list[SelectedSample], dict[str, object]]:
        by_source_class: dict[tuple[int, str], list[SelectedSample]] = {}
        for sample, cls, _motif in candidates:
            by_source_class.setdefault((sample.dataset_idx, cls), []).append(sample)

        required_groups = []
        for dataset_idx, _dataset in enumerate(datasets):
            for cls in ("positive", "negative"):
                key = (dataset_idx, cls)
                if key in by_source_class:
                    required_groups.append(key)

        if not required_groups:
            return [], {"source_class_group_counts": {}, "source_class_selected_per_group": {}}

        group_sizes = {key: len(by_source_class[key]) for key in required_groups}
        keep = min(group_sizes.values())
        if cap is not None:
            keep = min(keep, int(cap))
        selected: list[SelectedSample] = []
        if keep > 0:
            for key in required_groups:
                samples = list(by_source_class[key])
                rng.shuffle(samples)
                selected.extend(samples[:keep])
        details = {
            "source_class_cap": None if cap is None else int(cap),
            "source_class_keep_per_group": int(keep),
            "source_class_group_counts": {
                f"{datasets[dataset_idx].name}:{cls}": int(count)
                for (dataset_idx, cls), count in sorted(group_sizes.items())
            },
            "source_class_selected_per_group": {
                f"{datasets[dataset_idx].name}:{cls}": int(keep)
                for dataset_idx, cls in sorted(required_groups)
            },
        }
        return selected, details

    balance_details: dict[str, object] = {}
    if balance_mode == BALANCE_MODE_MOTIF:
        by_motif_class: dict[tuple[str, str], list[SelectedSample]] = {}
        for sample, cls, motif in train_candidates:
            by_motif_class.setdefault((motif, cls), []).append(sample)
        train_selected = []
        motif_counts = {}
        motif_selected = {}
        motifs = sorted({motif for motif, _ in by_motif_class})
        motif_items = tqdm(
            motifs,
            desc="balance:motifs",
            unit="motif",
            ascii=True,
            ncols=100,
            disable=not show_progress,
        )
        for motif in motif_items:
            positives = list(by_motif_class.get((motif, "positive"), []))
            negatives = list(by_motif_class.get((motif, "negative"), []))
            motif_counts[motif] = {"positive": len(positives), "negative": len(negatives)}
            if not positives or not negatives:
                continue
            keep = min(len(positives), len(negatives))
            rng.shuffle(positives)
            rng.shuffle(negatives)
            train_selected.extend(positives[:keep])
            train_selected.extend(negatives[:keep])
            motif_selected[motif] = {"positive": keep, "negative": keep}
        balance_details = {
            "motif_group_counts": motif_counts,
            "motif_selected_per_group": motif_selected,
        }
    elif balance_mode == BALANCE_MODE_SOURCE_CLASS:
        train_selected, balance_details = balance_source_class(train_candidates, source_class_cap)
    elif balance_mode == BALANCE_MODE_NONE:
        train_selected = [sample for sample, _, _ in train_candidates]
    else:
        raise ValueError(f"Unsupported balance mode: {balance_mode}")

    validation_balance_details: dict[str, object] = {}
    if balance_validation:
        if balance_mode != BALANCE_MODE_SOURCE_CLASS:
            raise ValueError("--balance-validation currently requires --balance-mode source-class")
        valid_selected, validation_balance_details = balance_source_class(valid_candidates, None)
        validation_balance_details = {
            f"validation_{key}": value for key, value in validation_balance_details.items()
        }
    else:
        valid_selected = [sample for sample, _, _ in valid_candidates]

    rng.shuffle(train_selected)
    rng.shuffle(valid_selected)
    summary = {
        "input_labeled_class_counts": class_counts,
        "num_strata": int(len(strata)),
        "train_selected": int(len(train_selected)),
        "validation_selected": int(len(valid_selected)),
        "balance_train": bool(balance_mode == BALANCE_MODE_MOTIF),
        "balance_mode": str(balance_mode),
        "balance_validation": bool(balance_validation),
        "valid_fraction": float(valid_fraction),
    }
    summary.update(balance_details)
    summary.update(validation_balance_details)
    return train_selected, valid_selected, summary


def write_split(
    output_dir: Path,
    datasets: Sequence[SourceDataset],
    selected: Sequence[SelectedSample],
    name: str,
    *,
    show_progress: bool,
) -> dict:
    if not selected:
        raise ValueError(f"{name}: no samples selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_width = max(int(dataset.chunks.shape[1]) for dataset in datasets)
    reference_width = max(int(dataset.references.shape[1]) for dataset in datasets)
    mod_width = max(int(dataset.mod_targets.shape[1]) for dataset in datasets)
    if reference_width != mod_width:
        width = max(reference_width, mod_width)
        reference_width = mod_width = width
    total = len(selected)

    out_chunks = np.lib.format.open_memmap(output_dir / "chunks.npy", mode="w+", dtype=np.float16, shape=(total, chunk_width))
    out_refs = np.lib.format.open_memmap(output_dir / "references.npy", mode="w+", dtype=np.uint8, shape=(total, reference_width))
    out_lens = np.lib.format.open_memmap(output_dir / "reference_lengths.npy", mode="w+", dtype=np.uint16, shape=(total,))
    out_mods = np.lib.format.open_memmap(output_dir / "mod_targets.npy", mode="w+", dtype=np.int16, shape=(total, mod_width))

    metadata_fields = sorted(set().union(*(set(dataset.metadata) for dataset in datasets)))
    metadata_out = {field: [] for field in metadata_fields}
    source_names = []

    write_blocks = tqdm(
        range(0, total, COPY_BLOCK_SIZE),
        desc=f"write:{name}",
        unit="block",
        ascii=True,
        ncols=100,
        disable=not show_progress,
    )
    for out_start in write_blocks:
        out_end = min(out_start + COPY_BLOCK_SIZE, total)
        out_chunks[out_start:out_end] = 0
        out_refs[out_start:out_end] = 0
        out_mods[out_start:out_end] = IGNORE_INDEX
        for offset, selected_sample in enumerate(selected[out_start:out_end], start=out_start):
            dataset = datasets[selected_sample.dataset_idx]
            idx = selected_sample.source_idx
            chunk_len = int(dataset.chunks.shape[1])
            ref_len = int(dataset.references.shape[1])
            mod_len = int(dataset.mod_targets.shape[1])
            out_chunks[offset, :chunk_len] = dataset.chunks[idx]
            out_refs[offset, :ref_len] = dataset.references[idx]
            out_lens[offset] = dataset.lengths[idx]
            out_mods[offset, :mod_len] = dataset.mod_targets[idx]
            source_names.append(dataset.name)
            for field in metadata_fields:
                if field in dataset.metadata:
                    metadata_out[field].append(dataset.metadata[field][idx])
                else:
                    metadata_out[field].append("")

    del out_chunks, out_refs, out_lens, out_mods
    metadata_arrays = {}
    for field, values in metadata_out.items():
        array = np.asarray(values)
        metadata_arrays[field] = array
    metadata_arrays["source_dataset"] = np.asarray(source_names, dtype=str)
    np.savez(output_dir / "metadata.npz", **metadata_arrays)

    pos = 0
    neg = 0
    mixed = 0
    for selected_sample in selected:
        dataset = datasets[selected_sample.dataset_idx]
        cls = sample_class(dataset.mod_targets[selected_sample.source_idx], int(dataset.lengths[selected_sample.source_idx]))
        pos += int(cls == "positive")
        neg += int(cls == "negative")
        mixed += int(cls == "mixed")
    return {
        "name": name,
        "num_samples": int(total),
        "positive_samples": int(pos),
        "negative_samples": int(neg),
        "mixed_samples": int(mixed),
        "output_shapes": {
            "chunks": [int(total), int(chunk_width)],
            "references": [int(total), int(reference_width)],
            "mod_targets": [int(total), int(mod_width)],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True, help="<name>:<dataset_dir>, repeat for each TRAIN run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.25)
    parser.add_argument("--no-balance-train", dest="balance_train", action="store_false")
    parser.set_defaults(balance_train=True)
    parser.add_argument(
        "--balance-mode",
        choices=[BALANCE_MODE_MOTIF, BALANCE_MODE_NONE, BALANCE_MODE_SOURCE_CLASS],
        default=None,
        help=(
            "Training balance strategy. Default preserves legacy behavior: motif unless "
            "--no-balance-train is set, otherwise none. source-class balances each "
            "input dataset's positive and negative samples for mixed Stage 1 datasets."
        ),
    )
    parser.add_argument(
        "--source-class-cap",
        type=int,
        default=None,
        help="Optional maximum samples kept per source/class group when --balance-mode source-class is used.",
    )
    parser.add_argument(
        "--balance-validation",
        action="store_true",
        help="Also balance validation source/class groups. Requires --balance-mode source-class.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and progress messages.")
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=0,
        help="Parallel worker processes for label scanning. 0 chooses min(number of datasets, CPU count).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.valid_fraction < 0 or args.valid_fraction >= 1:
        raise ValueError("--valid-fraction must be in [0, 1)")
    if args.source_class_cap is not None and args.source_class_cap <= 0:
        raise ValueError("--source-class-cap must be positive when provided")
    show_progress = not bool(args.no_progress)
    rng = np.random.default_rng(args.seed)
    specs = [parse_dataset_spec(text) for text in args.dataset]
    balance_mode = args.balance_mode
    if balance_mode is None:
        balance_mode = BALANCE_MODE_MOTIF if bool(args.balance_train) else BALANCE_MODE_NONE
    scan_workers = int(args.scan_workers)
    if scan_workers <= 0:
        scan_workers = min(len(specs), os.cpu_count() or 1)
    datasets = []
    if show_progress:
        print(f"[1/4] Loading {len(specs)} input datasets...")
    for spec in specs:
        if show_progress:
            print(f"      loading {spec.name}: {spec.directory}")
        dataset = load_dataset(spec)
        datasets.append(dataset)
        if show_progress:
            print(
                "      "
                f"{dataset.name}: samples={dataset.num_samples} "
                f"chunks={tuple(dataset.chunks.shape)} "
                f"references={tuple(dataset.references.shape)} "
                f"mod_targets={tuple(dataset.mod_targets.shape)}"
            )
    if show_progress:
        print(f"[2/4] Scanning labels with {scan_workers} worker(s) and selecting train/validation splits...")
    train_selected, valid_selected, selection_summary = select_splits(
        datasets,
        specs,
        valid_fraction=float(args.valid_fraction),
        balance_mode=str(balance_mode),
        balance_validation=bool(args.balance_validation),
        source_class_cap=args.source_class_cap,
        rng=rng,
        show_progress=show_progress,
        scan_workers=scan_workers,
    )
    if show_progress:
        print(
            "      "
            f"labeled_counts={selection_summary['input_labeled_class_counts']} "
            f"strata={selection_summary['num_strata']} "
            f"train={selection_summary['train_selected']} "
            f"validation={selection_summary['validation_selected']} "
            f"balance_mode={selection_summary['balance_mode']} "
            f"balance_validation={selection_summary['balance_validation']}"
        )
        print("[3/4] Writing train split...")
    train_summary = write_split(args.output_dir, datasets, train_selected, "train", show_progress=show_progress)
    if show_progress:
        print("[4/4] Writing validation split...")
    valid_summary = write_split(
        args.output_dir / "validation",
        datasets,
        valid_selected,
        "validation",
        show_progress=show_progress,
    )
    summary = {
        "inputs": [{"name": spec.name, "directory": str(spec.directory.resolve())} for spec in specs],
        "selection": selection_summary,
        "train": train_summary,
        "validation": valid_summary,
    }
    (args.output_dir / "mafia_stage1_merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Merged mAFiA Stage 1 dataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()
