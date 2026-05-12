#!/usr/bin/env python3
"""
Build TetraMod Stage 1 datasets from mAFiA synthetic RNA oligo runs.

The mAFiA paper labels only the center DRACH A of each synthetic oligo:
unmodified oligo runs provide canonical_A labels and modified oligo runs
provide m6A labels.  This builder projects those center labels onto Dorado
basecalled reads with move tables, then writes Bonito/TetraMod numpy arrays:

    chunks.npy
    references.npy
    reference_lengths.npy
    mod_targets.npy
    metadata.npz

Input BAMs must be Dorado RNA002 basecalls with --emit-moves.  Input signal is
POD5, converted from FAST5 beforehand when needed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing
import os
import shutil
import tempfile
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pysam
from tqdm import tqdm

try:
    import create_dataset_dorado_ctc_like as ctc
except ImportError:  # pragma: no cover - package-style execution fallback.
    from gen_data import create_dataset_dorado_ctc_like as ctc


IGNORE_INDEX = -100
CANONICAL_A_LABEL = 0
M6A_LABEL = 4
DEFAULT_MIN_OLIGO_IDENTITY = 0.86
DEFAULT_MAX_OLIGO_MISMATCHES = 4
DEFAULT_MAX_CHUNKS = -1
DEFAULT_TASK_BATCH_SIZE = 16
DEFAULT_MAX_PENDING_BATCHES = 2
DEFAULT_MAX_SAMPLES_PER_WORKER_FILE = 512
MAX_OPEN_MERGE_CHUNKS = 32

EXTRA_METADATA_STRING_FIELDS = (
    "oligo_ids",
    "oligo_motifs",
    "oligo_orientations",
    "modification_status",
    "ligation_strategy",
)
EXTRA_METADATA_NUMERIC_FIELDS = {
    "labeled_center_count": np.int16,
    "positive_center_count": np.int16,
    "negative_center_count": np.int16,
}
ALL_METADATA_STRING_FIELDS = (*ctc.METADATA_STRING_FIELDS, *EXTRA_METADATA_STRING_FIELDS)
ALL_METADATA_NUMERIC_FIELDS = {**ctc.METADATA_NUMERIC_FIELDS, **EXTRA_METADATA_NUMERIC_FIELDS}


@dataclass(frozen=True)
class OligoSpec:
    oligo_id: str
    sequence: str
    center_index: int
    motif: str
    ligation_strategy: str = ""
    role: str = "train"

    @property
    def reversed_sequence(self) -> str:
        return self.sequence[::-1]

    @property
    def reversed_center_index(self) -> int:
        return len(self.sequence) - 1 - self.center_index


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    accession: str
    local_name: str
    modification_status: str
    ligation_strategy: str
    oligo_ids: tuple[str, ...]
    split: str = "train"
    modified_oligo_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchedUnit:
    oligo: OligoSpec
    start: int
    end: int
    center_index: int
    label: int
    orientation: str
    identity: float
    mismatches: int


@dataclass
class MafiaSample:
    signal: np.ndarray
    label: np.ndarray
    mod_target: np.ndarray
    label_len: int
    record_id: str
    pod5_read_id: str
    run_id: str
    contig: str
    ref_start: int
    ref_end: int
    ref_strand: int
    chunk_start: int
    chunk_end: int
    primary_site_key: str
    primary_site_pos: int
    kmer_context: str
    motif_context: str
    mean_qscore: float
    mapping_accuracy: float
    mapping_coverage: float
    oligo_ids: str
    oligo_motifs: str
    oligo_orientations: str
    modification_status: str
    ligation_strategy: str
    labeled_center_count: int
    positive_center_count: int
    negative_center_count: int


@dataclass(frozen=True)
class ProcessSettings:
    chunk_len: int
    overlap: int
    sample_type: str
    min_qscore: float | None
    clip_value: float
    norm_strategy: str
    pa_mean: float
    pa_std: float
    quantile_a: float
    quantile_b: float
    shift_multiplier: float
    scale_multiplier: float
    max_label_len: int | None
    min_oligo_identity: float
    max_oligo_mismatches: int
    allow_reverse_match: bool
    metadata_kmer: int


def normalize_sequence(sequence: str) -> str:
    return (
        str(sequence)
        .strip()
        .upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("U", "T")
    )


def parse_modified_sequence(sequence: str, center_index: int | None = None) -> tuple[str, int]:
    """
    Parse Supplementary Table style sequences.

    The table may represent the modified base as /m6A, [m6A], m6A, or plain A.
    If no modification token is present, center_index must be supplied or the
    center-most A is used.
    """
    text = str(sequence).strip().replace(" ", "")
    lowered = text.lower()
    token_starts = [
        lowered.find(token)
        for token in ("/m6a", "[m6a]", "(m6a)", "m6a")
        if lowered.find(token) >= 0
    ]
    if token_starts:
        token_start = min(token_starts)
        prefix = normalize_sequence(text[:token_start])
        canonical = normalize_sequence(
            text.replace("/m6A", "A")
            .replace("/m6a", "A")
            .replace("[m6A]", "A")
            .replace("[m6a]", "A")
            .replace("(m6A)", "A")
            .replace("(m6a)", "A")
            .replace("m6A", "A")
            .replace("m6a", "A")
        )
        parsed_center = len(prefix)
    else:
        canonical = normalize_sequence(text)
        if center_index is not None:
            parsed_center = int(center_index)
        else:
            a_positions = [idx for idx, base in enumerate(canonical) if base == "A"]
            if not a_positions:
                raise ValueError(f"Oligo sequence has no A center candidate: {sequence!r}")
            midpoint = (len(canonical) - 1) / 2.0
            parsed_center = min(a_positions, key=lambda idx: abs(idx - midpoint))

    if parsed_center < 0 or parsed_center >= len(canonical):
        raise ValueError(f"Center index {parsed_center} is outside oligo sequence {canonical!r}")
    if canonical[parsed_center] != "A":
        raise ValueError(f"mAFiA center label must land on A, got {canonical[parsed_center]!r} in {canonical!r}")
    return canonical, parsed_center


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_ids(text: str | None) -> tuple[str, ...]:
    if text is None:
        return ()
    return tuple(item.strip() for item in str(text).replace(";", ",").split(",") if item.strip())


def read_table(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        reader = csv.DictReader((row for row in handle if not row.lstrip().startswith("#")), delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest is missing a header row: {path}")
        return [dict(row) for row in reader]


def pick(row: dict[str, str], *names: str, default=None):
    lookup = {key.strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lookup.get(name.lower())
        if value is not None and str(value).strip() != "":
            return value
    return default


def load_oligo_manifest(path: Path) -> dict[str, OligoSpec]:
    oligos: dict[str, OligoSpec] = {}
    for row in read_table(path):
        oligo_id = str(pick(row, "oligo_id", "id", "name")).strip()
        sequence = pick(row, "sequence", "seq", "oligo_sequence")
        if not oligo_id or sequence is None:
            raise ValueError(f"Oligo manifest row requires oligo_id and sequence: {row}")
        center_raw = pick(row, "center_index", "center", default=None)
        center_index = None if center_raw is None else int(center_raw)
        canonical, parsed_center = parse_modified_sequence(sequence, center_index=center_index)
        motif = str(pick(row, "motif", "motif_context", default=ctc.centered_kmer(canonical, parsed_center, 5))).upper()
        oligos[oligo_id] = OligoSpec(
            oligo_id=oligo_id,
            sequence=canonical,
            center_index=parsed_center,
            motif=motif.replace("U", "T"),
            ligation_strategy=str(pick(row, "ligation_strategy", "ligation", default="")).strip(),
            role=str(pick(row, "role", "split", default="train")).strip().lower(),
        )
    if not oligos:
        raise ValueError(f"No oligos were loaded from {path}")
    return oligos


def normalize_status(value: str) -> str:
    status = str(value).strip().lower().replace("-", "_")
    aliases = {
        "mod": "modified",
        "m6a": "modified",
        "full_mod": "modified",
        "unm": "unmodified",
        "canonical": "unmodified",
        "ivt": "unmodified",
        "unmod": "unmodified",
        "test": "mixed",
    }
    status = aliases.get(status, status)
    if status not in {"modified", "unmodified", "mixed"}:
        raise ValueError(f"Unsupported modification_status={value!r}; expected modified/unmodified/mixed")
    return status


def load_run_manifest(path: Path) -> dict[str, RunSpec]:
    runs: dict[str, RunSpec] = {}
    for row in read_table(path):
        run_id = str(pick(row, "run_id", "run", "name")).strip()
        if not run_id:
            raise ValueError(f"Run manifest row requires run_id: {row}")
        runs[run_id] = RunSpec(
            run_id=run_id,
            accession=str(pick(row, "accession", "ena", default="")).strip(),
            local_name=str(pick(row, "local_name", "directory", "dir", default="")).strip(),
            modification_status=normalize_status(pick(row, "modification_status", "status", "label")),
            ligation_strategy=str(pick(row, "ligation_strategy", "ligation", default="")).strip().lower(),
            oligo_ids=split_ids(pick(row, "oligo_ids", "oligos")),
            split=str(pick(row, "split", "role", default="train")).strip().lower(),
            modified_oligo_ids=split_ids(pick(row, "modified_oligo_ids", "positive_oligo_ids", default="")),
        )
    if not runs:
        raise ValueError(f"No runs were loaded from {path}")
    return runs


def label_for_match(run: RunSpec, oligo_id: str) -> int | None:
    if run.modification_status == "modified":
        return M6A_LABEL
    if run.modification_status == "unmodified":
        return CANONICAL_A_LABEL
    if oligo_id in set(run.modified_oligo_ids):
        return M6A_LABEL
    if run.modified_oligo_ids:
        return CANONICAL_A_LABEL
    return None


def mismatch_count(a: str, b: str) -> int:
    return sum(left != right for left, right in zip(a, b))


def find_oligo_units(
    signal_order_sequence: str,
    oligos: Sequence[OligoSpec],
    run: RunSpec,
    *,
    min_identity: float,
    max_mismatches: int,
    allow_reverse_match: bool,
) -> list[MatchedUnit]:
    sequence = normalize_sequence(signal_order_sequence)
    candidates: list[MatchedUnit] = []
    for oligo in oligos:
        label = label_for_match(run, oligo.oligo_id)
        if label is None:
            continue
        patterns = [("+", oligo.sequence, oligo.center_index)]
        if allow_reverse_match:
            patterns.append(("-", oligo.reversed_sequence, oligo.reversed_center_index))
        for orientation, pattern, center_offset in patterns:
            if len(pattern) > len(sequence):
                continue
            for start in range(0, len(sequence) - len(pattern) + 1):
                window = sequence[start:start + len(pattern)]
                mismatches = mismatch_count(window, pattern)
                identity = 1.0 - (mismatches / len(pattern))
                if mismatches <= max_mismatches and identity >= min_identity:
                    candidates.append(
                        MatchedUnit(
                            oligo=oligo,
                            start=start,
                            end=start + len(pattern),
                            center_index=start + center_offset,
                            label=label,
                            orientation=orientation,
                            identity=float(identity),
                            mismatches=int(mismatches),
                        )
                    )

    candidates.sort(key=lambda item: (-item.identity, item.mismatches, item.start, item.oligo.oligo_id))
    selected: list[MatchedUnit] = []
    occupied: list[tuple[int, int]] = []
    for candidate in candidates:
        if any(candidate.start < end and candidate.end > start for start, end in occupied):
            continue
        selected.append(candidate)
        occupied.append((candidate.start, candidate.end))
    selected.sort(key=lambda item: item.start)
    return selected


def assign_units_to_windows(
    units: Sequence[MatchedUnit],
    emitted_positions: np.ndarray,
    windows: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], list[MatchedUnit]]:
    assigned: dict[tuple[int, int], list[MatchedUnit]] = {}
    for unit in units:
        if unit.center_index < 0 or unit.center_index >= emitted_positions.shape[0]:
            continue
        center_signal = int(emitted_positions[unit.center_index])
        containing = [window for window in windows if window[0] <= center_signal < window[1]]
        if not containing:
            continue
        best = max(containing, key=lambda window: min(center_signal - window[0], window[1] - center_signal))
        assigned.setdefault(best, []).append(unit)
    return assigned


def primary_center_metadata(target_seq: str, units: Sequence[MatchedUnit], local_indices: Sequence[int], kmer_size: int):
    if not units:
        return "no_A", -1, "no_A", "no_A"
    primary_pos = int(local_indices[0])
    unit = units[0]
    return (
        f"{unit.oligo.oligo_id}:{primary_pos}:1:A",
        primary_pos,
        ctc.centered_kmer(target_seq, primary_pos, kmer_size),
        unit.oligo.motif,
    )


def build_samples_for_task(
    task: ctc.TaskData,
    oligos: Sequence[OligoSpec],
    run: RunSpec,
    settings: ProcessSettings,
) -> tuple[list[MafiaSample], dict[str, int]]:
    stats = {
        "records_seen": 1,
        "records_missing_signal": 0,
        "records_invalid_interval": 0,
        "records_invalid_move_table": 0,
        "records_move_seq_mismatch": 0,
        "records_signal_too_short": 0,
        "records_low_qscore": 0,
        "records_no_oligo_match": 0,
        "chunks_seen": 0,
        "chunks_written": 0,
        "chunks_no_label": 0,
        "chunks_too_long": 0,
        "labeled_centers": 0,
        "positive_centers": 0,
        "negative_centers": 0,
    }
    samples: list[MafiaSample] = []

    try:
        raw_signal = ctc.fetch_calibrated_signal(task.pod5_read_id)
    except Exception:
        stats["records_missing_signal"] += 1
        return samples, stats

    interval_start = task.sp + task.ts
    interval_end = task.sp + task.ns
    if interval_start < 0 or interval_end <= interval_start or interval_end > raw_signal.shape[0]:
        stats["records_invalid_interval"] += 1
        return samples, stats
    interval_signal = raw_signal[interval_start:interval_end]
    if interval_signal.shape[0] < settings.chunk_len:
        stats["records_signal_too_short"] += 1
        return samples, stats
    if settings.min_qscore is not None and task.mean_qscore is not None and task.mean_qscore < settings.min_qscore:
        stats["records_low_qscore"] += 1
        return samples, stats

    try:
        stride, moves = ctc.decode_move_table(task.mv_tag)
        signal_order_sequence = ctc.build_signal_order_sequence(
            task.query_sequence,
            settings.sample_type,
            task.is_reverse,
        )
    except Exception:
        stats["records_invalid_move_table"] += 1
        return samples, stats

    emitted_positions = []
    for step_idx, move_count in enumerate(moves):
        position = (step_idx * stride) + (stride // 2)
        for _ in range(move_count):
            emitted_positions.append(position)
    if len(emitted_positions) != len(signal_order_sequence):
        stats["records_move_seq_mismatch"] += 1
        return samples, stats
    if not emitted_positions:
        stats["records_invalid_move_table"] += 1
        return samples, stats
    emitted_positions_arr = np.asarray(emitted_positions, dtype=np.int64)

    units = find_oligo_units(
        signal_order_sequence,
        oligos,
        run,
        min_identity=settings.min_oligo_identity,
        max_mismatches=settings.max_oligo_mismatches,
        allow_reverse_match=settings.allow_reverse_match,
    )
    if not units:
        stats["records_no_oligo_match"] += 1
        return samples, stats

    task_for_norm = ctc.TaskData(
        **{
            **task.__dict__,
            "chunk_len": settings.chunk_len,
            "overlap": settings.overlap,
            "min_qscore": settings.min_qscore,
            "clip_value": settings.clip_value,
            "max_label_len": settings.max_label_len,
            "norm_strategy": settings.norm_strategy,
            "pa_mean": settings.pa_mean,
            "pa_std": settings.pa_std,
            "quantile_a": settings.quantile_a,
            "quantile_b": settings.quantile_b,
            "shift_multiplier": settings.shift_multiplier,
            "scale_multiplier": settings.scale_multiplier,
            "metadata_kmer": settings.metadata_kmer,
        }
    )
    try:
        normalised_signal = ctc.normalise_interval_signal(interval_signal, task_for_norm)
    except Exception:
        stats["records_invalid_interval"] += 1
        return samples, stats

    windows = list(ctc.chunk_windows(len(normalised_signal), settings.chunk_len, settings.overlap))
    unit_by_window = assign_units_to_windows(units, emitted_positions_arr, windows)
    for win_start, win_end in windows:
        stats["chunks_seen"] += 1
        window_units = unit_by_window.get((win_start, win_end), [])
        if not window_units:
            stats["chunks_no_label"] += 1
            continue
        left_idx = int(np.searchsorted(emitted_positions_arr, win_start, side="left"))
        right_idx = int(np.searchsorted(emitted_positions_arr, win_end, side="left"))
        if left_idx >= right_idx:
            stats["chunks_no_label"] += 1
            continue

        target_seq = signal_order_sequence[left_idx:right_idx]
        target = ctc.encode_reference(target_seq)
        if np.any(target == 0):
            stats["chunks_no_label"] += 1
            continue
        if settings.max_label_len is not None and target.shape[0] > settings.max_label_len:
            stats["chunks_too_long"] += 1
            continue

        mod_target = np.full(target.shape, IGNORE_INDEX, dtype=np.int16)
        local_indices = []
        kept_units = []
        for unit in window_units:
            local_idx = int(unit.center_index - left_idx)
            if local_idx < 0 or local_idx >= target.shape[0] or target[local_idx] != ctc.BASE_TO_INT["A"]:
                continue
            mod_target[local_idx] = unit.label
            local_indices.append(local_idx)
            kept_units.append(unit)
        if not kept_units:
            stats["chunks_no_label"] += 1
            continue

        positive_count = sum(1 for unit in kept_units if unit.label == M6A_LABEL)
        negative_count = sum(1 for unit in kept_units if unit.label == CANONICAL_A_LABEL)
        site_key, site_pos, kmer_context, motif_context = primary_center_metadata(
            target_seq,
            kept_units,
            local_indices,
            settings.metadata_kmer,
        )
        samples.append(
            MafiaSample(
                signal=normalised_signal[win_start:win_end],
                label=target.astype(np.uint8),
                mod_target=mod_target,
                label_len=int(target.shape[0]),
                record_id=task.record_id,
                pod5_read_id=task.pod5_read_id,
                run_id=run.run_id,
                contig="mafia_synthetic",
                ref_start=int(left_idx),
                ref_end=int(right_idx),
                ref_strand=1,
                chunk_start=int(win_start),
                chunk_end=int(win_end),
                primary_site_key=site_key,
                primary_site_pos=int(site_pos),
                kmer_context=kmer_context,
                motif_context=motif_context,
                mean_qscore=float(task.mean_qscore) if task.mean_qscore is not None else float("nan"),
                mapping_accuracy=float(np.mean([unit.identity for unit in kept_units])),
                mapping_coverage=1.0,
                oligo_ids=",".join(unit.oligo.oligo_id for unit in kept_units),
                oligo_motifs=",".join(unit.oligo.motif for unit in kept_units),
                oligo_orientations=",".join(unit.orientation for unit in kept_units),
                modification_status=run.modification_status,
                ligation_strategy=run.ligation_strategy,
                labeled_center_count=len(kept_units),
                positive_center_count=positive_count,
                negative_center_count=negative_count,
            )
        )
        stats["chunks_written"] += 1
        stats["labeled_centers"] += len(kept_units)
        stats["positive_centers"] += positive_count
        stats["negative_centers"] += negative_count

    return samples, stats


def merge_counters(total: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + int(value)


def write_metadata_npz(path: str | Path, samples: Sequence[MafiaSample]) -> None:
    arrays = {}
    for field in ALL_METADATA_STRING_FIELDS:
        arrays[field] = np.asarray([getattr(sample, field) for sample in samples], dtype=str)
    for field, dtype in ALL_METADATA_NUMERIC_FIELDS.items():
        arrays[field] = np.asarray([getattr(sample, field) for sample in samples], dtype=dtype)
    np.savez(path, **arrays)


def write_worker_samples(samples: Sequence[MafiaSample]) -> dict[str, object] | None:
    if not samples:
        return None
    temp_dir = ctc.WorkerState.temp_dir
    if temp_dir is None:
        raise RuntimeError("Worker temp_dir is not initialised.")
    signals = np.stack([sample.signal for sample in samples], axis=0)
    lengths = np.asarray([sample.label_len for sample in samples], dtype=np.uint16)
    offsets = np.zeros((len(samples) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    labels_flat = np.concatenate([sample.label for sample in samples], axis=0)
    mods_flat = np.concatenate([sample.mod_target for sample in samples], axis=0)

    def temp_path(prefix: str, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, dir=temp_dir, delete=False) as handle:
            return handle.name

    sig_path = temp_path("mafia_sig_", ".npy")
    lbl_path = temp_path("mafia_lbl_", ".npy")
    mod_path = temp_path("mafia_mod_", ".npy")
    off_path = temp_path("mafia_off_", ".npy")
    len_path = temp_path("mafia_len_", ".npy")
    meta_path = temp_path("mafia_meta_", ".npz")
    np.save(sig_path, signals)
    np.save(lbl_path, labels_flat)
    np.save(mod_path, mods_flat)
    np.save(off_path, offsets)
    np.save(len_path, lengths)
    write_metadata_npz(meta_path, samples)
    return {
        "signals": sig_path,
        "labels": lbl_path,
        "mod_targets": mod_path,
        "offsets": off_path,
        "lengths": len_path,
        "metadata": meta_path,
        "num_samples": int(signals.shape[0]),
    }


def process_task_batch(
    tasks: Sequence[ctc.TaskData],
    oligos: Sequence[OligoSpec],
    run: RunSpec,
    settings: ProcessSettings,
    max_samples_per_file: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    stats: dict[str, int] = {}
    batch_samples: list[MafiaSample] = []
    manifest_entries: list[dict[str, object]] = []
    for task in tasks:
        samples, sample_stats = build_samples_for_task(task, oligos, run, settings)
        merge_counters(stats, sample_stats)
        batch_samples.extend(samples)
        if len(batch_samples) >= max_samples_per_file:
            entry = write_worker_samples(batch_samples)
            if entry is not None:
                manifest_entries.append(entry)
            batch_samples = []
    if batch_samples:
        entry = write_worker_samples(batch_samples)
        if entry is not None:
            manifest_entries.append(entry)
    return manifest_entries, stats


def close_maybe_mmap(array) -> None:
    if hasattr(array, "close"):
        array.close()
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def merge_chunks_to_final(
    output_dir: Path,
    chunk_manifest: list[dict[str, object]],
    signal_len: int,
    max_label_len: int | None,
    max_chunks: int,
) -> dict[str, int]:
    if not chunk_manifest:
        raise RuntimeError("No valid mAFiA synthetic samples remained after filtering.")

    lengths_list = [np.load(info["lengths"]) for info in tqdm(chunk_manifest, desc="Loading lengths", ascii=True, ncols=100)]
    lengths = np.concatenate(lengths_list, axis=0)
    keep_indices = ctc.typical_indices(lengths)
    if keep_indices.size == 0:
        raise RuntimeError("No samples remained after typical-length filtering.")
    keep_indices = np.random.permutation(keep_indices)
    if max_chunks > 0:
        keep_indices = keep_indices[:max_chunks]
    if keep_indices.size == 0:
        raise RuntimeError("No samples remained after max_chunks.")
    if max_label_len is None:
        max_label_len = int(lengths[keep_indices].max())

    total_samples = int(keep_indices.size)
    chunks_out = np.lib.format.open_memmap(output_dir / "chunks.npy", mode="w+", dtype=np.float16, shape=(total_samples, signal_len))
    refs_out = np.lib.format.open_memmap(output_dir / "references.npy", mode="w+", dtype=np.uint8, shape=(total_samples, max_label_len))
    mods_out = np.lib.format.open_memmap(output_dir / "mod_targets.npy", mode="w+", dtype=np.int16, shape=(total_samples, max_label_len))
    lens_out = np.lib.format.open_memmap(output_dir / "reference_lengths.npy", mode="w+", dtype=np.uint16, shape=(total_samples,))

    starts = ctc.build_chunk_ranges(chunk_manifest)
    metadata_out = {field: [] for field in ALL_METADATA_STRING_FIELDS}
    metadata_out.update({field: [] for field in ALL_METADATA_NUMERIC_FIELDS})
    chunk_cache: OrderedDict[int, tuple] = OrderedDict()

    def load_chunk(idx: int):
        if idx in chunk_cache:
            chunk_cache.move_to_end(idx)
            return chunk_cache[idx]
        if len(chunk_cache) >= MAX_OPEN_MERGE_CHUNKS:
            _, old = chunk_cache.popitem(last=False)
            for item in old:
                close_maybe_mmap(item)
        info = chunk_manifest[idx]
        chunk_cache[idx] = (
            np.load(info["signals"], mmap_mode="r"),
            np.load(info["labels"], mmap_mode="r"),
            np.load(info["mod_targets"], mmap_mode="r"),
            np.load(info["offsets"], mmap_mode="r"),
            np.load(info["lengths"], mmap_mode="r"),
            np.load(info["metadata"]),
        )
        return chunk_cache[idx]

    block_size = 2048
    for out_start in tqdm(range(0, total_samples, block_size), desc="Writing dataset", ascii=True, ncols=100):
        out_end = min(out_start + block_size, total_samples)
        block_indices = keep_indices[out_start:out_end]
        block_signals = np.empty((out_end - out_start, signal_len), dtype=np.float16)
        block_refs = np.zeros((out_end - out_start, max_label_len), dtype=np.uint8)
        block_mods = np.full((out_end - out_start, max_label_len), IGNORE_INDEX, dtype=np.int16)
        block_lengths = lengths[block_indices].astype(np.uint16)
        for pos, global_idx in enumerate(block_indices):
            chunk_idx = ctc.find_chunk_index(starts, int(global_idx))
            local_idx = int(global_idx - starts[chunk_idx])
            signals, labels, mods, offsets, lengths_chunk, metadata = load_chunk(chunk_idx)
            label_len = int(lengths_chunk[local_idx])
            label_start = int(offsets[local_idx])
            block_signals[pos] = signals[local_idx]
            block_refs[pos, :label_len] = labels[label_start:label_start + label_len]
            block_mods[pos, :label_len] = mods[label_start:label_start + label_len]
            for field in metadata_out:
                metadata_out[field].append(metadata[field][local_idx])
        chunks_out[out_start:out_end] = block_signals
        refs_out[out_start:out_end] = block_refs
        mods_out[out_start:out_end] = block_mods
        lens_out[out_start:out_end] = block_lengths

    del chunks_out, refs_out, mods_out, lens_out
    metadata_arrays = {}
    for field in ALL_METADATA_STRING_FIELDS:
        metadata_arrays[field] = np.asarray(metadata_out[field], dtype=str)
    for field, dtype in ALL_METADATA_NUMERIC_FIELDS.items():
        metadata_arrays[field] = np.asarray(metadata_out[field], dtype=dtype)
    np.savez(output_dir / "metadata.npz", **metadata_arrays)
    (output_dir / "metadata_fields.json").write_text(
        json.dumps(
            {
                "primary_site_key": "Synthetic center A label encoded as oligo_id:target_pos:1:A.",
                "mod_targets": "Only mAFiA oligo center DRACH A positions are labeled; all other sites are -100.",
                "oligo_ids": "Comma-separated center-labeled oligos in this chunk.",
                "oligo_motifs": "Comma-separated center motif labels for labeled centers.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for cached in chunk_cache.values():
        for item in cached:
            close_maybe_mmap(item)
    for info in chunk_manifest:
        for key in ("signals", "labels", "mod_targets", "offsets", "lengths", "metadata"):
            os.remove(info[key])

    positive = int(np.sum(metadata_arrays["positive_center_count"].astype(np.int64)))
    negative = int(np.sum(metadata_arrays["negative_center_count"].astype(np.int64)))
    return {
        "total_pre_typical_filter": int(lengths.shape[0]),
        "total_post_typical_filter": int(keep_indices.size),
        "total_written": total_samples,
        "max_label_len": int(max_label_len),
        "positive_centers": positive,
        "negative_centers": negative,
    }


def write_template_manifest(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mafia_oligos.tsv").write_text(
        "\n".join(
            [
                "oligo_id\tsequence\tmotif\tligation_strategy\trole",
                "# Fill sequence from Supplementary Table 1. Use /m6A at the controlled center or add center_index.",
                "# Example only; do not use placeholder rows for training.",
                "RL_EXAMPLE\tFILL_FROM_SUPP_TABLE_1\tGGACT\trandom_ligation\ttrain",
                "SL_EXAMPLE\tFILL_FROM_SUPP_TABLE_1\tGGACT\tsplint_ligation\ttrain",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "mafia_runs.tsv").write_text(
        "\n".join(
            [
                "run_id\taccession\tlocal_name\tmodification_status\tligation_strategy\tsplit\toligo_ids\tmodified_oligo_ids",
                "# Fill from Supplementary Table 3 and downloaded directory names.",
                "# modification_status is modified, unmodified, or mixed. For mixed runs, set modified_oligo_ids.",
                "RUN_UNMOD_EXAMPLE\tERRxxxx\tlocal_dir\tunmodified\trandom_ligation\ttrain\tRL_EXAMPLE\t",
                "RUN_MOD_EXAMPLE\tERRyyyy\tlocal_dir\tmodified\trandom_ligation\ttrain\tRL_EXAMPLE\t",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def settings_from_args(args) -> ProcessSettings:
    return ProcessSettings(
        chunk_len=int(args.chunk_len),
        overlap=int(args.overlap),
        sample_type=args.sample_type,
        min_qscore=None if args.min_qscore is None else float(args.min_qscore),
        clip_value=float(args.clip_value),
        norm_strategy=args.norm_strategy,
        pa_mean=float(args.pa_mean),
        pa_std=float(args.pa_std),
        quantile_a=float(args.quantile_a),
        quantile_b=float(args.quantile_b),
        shift_multiplier=float(args.shift_multiplier),
        scale_multiplier=float(args.scale_multiplier),
        max_label_len=None if args.max_label_len is None else int(args.max_label_len),
        min_oligo_identity=float(args.min_oligo_identity),
        max_oligo_mismatches=int(args.max_oligo_mismatches),
        allow_reverse_match=bool(args.allow_reverse_match),
        metadata_kmer=int(args.metadata_kmer),
    )


def worker_init(pod5_lookup: dict, temp_dir: str) -> None:
    os.environ["TMPDIR"] = "/tmp"
    os.environ["TMP"] = "/tmp"
    os.environ["TEMP"] = "/tmp"
    tempfile.tempdir = "/tmp"
    ctc.WorkerState.pod5_lookup = pod5_lookup
    ctc.WorkerState.pod5_reader_cache = OrderedDict()
    ctc.WorkerState.temp_dir = temp_dir


def build_task_from_read(read: pysam.AlignedSegment, args) -> ctc.TaskData | None:
    return ctc.build_task(read, args)


def write_summary(output_dir: Path, args, run: RunSpec, oligos: Sequence[OligoSpec], counters, merge_summary):
    summary = {
        "builder": "mafia_synthetic_stage1",
        "bam_file": str(Path(args.bam_file).resolve()),
        "pod5_dir": str(Path(args.pod5_dir).resolve()),
        "run": run.__dict__,
        "oligo_ids": [item.oligo_id for item in oligos],
        "output_dir": str(output_dir.resolve()),
        "sample_type": args.sample_type,
        "chunk_len": int(args.chunk_len),
        "overlap": int(args.overlap),
        "min_oligo_identity": float(args.min_oligo_identity),
        "max_oligo_mismatches": int(args.max_oligo_mismatches),
        "allow_reverse_match": bool(args.allow_reverse_match),
        "counters": {key: int(value) for key, value in sorted(counters.items())},
        "merge": merge_summary,
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bam-file", help="Dorado BAM produced with --emit-moves.")
    parser.add_argument("--pod5-dir", help="POD5 directory converted from FAST5.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--oligo-manifest", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--run-id", help="run_id from --run-manifest to process.")
    parser.add_argument("--write-template-manifest", type=Path, default=None)
    parser.add_argument("--sample-type", choices=["rna", "dna"], default="rna")
    parser.add_argument("--chunk-len", type=int, default=10000)
    parser.add_argument("--overlap", type=int, default=500)
    parser.add_argument("--max-label-len", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="Unused compatibility field for Dorado task construction.")
    parser.add_argument("--min-coverage", type=float, default=0.0, help="Unused compatibility field for Dorado task construction.")
    parser.add_argument("--min-qscore", type=float, default=None)
    parser.add_argument("--clip-value", type=float, default=5.0)
    parser.add_argument("--rna002", action="store_true", default=False)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--norm-strategy", choices=["from-bam", "pa", "quantile", "model-config"], default=None)
    parser.add_argument("--pa-mean", type=float, default=0.0)
    parser.add_argument("--pa-std", type=float, default=1.0)
    parser.add_argument("--quantile-a", type=float, default=ctc.DEFAULT_NORM_PARAMS["quantile_a"])
    parser.add_argument("--quantile-b", type=float, default=ctc.DEFAULT_NORM_PARAMS["quantile_b"])
    parser.add_argument("--shift-multiplier", type=float, default=ctc.DEFAULT_NORM_PARAMS["shift_multiplier"])
    parser.add_argument("--scale-multiplier", type=float, default=ctc.DEFAULT_NORM_PARAMS["scale_multiplier"])
    parser.add_argument("--metadata-kmer", type=int, default=5)
    parser.add_argument("--min-oligo-identity", type=float, default=DEFAULT_MIN_OLIGO_IDENTITY)
    parser.add_argument("--max-oligo-mismatches", type=int, default=DEFAULT_MAX_OLIGO_MISMATCHES)
    parser.add_argument("--allow-reverse-match", action="store_true", default=True)
    parser.add_argument("--no-reverse-match", dest="allow_reverse_match", action="store_false")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task-batch-size", type=int, default=DEFAULT_TASK_BATCH_SIZE)
    parser.add_argument("--max-pending-batches", type=int, default=DEFAULT_MAX_PENDING_BATCHES)
    parser.add_argument("--max-samples-per-worker-file", type=int, default=DEFAULT_MAX_SAMPLES_PER_WORKER_FILE)
    parser.add_argument("--mp-start-method", choices=["auto", "fork", "spawn", "forkserver"], default="auto")
    return parser.parse_args(argv)


def validate_required_args(args) -> None:
    if args.write_template_manifest is not None:
        return
    required = ("bam_file", "pod5_dir", "output_dir", "oligo_manifest", "run_manifest", "run_id")
    missing = [name for name in required if getattr(args, name) in {None, ""}]
    if missing:
        raise ValueError(f"Missing required arguments unless --write-template-manifest is used: {missing}")


def main(argv=None):
    args = parse_args(argv)
    if args.write_template_manifest is not None:
        write_template_manifest(args.write_template_manifest)
        print(f"Wrote mAFiA manifest templates to: {args.write_template_manifest}")
        return
    validate_required_args(args)
    ctc.resolve_signal_normalisation(args)
    if args.metadata_kmer <= 0 or args.metadata_kmer % 2 == 0:
        raise ValueError(f"--metadata-kmer must be a positive odd integer, got {args.metadata_kmer}")
    np.random.seed(args.seed)

    oligo_manifest = load_oligo_manifest(args.oligo_manifest)
    run_manifest = load_run_manifest(args.run_manifest)
    if args.run_id not in run_manifest:
        raise KeyError(f"--run-id {args.run_id!r} was not found in {args.run_manifest}")
    run = run_manifest[args.run_id]
    missing_oligos = [oligo_id for oligo_id in run.oligo_ids if oligo_id not in oligo_manifest]
    if missing_oligos:
        raise KeyError(f"Run {run.run_id} references undefined oligos: {missing_oligos}")
    oligos = [oligo_manifest[oligo_id] for oligo_id in run.oligo_ids]
    settings = settings_from_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_chunks"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    print("[1/5] Scanning BAM for required POD5 read ids...")
    required_ids, candidate_records = ctc.collect_required_pod5_ids(args.bam_file, int(args.max_records))
    print(f"      Candidate BAM records: {candidate_records}")
    print("[2/5] Building filtered POD5 index...")
    lookup = ctc.build_pod5_lookup(Path(args.pod5_dir), required_read_ids=required_ids)
    print(f"      Indexed {len(lookup)} reads in POD5.")
    del required_ids
    gc.collect()

    start_method = ctc.resolve_mp_start_method(args.mp_start_method)
    workers = max(1, min(int(args.workers), max((os.cpu_count() or 1) - 2, 1)))
    task_batch_size = max(int(args.task_batch_size), 1)
    max_pending = max(int(args.max_pending_batches), 1)
    max_chunks = int(args.max_chunks)
    chunk_manifest: list[dict[str, object]] = []
    counters: dict[str, int] = {}
    task_batch = []
    task_count = 0

    print("[3/5] Dispatching BAM records...")
    if workers == 1:
        worker_init(lookup, str(temp_dir.resolve()))
        with pysam.AlignmentFile(args.bam_file, "rb", check_sq=False) as bam_file:
            for read in tqdm(bam_file, desc="Processing", unit="record", ascii=True, ncols=100):
                if args.max_records > 0 and task_count >= args.max_records:
                    break
                task = build_task_from_read(read, args)
                if task is None or task.pod5_read_id not in lookup:
                    continue
                task_count += 1
                entries, stats = process_task_batch((task,), oligos, run, settings, int(args.max_samples_per_worker_file))
                chunk_manifest.extend(entries)
                merge_counters(counters, stats)
                if max_chunks > 0 and counters.get("chunks_written", 0) >= max_chunks:
                    break
    else:
        mp_context = multiprocessing.get_context(start_method)
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
            initializer=worker_init,
            initargs=(lookup, str(temp_dir.resolve())),
        )
        try:
            with pysam.AlignmentFile(args.bam_file, "rb", check_sq=False) as bam_file:
                futures = set()
                stop_dispatch = False
                for read in tqdm(bam_file, desc="Dispatching", unit="record", ascii=True, ncols=100):
                    if stop_dispatch or (args.max_records > 0 and task_count >= args.max_records):
                        break
                    task = build_task_from_read(read, args)
                    if task is None or task.pod5_read_id not in lookup:
                        continue
                    task_batch.append(task)
                    task_count += 1
                    if len(task_batch) >= task_batch_size:
                        futures.add(
                            executor.submit(
                                process_task_batch,
                                tuple(task_batch),
                                tuple(oligos),
                                run,
                                settings,
                                int(args.max_samples_per_worker_file),
                            )
                        )
                        task_batch = []
                    if len(futures) >= max_pending:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        futures.difference_update(done)
                        for future in done:
                            entries, stats = future.result()
                            chunk_manifest.extend(entries)
                            merge_counters(counters, stats)
                            if max_chunks > 0 and counters.get("chunks_written", 0) >= max_chunks:
                                stop_dispatch = True
                if task_batch and not stop_dispatch:
                    futures.add(
                        executor.submit(
                            process_task_batch,
                            tuple(task_batch),
                            tuple(oligos),
                            run,
                            settings,
                            int(args.max_samples_per_worker_file),
                        )
                    )
                for future in tqdm(as_completed(futures), total=len(futures), desc="Finishing", ascii=True, ncols=100):
                    entries, stats = future.result()
                    chunk_manifest.extend(entries)
                    merge_counters(counters, stats)
            executor.shutdown(wait=True, cancel_futures=False)
        finally:
            del lookup
            gc.collect()

    print("[4/5] Merging passing chunks into final dataset...")
    merge_summary = merge_chunks_to_final(output_dir, chunk_manifest, int(args.chunk_len), args.max_label_len, max_chunks)
    print("[5/5] Writing summary...")
    write_summary(output_dir, args, run, oligos, counters, merge_summary)
    try:
        temp_dir.rmdir()
    except OSError:
        pass
    print(f"Dataset ready at: {output_dir}")
    print(f"Final stats: {merge_summary['total_written']} chunks written.")


if __name__ == "__main__":
    main()
