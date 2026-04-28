#!/usr/bin/env python3
"""
Generate TetraMod chunk/reference datasets for high-modification controls.

This is a parallel rescue path for control samples where whole-read BAM
alignment is usable, but the Bonito-like chunk-local remapping in
create_dataset_dorado_ctc_like.py is too brittle. It keeps the same Dorado
POD5/move-table signal slicing and output array format, but derives each
chunk's reference labels from the BAM CIGAR query-to-reference mapping instead
of remapping a short chunk sequence.

Use this for difficult 100% modification controls only after confirming that
the original Bonito-like dataset builder fails because chunk-local remapping
cannot handle low-identity modified basecalls.
"""

from __future__ import annotations

import argparse
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
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import mappy
import numpy as np
import pysam
from tqdm import tqdm

import create_dataset_dorado_ctc_like as ctc


STRICT_MIN_READ_IDENTITY = 0.80
STRICT_MIN_READ_ALIGNED_FRACTION = 0.85
STRICT_MIN_MAPQ = 20
STRICT_MIN_CHUNK_ALIGNED_FRACTION = 0.85
STRICT_MIN_CHUNK_BASE_IDENTITY = 0.70

RELAXED_MIN_READ_IDENTITY = 0.70
RELAXED_MIN_READ_ALIGNED_FRACTION = 0.75
RELAXED_MIN_MAPQ = 1
RELAXED_MIN_CHUNK_ALIGNED_FRACTION = 0.70
RELAXED_MIN_CHUNK_BASE_IDENTITY = 0.60

DEFAULT_MIN_REFERENCE_LEN = 25
DEFAULT_MAX_REFERENCE_SPAN_FACTOR = 2.5


@dataclass
class BamAlignedTask:
    record_id: str
    pod5_read_id: str
    run_id: str
    query_sequence: str
    sample_type: str
    is_reverse: bool
    contig: str
    ref_strand: int
    q_to_ref: Tuple[int, ...]
    read_identity: float
    read_aligned_fraction: float
    mapq: int
    ts: int
    ns: int
    sp: int
    mv_tag: Sequence[int]
    mean_qscore: Optional[float]
    scaling_shift: Optional[float]
    scaling_scale: Optional[float]
    chunk_len: int
    overlap: int
    min_read_identity: float
    min_read_aligned_fraction: float
    min_mapq: int
    min_chunk_aligned_fraction: float
    min_chunk_base_identity: float
    min_reference_len: int
    max_reference_span_factor: float
    require_a: bool
    min_qscore: Optional[float]
    clip_value: float
    max_label_len: Optional[int]
    norm_strategy: str
    pa_mean: float
    pa_std: float
    quantile_a: float
    quantile_b: float
    shift_multiplier: float
    scale_multiplier: float
    metadata_kmer: int


def compute_read_identity(read: pysam.AlignedSegment) -> Optional[float]:
    if not read.has_tag("NM") or not read.cigartuples:
        return None
    edit_distance = int(read.get_tag("NM"))
    aligned_bases = 0
    for op, length in read.cigartuples:
        if op in (0, 1, 2, 7, 8):
            aligned_bases += int(length)
    if aligned_bases <= 0:
        return None
    return max(0.0, 1.0 - (float(edit_distance) / float(aligned_bases)))


def build_query_to_ref(read: pysam.AlignedSegment, query_len: int) -> Tuple[int, ...]:
    q_to_ref = np.full((query_len,), -1, dtype=np.int64)
    for query_pos, ref_pos in read.get_aligned_pairs(matches_only=False):
        if query_pos is None or ref_pos is None:
            continue
        if 0 <= query_pos < query_len:
            q_to_ref[int(query_pos)] = int(ref_pos)
    return tuple(int(value) for value in q_to_ref)


def build_signal_order_query_positions(read_length: int, is_reverse: bool, sample_type: str) -> np.ndarray:
    signal_order = np.empty((read_length,), dtype=np.int64)
    for query_pos in range(read_length):
        signal_idx = ctc.query_pos_to_signal_idx(query_pos, read_length, is_reverse, sample_type)
        signal_order[signal_idx] = query_pos
    return signal_order


def reference_base_for_query_base(base: str, is_reverse: bool) -> str:
    base = base.upper()
    if is_reverse:
        return base.translate(ctc.COMPLEMENT)
    return base


def chunk_base_identity(
    query_sequence: str,
    q_positions: np.ndarray,
    q_to_ref: np.ndarray,
    ref_forward: str,
    ref_start: int,
    is_reverse: bool,
) -> float:
    matches = 0
    compared = 0
    for query_pos in q_positions:
        ref_pos = int(q_to_ref[int(query_pos)])
        if ref_pos < ref_start:
            continue
        ref_idx = ref_pos - ref_start
        if ref_idx < 0 or ref_idx >= len(ref_forward):
            continue
        query_base = reference_base_for_query_base(query_sequence[int(query_pos)], is_reverse)
        ref_base = ref_forward[ref_idx].upper()
        if query_base == ref_base:
            matches += 1
        compared += 1
    return float(matches) / float(compared) if compared else 0.0


def build_task(read: pysam.AlignedSegment, args) -> Optional[BamAlignedTask]:
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        return None
    if read.has_tag("dx") and int(read.get_tag("dx")) == 1:
        return None
    if not read.has_tag("mv") or not read.has_tag("ns"):
        return None

    query_sequence = (read.query_sequence or "").upper().replace("U", "T")
    if not query_sequence:
        return None
    query_len = len(query_sequence)

    read_identity = compute_read_identity(read)
    if read_identity is None:
        return None
    read_aligned_fraction = float(read.query_alignment_length or 0) / max(float(query_len), 1.0)
    q_to_ref = build_query_to_ref(read, query_len)

    split_parent = str(read.get_tag("pi")) if read.has_tag("pi") else None
    split_start = int(read.get_tag("sp")) if read.has_tag("sp") else 0
    pod5_read_id = split_parent if split_parent else read.query_name
    run_id = args.run_id or (str(read.get_tag("RG")) if read.has_tag("RG") else "unknown")

    return BamAlignedTask(
        record_id=str(read.query_name),
        pod5_read_id=str(pod5_read_id),
        run_id=str(run_id),
        query_sequence=query_sequence,
        sample_type=args.sample_type,
        is_reverse=bool(read.is_reverse),
        contig=str(read.reference_name),
        ref_strand=-1 if read.is_reverse else 1,
        q_to_ref=q_to_ref,
        read_identity=float(read_identity),
        read_aligned_fraction=float(read_aligned_fraction),
        mapq=int(read.mapping_quality),
        ts=int(read.get_tag("ts")) if read.has_tag("ts") else 0,
        ns=int(read.get_tag("ns")),
        sp=split_start,
        mv_tag=tuple(int(value) for value in read.get_tag("mv")),
        mean_qscore=float(read.get_tag("qs")) if read.has_tag("qs") else None,
        scaling_shift=float(read.get_tag("sm")) if read.has_tag("sm") else None,
        scaling_scale=float(read.get_tag("sd")) if read.has_tag("sd") else None,
        chunk_len=int(args.chunk_len),
        overlap=int(args.overlap),
        min_read_identity=float(args.min_read_identity),
        min_read_aligned_fraction=float(args.min_read_aligned_fraction),
        min_mapq=int(args.min_mapq),
        min_chunk_aligned_fraction=float(args.min_chunk_aligned_fraction),
        min_chunk_base_identity=float(args.min_chunk_base_identity),
        min_reference_len=int(args.min_reference_len),
        max_reference_span_factor=float(args.max_reference_span_factor),
        require_a=bool(args.require_a),
        min_qscore=float(args.min_qscore) if args.min_qscore is not None else None,
        clip_value=float(args.clip_value),
        max_label_len=int(args.max_label_len) if args.max_label_len is not None else None,
        norm_strategy=str(args.norm_strategy),
        pa_mean=float(args.pa_mean),
        pa_std=float(args.pa_std),
        quantile_a=float(args.quantile_a),
        quantile_b=float(args.quantile_b),
        shift_multiplier=float(args.shift_multiplier),
        scale_multiplier=float(args.scale_multiplier),
        metadata_kmer=int(args.metadata_kmer),
    )


def collect_required_pod5_ids(bam_path: str, max_records: int) -> tuple[set[str], int]:
    required: set[str] = set()
    candidate_records = 0
    total_est = ctc.get_total_reads_from_index(bam_path)
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam_file:
        for read in tqdm(bam_file, total=total_est or None, desc="Scanning BAM", unit="record", ascii=True, ncols=100):
            if max_records > 0 and candidate_records >= max_records:
                break
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.has_tag("dx") and int(read.get_tag("dx")) == 1:
                continue
            if not read.has_tag("mv") or not read.has_tag("ns"):
                continue
            candidate_records += 1
            required.add(str(read.get_tag("pi")) if read.has_tag("pi") else str(read.query_name))
    return required, candidate_records


def empty_stats() -> Dict[str, int]:
    return {
        "records_seen": 1,
        "records_missing_signal": 0,
        "records_invalid_interval": 0,
        "records_invalid_move_table": 0,
        "records_move_seq_mismatch": 0,
        "records_signal_too_short": 0,
        "records_low_qscore": 0,
        "records_low_mapq": 0,
        "records_low_read_identity": 0,
        "records_low_read_aligned_fraction": 0,
        "chunks_seen": 0,
        "chunks_written": 0,
        "chunks_zerolen_sequence": 0,
        "chunks_low_aligned_fraction": 0,
        "chunks_low_base_identity": 0,
        "chunks_short_reference": 0,
        "chunks_long_reference": 0,
        "chunks_no_reference": 0,
        "chunks_n_in_reference": 0,
        "chunks_no_a": 0,
        "chunks_too_long": 0,
    }


def process_task(task: BamAlignedTask) -> Tuple[List[ctc.Sample], Dict[str, int]]:
    stats = empty_stats()
    samples: List[ctc.Sample] = []

    if task.mapq < task.min_mapq:
        stats["records_low_mapq"] += 1
        return samples, stats
    if task.read_identity < task.min_read_identity:
        stats["records_low_read_identity"] += 1
        return samples, stats
    if task.read_aligned_fraction < task.min_read_aligned_fraction:
        stats["records_low_read_aligned_fraction"] += 1
        return samples, stats

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
    if interval_signal.shape[0] < task.chunk_len:
        stats["records_signal_too_short"] += 1
        return samples, stats

    if task.min_qscore is not None and task.mean_qscore is not None and task.mean_qscore < task.min_qscore:
        stats["records_low_qscore"] += 1
        return samples, stats

    try:
        stride, moves = ctc.decode_move_table(task.mv_tag)
        signal_order_query_positions = build_signal_order_query_positions(
            len(task.query_sequence),
            task.is_reverse,
            task.sample_type,
        )
    except Exception:
        stats["records_invalid_move_table"] += 1
        return samples, stats

    emitted_positions: List[int] = []
    for step_idx, move_count in enumerate(moves):
        position = (step_idx * stride) + (stride // 2)
        for _ in range(move_count):
            emitted_positions.append(position)

    if len(emitted_positions) != len(task.query_sequence):
        stats["records_move_seq_mismatch"] += 1
        return samples, stats
    if not emitted_positions:
        stats["records_invalid_move_table"] += 1
        return samples, stats

    try:
        normalised_signal = ctc.normalise_interval_signal(interval_signal, task)  # type: ignore[arg-type]
    except Exception:
        stats["records_invalid_interval"] += 1
        return samples, stats

    aligner = ctc.WorkerState.aligner
    if aligner is None:
        raise RuntimeError("Worker aligner is not initialised.")

    emitted_positions_arr = np.asarray(emitted_positions, dtype=np.int64)
    q_to_ref = np.asarray(task.q_to_ref, dtype=np.int64)

    for win_start, win_end in ctc.chunk_windows(len(normalised_signal), task.chunk_len, task.overlap):
        stats["chunks_seen"] += 1
        left_idx = int(np.searchsorted(emitted_positions_arr, win_start, side="left"))
        right_idx = int(np.searchsorted(emitted_positions_arr, win_end, side="left"))
        if left_idx >= right_idx:
            stats["chunks_zerolen_sequence"] += 1
            continue

        q_positions = signal_order_query_positions[left_idx:right_idx]
        ref_positions = q_to_ref[q_positions]
        aligned_mask = ref_positions >= 0
        aligned_count = int(aligned_mask.sum())
        chunk_base_count = int(q_positions.shape[0])
        chunk_aligned_fraction = aligned_count / max(chunk_base_count, 1)
        if chunk_aligned_fraction < task.min_chunk_aligned_fraction:
            stats["chunks_low_aligned_fraction"] += 1
            continue

        aligned_ref_positions = ref_positions[aligned_mask]
        ref_start = int(aligned_ref_positions.min())
        ref_end = int(aligned_ref_positions.max()) + 1
        ref_span = ref_end - ref_start
        if ref_span < task.min_reference_len:
            stats["chunks_short_reference"] += 1
            continue
        if task.max_reference_span_factor > 0 and ref_span > int(np.ceil(task.max_reference_span_factor * chunk_base_count)):
            stats["chunks_long_reference"] += 1
            continue

        ref_forward = aligner.seq(task.contig, ref_start, ref_end)
        if ref_forward is None:
            stats["chunks_no_reference"] += 1
            continue
        ref_forward = ref_forward.upper()
        if "N" in ref_forward:
            stats["chunks_n_in_reference"] += 1
            continue

        base_identity = chunk_base_identity(
            task.query_sequence,
            q_positions[aligned_mask],
            q_to_ref,
            ref_forward,
            ref_start,
            task.is_reverse,
        )
        if base_identity < task.min_chunk_base_identity:
            stats["chunks_low_base_identity"] += 1
            continue

        ref_seq = mappy.revcomp(ref_forward) if task.is_reverse else ref_forward
        target_seq = ref_seq[::-1] if task.sample_type == "rna" else ref_seq
        target = ctc.encode_reference(target_seq)
        if np.any(target == 0):
            stats["chunks_n_in_reference"] += 1
            continue
        if task.require_a and "A" not in target_seq.upper():
            stats["chunks_no_a"] += 1
            continue
        if task.max_label_len is not None and target.shape[0] > task.max_label_len:
            stats["chunks_too_long"] += 1
            continue

        pseudo_mapping = argparse.Namespace(
            ctg=task.contig,
            r_st=ref_start,
            r_en=ref_end,
            strand=task.ref_strand,
        )
        site_key, site_pos, kmer_context, motif_context = ctc.primary_a_site_metadata(
            target_seq,
            pseudo_mapping,
            task.sample_type,
            task.metadata_kmer,
        )

        samples.append(
            ctc.Sample(
                signal=normalised_signal[win_start:win_end],
                label=target.astype(np.uint8),
                label_len=int(target.shape[0]),
                record_id=task.record_id,
                pod5_read_id=task.pod5_read_id,
                run_id=task.run_id,
                contig=task.contig,
                ref_start=ref_start,
                ref_end=ref_end,
                ref_strand=task.ref_strand,
                chunk_start=int(win_start),
                chunk_end=int(win_end),
                primary_site_key=site_key,
                primary_site_pos=site_pos,
                kmer_context=kmer_context,
                motif_context=motif_context,
                mean_qscore=float(task.mean_qscore) if task.mean_qscore is not None else float("nan"),
                mapping_accuracy=float(base_identity),
                mapping_coverage=float(chunk_aligned_fraction),
            )
        )
        stats["chunks_written"] += 1

    return samples, stats


def process_task_batch(tasks: Sequence[BamAlignedTask], max_samples_per_file: int) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    batch_stats: Dict[str, int] = {}
    batch_samples: List[ctc.Sample] = []
    manifest_entries: List[Dict[str, object]] = []
    for task in tasks:
        samples, stats = process_task(task)
        ctc.merge_counters(batch_stats, stats)
        batch_samples.extend(samples)
        if len(batch_samples) >= max_samples_per_file:
            manifest_entry = ctc.write_worker_samples(batch_samples)
            if manifest_entry is not None:
                manifest_entries.append(manifest_entry)
            batch_samples = []
    if batch_samples:
        manifest_entry = ctc.write_worker_samples(batch_samples)
        if manifest_entry is not None:
            manifest_entries.append(manifest_entry)
    return manifest_entries, batch_stats


def resolve_thresholds(args) -> None:
    if args.filter_preset == "strict":
        defaults = {
            "min_read_identity": STRICT_MIN_READ_IDENTITY,
            "min_read_aligned_fraction": STRICT_MIN_READ_ALIGNED_FRACTION,
            "min_mapq": STRICT_MIN_MAPQ,
            "min_chunk_aligned_fraction": STRICT_MIN_CHUNK_ALIGNED_FRACTION,
            "min_chunk_base_identity": STRICT_MIN_CHUNK_BASE_IDENTITY,
        }
    else:
        defaults = {
            "min_read_identity": RELAXED_MIN_READ_IDENTITY,
            "min_read_aligned_fraction": RELAXED_MIN_READ_ALIGNED_FRACTION,
            "min_mapq": RELAXED_MIN_MAPQ,
            "min_chunk_aligned_fraction": RELAXED_MIN_CHUNK_ALIGNED_FRACTION,
            "min_chunk_base_identity": RELAXED_MIN_CHUNK_BASE_IDENTITY,
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a high-modification control dataset using Dorado move tables "
            "and BAM CIGAR alignment instead of chunk-local remapping."
        )
    )
    parser.add_argument("--bam-file", required=True, help="Mapped Dorado BAM produced with --reference --emit-moves")
    parser.add_argument("--pod5-dir", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=None, help="Optional run identifier override; defaults to BAM RG tag or 'unknown'.")
    parser.add_argument("--sample-type", choices=["dna", "rna"], default="rna")
    parser.add_argument("--chunk-len", type=int, default=10000)
    parser.add_argument("--overlap", type=int, default=500)
    parser.add_argument("--max-label-len", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--max-chunks", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip-value", type=float, default=5.0)
    parser.add_argument("--filter-preset", choices=["strict", "relaxed"], default="relaxed")
    parser.add_argument("--min-read-identity", type=float, default=None)
    parser.add_argument("--min-read-aligned-fraction", type=float, default=None)
    parser.add_argument("--min-mapq", type=int, default=None)
    parser.add_argument("--min-chunk-aligned-fraction", type=float, default=None)
    parser.add_argument("--min-chunk-base-identity", type=float, default=None)
    parser.add_argument("--min-reference-len", type=int, default=DEFAULT_MIN_REFERENCE_LEN)
    parser.add_argument("--max-reference-span-factor", type=float, default=DEFAULT_MAX_REFERENCE_SPAN_FACTOR)
    require_a_group = parser.add_mutually_exclusive_group()
    require_a_group.add_argument("--require-a", dest="require_a", action="store_true")
    require_a_group.add_argument("--allow-no-a", dest="require_a", action="store_false")
    require_a_group.set_defaults(require_a=True)
    parser.add_argument("--min-qscore", type=float, default=None)
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task-batch-size", type=int, default=ctc.DEFAULT_TASK_BATCH_SIZE)
    parser.add_argument("--max-pending-batches", type=int, default=ctc.DEFAULT_MAX_PENDING_BATCHES)
    parser.add_argument("--max-samples-per-worker-file", type=int, default=ctc.DEFAULT_MAX_SAMPLES_PER_WORKER_FILE)
    parser.add_argument("--progress-log-interval", type=int, default=ctc.DEFAULT_PROGRESS_LOG_INTERVAL)
    parser.add_argument("--mp-start-method", choices=["auto", "fork", "spawn", "forkserver"], default="auto")
    return parser.parse_args(argv)


def write_metadata_description(output_dir: Path) -> None:
    metadata_description = {
        "primary_site_key": "Center-most A in the BAM-CIGAR-derived chunk reference interval, encoded as contig:ref_pos:strand:A.",
        "kmer_context": "Centered k-mer around primary_site_key; edge means insufficient flank.",
        "motif_context": "DRACH/non_DRACH/edge/no_A coarse m6A motif class around primary_site_key.",
        "mean_qscore": "Dorado qs tag when available, else NaN.",
        "mapping_accuracy": "Chunk base identity against the BAM-CIGAR-derived reference interval.",
        "mapping_coverage": "Fraction of move-table-derived chunk query bases with aligned BAM reference positions.",
    }
    (output_dir / "metadata_fields.json").write_text(json.dumps(metadata_description, indent=2), encoding="utf-8")


def write_summary(output_dir: Path, args, counters: Dict[str, int], merge_stats: Dict[str, int]) -> None:
    summary = {
        "builder": "bam_aligned_highmod",
        "label_source": "bam_cigar",
        "bam_file": str(Path(args.bam_file).resolve()),
        "pod5_dir": str(Path(args.pod5_dir).resolve()),
        "reference_fasta": str(Path(args.reference_fasta).resolve()),
        "output_dir": str(output_dir.resolve()),
        "sample_type": args.sample_type,
        "run_id_override": args.run_id,
        "chunk_len": int(args.chunk_len),
        "overlap": int(args.overlap),
        "metadata_kmer": int(args.metadata_kmer),
        "max_chunks": None if int(args.max_chunks) <= 0 else int(args.max_chunks),
        "filter_preset": args.filter_preset,
        "min_read_identity": float(args.min_read_identity),
        "min_read_aligned_fraction": float(args.min_read_aligned_fraction),
        "min_mapq": int(args.min_mapq),
        "min_chunk_aligned_fraction": float(args.min_chunk_aligned_fraction),
        "min_chunk_base_identity": float(args.min_chunk_base_identity),
        "min_reference_len": int(args.min_reference_len),
        "max_reference_span_factor": float(args.max_reference_span_factor),
        "require_a": bool(args.require_a),
        "min_qscore": None if args.min_qscore is None else float(args.min_qscore),
        "rna002": bool(args.rna002),
        "model_config": None if args.model_config is None else str(Path(args.model_config).resolve()),
        "norm_strategy": args.norm_strategy,
        "pa_mean": float(args.pa_mean),
        "pa_std": float(args.pa_std),
        "quantile_a": float(args.quantile_a),
        "quantile_b": float(args.quantile_b),
        "shift_multiplier": float(args.shift_multiplier),
        "scale_multiplier": float(args.scale_multiplier),
        "clip_value": float(args.clip_value),
        "counters": {key: int(value) for key, value in sorted(counters.items())},
        "merge": merge_stats,
        "caution": (
            "This dataset trusts whole-read BAM CIGAR alignment and is intended "
            "for high-modification control rescue. Validate qscore, read identity, "
            "site/k-mer balance, and control separation before Stage 1 claims."
        ),
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    os.environ["TMPDIR"] = "/tmp"
    os.environ["TMP"] = "/tmp"
    os.environ["TEMP"] = "/tmp"
    tempfile.tempdir = "/tmp"
    args = parse_args()
    resolve_thresholds(args)
    ctc.resolve_signal_normalisation(args)
    if args.metadata_kmer <= 0 or args.metadata_kmer % 2 == 0:
        raise ValueError(f"--metadata-kmer must be a positive odd integer, got {args.metadata_kmer}")
    if args.overlap >= args.chunk_len:
        raise ValueError("--overlap must be smaller than --chunk-len")
    np.random.seed(args.seed)

    start_method = ctc.resolve_mp_start_method(args.mp_start_method)
    cwd = Path.cwd()
    existing_hidden_tmp_dirs = ctc.snapshot_hidden_temp_dirs(cwd)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Scanning BAM for required POD5 read ids...")
    required_read_ids, candidate_records = collect_required_pod5_ids(args.bam_file, args.max_records)
    print(f"      Candidate mapped BAM records: {candidate_records}")
    print(f"      Required POD5 parent/read ids: {len(required_read_ids)}")
    print("[2/5] Building filtered POD5 index...")
    lookup = ctc.build_pod5_lookup(Path(args.pod5_dir), required_read_ids=required_read_ids)
    print(f"      Indexed {len(lookup)} reads in POD5.")
    del required_read_ids
    gc.collect()

    print(f"      multiprocessing start method: {start_method}")
    if start_method == "fork":
        print("      preloading reference index in parent for fork-shared workers...")
        ctc.PARENT_ALIGNER = mappy.Aligner(args.reference_fasta, preset="map-ont")
        if ctc.PARENT_ALIGNER is None:
            raise RuntimeError(f"Failed to build/load minimap2 index for {args.reference_fasta}")
    else:
        ctc.PARENT_ALIGNER = None
        print("      worker-local reference indices enabled; this uses more memory than fork sharing.")

    effective_workers = max(1, min(int(args.workers), max((os.cpu_count() or 1) - 2, 1)))
    print("[3/5] Dispatching BAM records...")
    print(f"      requested workers: {args.workers}, effective workers: {effective_workers}")
    print(
        "      filters: "
        f"read_identity>={args.min_read_identity}, "
        f"read_aligned_fraction>={args.min_read_aligned_fraction}, "
        f"mapq>={args.min_mapq}, "
        f"chunk_aligned_fraction>={args.min_chunk_aligned_fraction}, "
        f"chunk_base_identity>={args.min_chunk_base_identity}"
    )

    chunk_manifest: List[Dict[str, object]] = []
    counters: Dict[str, int] = {}
    task_count = 0
    task_batch: List[BamAlignedTask] = []
    max_pending_batches = max(int(args.max_pending_batches), 1)
    max_chunks = int(args.max_chunks)
    progress_log_interval = int(args.progress_log_interval)
    next_progress_log = progress_log_interval if progress_log_interval > 0 else None
    mp_context = multiprocessing.get_context(start_method)

    ctc.PARENT_POD5_LOOKUP = lookup
    init_lookup = None if start_method == "fork" else lookup
    use_parent_aligner = start_method == "fork"

    executor = ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=mp_context,
        initializer=ctc.worker_init,
        initargs=(
            str(Path(args.reference_fasta).resolve()),
            init_lookup,
            "map-ont",
            str(temp_dir.resolve()),
            use_parent_aligner,
        ),
    )
    try:
        with pysam.AlignmentFile(args.bam_file, "rb", check_sq=False) as bam_file:
            futures = set()
            stop_dispatch = False
            with tqdm(bam_file, desc="Dispatching", unit="record", ascii=True, ncols=100) as dispatch_bar:
                for read in dispatch_bar:
                    if stop_dispatch:
                        break
                    if args.max_records > 0 and task_count >= args.max_records:
                        break
                    task = build_task(read, args)
                    if task is None:
                        continue
                    if task.pod5_read_id not in lookup:
                        continue
                    task_batch.append(task)
                    task_count += 1

                    if len(task_batch) >= max(int(args.task_batch_size), 1):
                        futures.add(executor.submit(process_task_batch, tuple(task_batch), int(args.max_samples_per_worker_file)))
                        task_batch = []

                    if len(futures) >= max_pending_batches:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        futures.difference_update(done)
                        for future in done:
                            manifest_entries, stats = future.result()
                            ctc.merge_counters(counters, stats)
                            chunk_manifest.extend(manifest_entries)
                            current_chunks = counters.get("chunks_written", 0)
                            dispatch_bar.set_postfix_str(f"accepted_chunks={current_chunks}")
                            if next_progress_log is not None and current_chunks >= next_progress_log:
                                print(
                                    f"      progress: accepted_chunks={current_chunks}, "
                                    f"tasks_processed={counters.get('records_seen', 0)}, "
                                    f"bam_records_dispatched={task_count}"
                                )
                                while next_progress_log is not None and current_chunks >= next_progress_log:
                                    next_progress_log += progress_log_interval
                            if max_chunks > 0 and current_chunks >= max_chunks:
                                stop_dispatch = True
                        if stop_dispatch:
                            print(f"      reached max_chunks={max_chunks}; stopping new task dispatch and draining in-flight batches...")
                            break

                if task_batch and not stop_dispatch:
                    futures.add(executor.submit(process_task_batch, tuple(task_batch), int(args.max_samples_per_worker_file)))
                    task_batch = []

            for future in tqdm(as_completed(futures), total=len(futures), desc="Finishing", ascii=True, ncols=100):
                manifest_entries, stats = future.result()
                ctc.merge_counters(counters, stats)
                chunk_manifest.extend(manifest_entries)
                current_chunks = counters.get("chunks_written", 0)
                if next_progress_log is not None and current_chunks >= next_progress_log:
                    print(
                        f"      progress: accepted_chunks={current_chunks}, "
                        f"tasks_processed={counters.get('records_seen', 0)}, "
                        f"bam_records_dispatched={task_count}"
                    )
                    while next_progress_log is not None and current_chunks >= next_progress_log:
                        next_progress_log += progress_log_interval
        executor.shutdown(wait=True, cancel_futures=False)
    except KeyboardInterrupt:
        print("\n[interrupt] stopping workers and cancelling pending batches...", flush=True)
        executor.shutdown(wait=False, cancel_futures=True)
        for process in list(getattr(executor, "_processes", {}).values()):
            try:
                process.terminate()
            except Exception:
                pass
        raise
    finally:
        ctc.PARENT_POD5_LOOKUP = {}
        ctc.PARENT_ALIGNER = None
        del lookup
        gc.collect()
        removed_tmp_dirs = ctc.cleanup_new_hidden_temp_dirs(cwd, existing_hidden_tmp_dirs)
        if removed_tmp_dirs:
            print(f"      cleaned {removed_tmp_dirs} hidden temporary directories under {cwd}")

    print("[4/5] Merging passing chunks into final dataset...")
    merge_summary = ctc.merge_chunks_to_final(
        output_dir,
        chunk_manifest,
        int(args.chunk_len),
        args.max_label_len,
        max_chunks,
    )
    write_metadata_description(output_dir)

    print("[5/5] Writing summary...")
    write_summary(output_dir, args, counters, merge_summary)

    print("[6/6] Cleaning up temp chunks...")
    try:
        temp_dir.rmdir()
    except OSError:
        pass
    removed_tmp_dirs = ctc.cleanup_new_hidden_temp_dirs(cwd, existing_hidden_tmp_dirs)
    if removed_tmp_dirs:
        print(f"      cleaned {removed_tmp_dirs} hidden temporary directories under {cwd}")

    print(f"Dataset ready at: {output_dir}")
    print(f"Final stats: {merge_summary['total_written']} chunks written.")
    print("Reject counters:")
    for key in sorted(counters):
        print(f" - {key}: {counters[key]}")


if __name__ == "__main__":
    main()
