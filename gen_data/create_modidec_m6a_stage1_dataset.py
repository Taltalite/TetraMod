#!/usr/bin/env python3
"""
Build TetraMod Stage 1 datasets from MoDiDeC m6A construct POD5/BAM.

MoDiDeC m6A data are not a matched fully-unmodified vs fully-modified control.
The m6A sites are explicit oligo positions in Supplementary Table 1.  This
builder therefore treats each read as an internal-label source:

- chunks containing a matched m6A oligo center are positive, with only that
  explicit center A labeled as m6A;
- chunks that do not overlap any matched m6A oligo interval are negative, with
  A sites in the chunk labeled canonical_A;
- all other chunks are ignored.

Input BAM must be Dorado RNA002 basecalls with --emit-moves.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import shutil
import tempfile
import zlib
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Sequence

import numpy as np
import pysam
from tqdm import tqdm

try:
    import create_dataset_dorado_ctc_like as ctc
    import create_mafia_synthetic_stage1_dataset as mafia
except ImportError:  # pragma: no cover - package-style execution fallback.
    from gen_data import create_dataset_dorado_ctc_like as ctc
    from gen_data import create_mafia_synthetic_stage1_dataset as mafia


DEFAULT_NEGATIVE_CHUNKS_PER_POSITIVE = 2
DEFAULT_NEGATIVE_EXCLUSION_BASES = 0


def interval_overlaps_any(start: int, end: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(start < interval_end and end > interval_start for interval_start, interval_end in intervals)


def choose_negative_windows(
    windows: Sequence[tuple[int, int]],
    emitted_positions: np.ndarray,
    signal_order_sequence: str,
    excluded_base_intervals: Sequence[tuple[int, int]],
    *,
    max_negative: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for win_start, win_end in windows:
        left_idx = int(np.searchsorted(emitted_positions, win_start, side="left"))
        right_idx = int(np.searchsorted(emitted_positions, win_end, side="left"))
        if left_idx >= right_idx:
            continue
        if interval_overlaps_any(left_idx, right_idx, excluded_base_intervals):
            continue
        target_seq = signal_order_sequence[left_idx:right_idx]
        if "A" not in target_seq.upper():
            continue
        candidates.append((win_start, win_end))

    if max_negative <= 0 or len(candidates) <= max_negative:
        return candidates
    selected = rng.choice(len(candidates), size=max_negative, replace=False)
    return [candidates[int(idx)] for idx in sorted(selected)]


def sample_from_window(
    task: ctc.TaskData,
    normalised_signal: np.ndarray,
    signal_order_sequence: str,
    emitted_positions: np.ndarray,
    win_start: int,
    win_end: int,
    *,
    run_id: str,
    units: Sequence[mafia.MatchedUnit],
    label_all_a_canonical: bool,
    metadata_kmer: int,
    max_label_len: int | None,
) -> mafia.MafiaSample | None:
    left_idx = int(np.searchsorted(emitted_positions, win_start, side="left"))
    right_idx = int(np.searchsorted(emitted_positions, win_end, side="left"))
    if left_idx >= right_idx:
        return None

    target_seq = signal_order_sequence[left_idx:right_idx]
    target = ctc.encode_reference(target_seq)
    if np.any(target == 0):
        return None
    if max_label_len is not None and target.shape[0] > max_label_len:
        return None

    mod_target = np.full(target.shape, mafia.IGNORE_INDEX, dtype=np.int16)
    kept_units: list[mafia.MatchedUnit] = []
    local_indices: list[int] = []

    if label_all_a_canonical:
        a_indices = np.flatnonzero(target == ctc.BASE_TO_INT["A"]).astype(np.int64)
        if a_indices.size == 0:
            return None
        mod_target[a_indices] = mafia.CANONICAL_A_LABEL
        primary_idx = int(a_indices[np.argmin(np.abs(a_indices - ((target.shape[0] - 1) / 2.0)))])
        site_key = f"modidec_unmodified:{primary_idx}:1:A"
        site_pos = primary_idx
        kmer_context = ctc.centered_kmer(target_seq, primary_idx, metadata_kmer)
        motif_context = kmer_context
        oligo_ids = "unmatched_unmodified_window"
        oligo_motifs = motif_context
        oligo_orientations = "."
        positive_count = 0
        negative_count = int(a_indices.size)
    else:
        for unit in units:
            local_idx = int(unit.center_index - left_idx)
            if local_idx < 0 or local_idx >= target.shape[0] or target[local_idx] != ctc.BASE_TO_INT["A"]:
                continue
            mod_target[local_idx] = mafia.M6A_LABEL
            kept_units.append(unit)
            local_indices.append(local_idx)
        if not kept_units:
            return None
        positive_count = len(kept_units)
        negative_count = 0
        site_key, site_pos, kmer_context, motif_context = mafia.primary_center_metadata(
            target_seq,
            kept_units,
            local_indices,
            metadata_kmer,
        )
        oligo_ids = ",".join(unit.oligo.oligo_id for unit in kept_units)
        oligo_motifs = ",".join(unit.oligo.motif for unit in kept_units)
        oligo_orientations = ",".join(unit.orientation for unit in kept_units)

    return mafia.MafiaSample(
        signal=normalised_signal[win_start:win_end],
        label=target.astype(np.uint8),
        mod_target=mod_target,
        label_len=int(target.shape[0]),
        record_id=task.record_id,
        pod5_read_id=task.pod5_read_id,
        run_id=run_id,
        contig="modidec_construct",
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
        mapping_accuracy=float(np.mean([unit.identity for unit in kept_units])) if kept_units else float("nan"),
        mapping_coverage=1.0,
        oligo_ids=oligo_ids,
        oligo_motifs=oligo_motifs,
        oligo_orientations=oligo_orientations,
        modification_status="modified" if positive_count else "unmodified_internal",
        ligation_strategy="modidec_internal",
        labeled_center_count=int(positive_count + negative_count),
        positive_center_count=int(positive_count),
        negative_center_count=int(negative_count),
    )


def build_samples_for_task(
    task: ctc.TaskData,
    oligos: Sequence[mafia.OligoSpec],
    args,
    settings: mafia.ProcessSettings,
) -> tuple[list[mafia.MafiaSample], dict[str, int]]:
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
        "positive_chunks_written": 0,
        "negative_chunks_written": 0,
        "chunks_written": 0,
        "chunks_no_label": 0,
        "chunks_too_long": 0,
        "labeled_centers": 0,
        "positive_centers": 0,
        "negative_centers": 0,
    }
    samples: list[mafia.MafiaSample] = []

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

    run = mafia.RunSpec(
        run_id=str(args.run_id),
        accession="modidec",
        local_name="direct_pod5",
        modification_status="modified",
        ligation_strategy="modidec_internal",
        oligo_ids=tuple(item.oligo_id for item in oligos),
    )
    units = mafia.find_oligo_units(
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
    unit_by_window = mafia.assign_units_to_windows(units, emitted_positions_arr, windows)
    excluded_base_intervals = [
        (
            max(0, int(unit.start) - int(args.negative_exclusion_bases)),
            min(len(signal_order_sequence), int(unit.end) + int(args.negative_exclusion_bases)),
        )
        for unit in units
    ]

    for win_start, win_end in windows:
        stats["chunks_seen"] += 1
        window_units = unit_by_window.get((win_start, win_end), [])
        if not window_units:
            continue
        sample = sample_from_window(
            task,
            normalised_signal,
            signal_order_sequence,
            emitted_positions_arr,
            win_start,
            win_end,
            run_id=str(args.run_id),
            units=window_units,
            label_all_a_canonical=False,
            metadata_kmer=settings.metadata_kmer,
            max_label_len=settings.max_label_len,
        )
        if sample is None:
            stats["chunks_no_label"] += 1
            continue
        samples.append(sample)
        stats["positive_chunks_written"] += 1
        stats["chunks_written"] += 1
        stats["labeled_centers"] += sample.labeled_center_count
        stats["positive_centers"] += sample.positive_center_count

    max_negative = int(args.negative_chunks_per_positive) * max(stats["positive_chunks_written"], 1)
    read_seed = (zlib.adler32(task.record_id.encode("utf-8")) + int(args.seed)) & 0xFFFFFFFF
    rng = np.random.default_rng(read_seed)
    negative_windows = choose_negative_windows(
        windows,
        emitted_positions_arr,
        signal_order_sequence,
        excluded_base_intervals,
        max_negative=max_negative,
        rng=rng,
    )
    for win_start, win_end in negative_windows:
        sample = sample_from_window(
            task,
            normalised_signal,
            signal_order_sequence,
            emitted_positions_arr,
            win_start,
            win_end,
            run_id=str(args.run_id),
            units=(),
            label_all_a_canonical=True,
            metadata_kmer=settings.metadata_kmer,
            max_label_len=settings.max_label_len,
        )
        if sample is None:
            stats["chunks_no_label"] += 1
            continue
        samples.append(sample)
        stats["negative_chunks_written"] += 1
        stats["chunks_written"] += 1
        stats["labeled_centers"] += sample.labeled_center_count
        stats["negative_centers"] += sample.negative_center_count

    return samples, stats


def process_task_batch(
    tasks: Sequence[ctc.TaskData],
    oligos: Sequence[mafia.OligoSpec],
    args,
    settings: mafia.ProcessSettings,
    max_samples_per_file: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    stats: dict[str, int] = {}
    batch_samples: list[mafia.MafiaSample] = []
    manifest_entries: list[dict[str, object]] = []
    for task in tasks:
        samples, sample_stats = build_samples_for_task(task, oligos, args, settings)
        mafia.merge_counters(stats, sample_stats)
        batch_samples.extend(samples)
        if len(batch_samples) >= max_samples_per_file:
            entry = mafia.write_worker_samples(batch_samples)
            if entry is not None:
                manifest_entries.append(entry)
            batch_samples = []
    if batch_samples:
        entry = mafia.write_worker_samples(batch_samples)
        if entry is not None:
            manifest_entries.append(entry)
    return manifest_entries, stats


def write_summary(output_dir: Path, args, oligos: Sequence[mafia.OligoSpec], counters, merge_summary):
    summary = {
        "builder": "modidec_m6a_internal_stage1",
        "label_strategy": {
            "positive": "matched MoDiDeC m6A oligo center A only",
            "negative": "A sites in chunks outside matched m6A oligo intervals",
        },
        "bam_file": str(Path(args.bam_file).resolve()),
        "pod5_dir": str(Path(args.pod5_dir).resolve()),
        "run_id": str(args.run_id),
        "oligo_ids": [item.oligo_id for item in oligos],
        "output_dir": str(output_dir.resolve()),
        "sample_type": args.sample_type,
        "chunk_len": int(args.chunk_len),
        "overlap": int(args.overlap),
        "negative_chunks_per_positive": int(args.negative_chunks_per_positive),
        "negative_exclusion_bases": int(args.negative_exclusion_bases),
        "min_oligo_identity": float(args.min_oligo_identity),
        "max_oligo_mismatches": int(args.max_oligo_mismatches),
        "allow_reverse_match": bool(args.allow_reverse_match),
        "counters": {key: int(value) for key, value in sorted(counters.items())},
        "merge": merge_summary,
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bam-file", required=True, help="Dorado BAM produced with --emit-moves.")
    parser.add_argument("--pod5-dir", required=True, help="POD5 directory or directory containing symlinked POD5.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oligo-manifest", type=Path, required=True)
    parser.add_argument("--oligo-ids", default="", help="Comma-separated oligo ids to use. Empty means all manifest rows.")
    parser.add_argument("--run-id", default="modidec_m6a")
    parser.add_argument("--sample-type", choices=["rna", "dna"], default="rna")
    parser.add_argument("--chunk-len", type=int, default=5000)
    parser.add_argument("--overlap", type=int, default=500)
    parser.add_argument("--max-label-len", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--max-chunks", type=int, default=mafia.DEFAULT_MAX_CHUNKS)
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
    parser.add_argument("--min-oligo-identity", type=float, default=mafia.DEFAULT_MIN_OLIGO_IDENTITY)
    parser.add_argument("--max-oligo-mismatches", type=int, default=mafia.DEFAULT_MAX_OLIGO_MISMATCHES)
    parser.add_argument("--allow-reverse-match", action="store_true", default=True)
    parser.add_argument("--no-reverse-match", dest="allow_reverse_match", action="store_false")
    parser.add_argument("--negative-chunks-per-positive", type=int, default=DEFAULT_NEGATIVE_CHUNKS_PER_POSITIVE)
    parser.add_argument("--negative-exclusion-bases", type=int, default=DEFAULT_NEGATIVE_EXCLUSION_BASES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task-batch-size", type=int, default=mafia.DEFAULT_TASK_BATCH_SIZE)
    parser.add_argument("--max-pending-batches", type=int, default=mafia.DEFAULT_MAX_PENDING_BATCHES)
    parser.add_argument("--max-samples-per-worker-file", type=int, default=mafia.DEFAULT_MAX_SAMPLES_PER_WORKER_FILE)
    parser.add_argument("--mp-start-method", choices=["auto", "fork", "spawn", "forkserver"], default="auto")
    return parser.parse_args(argv)


def selected_oligos(args) -> list[mafia.OligoSpec]:
    manifest = mafia.load_oligo_manifest(args.oligo_manifest)
    selected_ids = mafia.split_ids(args.oligo_ids)
    if not selected_ids:
        selected_ids = tuple(manifest)
    missing = [oligo_id for oligo_id in selected_ids if oligo_id not in manifest]
    if missing:
        raise KeyError(f"Requested oligo ids are missing from {args.oligo_manifest}: {missing}")
    return [manifest[oligo_id] for oligo_id in selected_ids]


def main(argv=None):
    args = parse_args(argv)
    ctc.resolve_signal_normalisation(args)
    if args.metadata_kmer <= 0 or args.metadata_kmer % 2 == 0:
        raise ValueError(f"--metadata-kmer must be a positive odd integer, got {args.metadata_kmer}")
    if args.negative_chunks_per_positive < 0:
        raise ValueError("--negative-chunks-per-positive must be >= 0")
    if args.negative_exclusion_bases < 0:
        raise ValueError("--negative-exclusion-bases must be >= 0")
    np.random.seed(args.seed)

    oligos = selected_oligos(args)
    settings = mafia.settings_from_args(args)

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
        mafia.worker_init(lookup, str(temp_dir.resolve()))
        with pysam.AlignmentFile(args.bam_file, "rb", check_sq=False) as bam_file:
            for read in tqdm(bam_file, desc="Processing", unit="record", ascii=True, ncols=100):
                if args.max_records > 0 and task_count >= args.max_records:
                    break
                task = ctc.build_task(read, args)
                if task is None or task.pod5_read_id not in lookup:
                    continue
                task_count += 1
                entries, stats = process_task_batch((task,), tuple(oligos), args, settings, int(args.max_samples_per_worker_file))
                chunk_manifest.extend(entries)
                mafia.merge_counters(counters, stats)
                if max_chunks > 0 and counters.get("chunks_written", 0) >= max_chunks:
                    break
    else:
        mp_context = multiprocessing.get_context(start_method)
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
            initializer=mafia.worker_init,
            initargs=(lookup, str(temp_dir.resolve())),
        )
        try:
            with pysam.AlignmentFile(args.bam_file, "rb", check_sq=False) as bam_file:
                futures = set()
                stop_dispatch = False
                for read in tqdm(bam_file, desc="Dispatching", unit="record", ascii=True, ncols=100):
                    if stop_dispatch or (args.max_records > 0 and task_count >= args.max_records):
                        break
                    task = ctc.build_task(read, args)
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
                                args,
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
                            mafia.merge_counters(counters, stats)
                            if max_chunks > 0 and counters.get("chunks_written", 0) >= max_chunks:
                                stop_dispatch = True
                if task_batch and not stop_dispatch:
                    futures.add(
                        executor.submit(
                            process_task_batch,
                            tuple(task_batch),
                            tuple(oligos),
                            args,
                            settings,
                            int(args.max_samples_per_worker_file),
                        )
                    )
                for future in tqdm(as_completed(futures), total=len(futures), desc="Finishing", ascii=True, ncols=100):
                    entries, stats = future.result()
                    chunk_manifest.extend(entries)
                    mafia.merge_counters(counters, stats)
            executor.shutdown(wait=True, cancel_futures=False)
        finally:
            del lookup
            gc.collect()

    print("[4/5] Merging passing chunks into final dataset...")
    merge_summary = mafia.merge_chunks_to_final(output_dir, chunk_manifest, int(args.chunk_len), args.max_label_len, max_chunks)
    print("[5/5] Writing summary...")
    write_summary(output_dir, args, oligos, counters, merge_summary)
    try:
        temp_dir.rmdir()
    except OSError:
        pass
    print(f"Dataset ready at: {output_dir}")
    print(f"Final stats: {merge_summary['total_written']} chunks written.")


if __name__ == "__main__":
    main()
