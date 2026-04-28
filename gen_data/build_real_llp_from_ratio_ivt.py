#!/usr/bin/env python3
"""
Build a promoted LLP dataset from real known-ratio IVT runs.

This is for experiments where each input run already has a known expected m6A
fraction, for example IVT with 12.5/25/50/75% modified ATP in the ATP pool. It
does not synthesize read composition from 0/100 controls. Instead it:
1. optionally creates one Dorado/Bonito-style chunk dataset per ratio run,
2. marks A positions as LLP candidates without assigning read-level positives,
3. combines the ratio datasets with bag_targets.npy equal to the known ratio.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RatioRun:
    ratio_label: str
    ratio_fraction: float
    bam: Path
    pod5_dir: Path
    run_id: str | None


@dataclass(frozen=True)
class RatioDataset:
    ratio_label: str
    ratio_fraction: float
    directory: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def script_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def normalize_ratio(text: str) -> tuple[str, float]:
    ratio = float(text)
    fraction = ratio / 100.0 if ratio > 1.0 else ratio
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"ratio must be in [0, 1] or [0, 100], got {text!r}")
    return text, fraction


def parse_ratio_run(spec: str) -> RatioRun:
    parts = spec.split(":")
    if len(parts) not in {3, 4}:
        raise ValueError(
            f"Invalid --ratio-run {spec!r}; expected <ratio>:<bam>:<pod5_dir>[:run_id]. "
            "Use --ratio-dataset for prebuilt chunk datasets."
        )
    ratio_label, fraction = normalize_ratio(parts[0])
    return RatioRun(
        ratio_label=ratio_label,
        ratio_fraction=fraction,
        bam=Path(parts[1]),
        pod5_dir=Path(parts[2]),
        run_id=parts[3] if len(parts) == 4 and parts[3] else None,
    )


def parse_ratio_dataset(spec: str) -> RatioDataset:
    try:
        ratio_text, directory_text = spec.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid --ratio-dataset {spec!r}; expected <ratio>:<dataset_dir>.") from exc
    ratio_label, fraction = normalize_ratio(ratio_text)
    return RatioDataset(ratio_label, fraction, Path(directory_text))


def required_dataset_files_exist(directory: Path) -> bool:
    return all(
        (directory / name).exists()
        for name in ("chunks.npy", "references.npy", "reference_lengths.npy", "mod_targets.npy", "metadata.npz")
    )


def run_command(cmd: Sequence[object], *, dry_run: bool = False) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"[cmd] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=repo_root(), check=True)


def build_ratio_dataset(args, ratio_run: RatioRun) -> RatioDataset:
    dataset_dir = args.work_dir / f"ratio_{ratio_run.ratio_label.replace('.', 'p')}"
    if required_dataset_files_exist(dataset_dir) and not args.rebuild_sources:
        print(f"[source] using existing ratio dataset: {dataset_dir}", flush=True)
        return RatioDataset(ratio_run.ratio_label, ratio_run.ratio_fraction, dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    create_cmd = [
        sys.executable,
        script_path("create_dataset_dorado_ctc_like.py"),
        "--bam-file",
        ratio_run.bam,
        "--pod5-dir",
        ratio_run.pod5_dir,
        "--reference-fasta",
        args.reference_fasta,
        "--output-dir",
        dataset_dir,
        "--sample-type",
        args.sample_type,
        "--chunk-len",
        args.chunk_len,
        "--overlap",
        args.overlap,
        "--workers",
        args.workers,
        "--filter-preset",
        args.filter_preset,
        "--metadata-kmer",
        args.metadata_kmer,
        "--seed",
        args.seed,
        "--mm2-preset",
        args.mm2_preset,
    ]
    if ratio_run.run_id:
        create_cmd.extend(["--run-id", ratio_run.run_id])
    if args.max_label_len is not None:
        create_cmd.extend(["--max-label-len", args.max_label_len])
    if args.max_records > 0:
        create_cmd.extend(["--max-records", args.max_records])
    if args.max_chunks > 0:
        create_cmd.extend(["--max-chunks", args.max_chunks])
    if args.min_accuracy is not None:
        create_cmd.extend(["--min-accuracy", args.min_accuracy])
    if args.min_coverage is not None:
        create_cmd.extend(["--min-coverage", args.min_coverage])
    if args.min_qscore is not None:
        create_cmd.extend(["--min-qscore", args.min_qscore])
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
        dataset_dir,
        "--mode",
        "llp-candidate",
        "--non-a-policy",
        args.non_a_policy,
    ]
    run_command(target_cmd, dry_run=args.dry_run)
    return RatioDataset(ratio_run.ratio_label, ratio_run.ratio_fraction, dataset_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratio-run",
        action="append",
        default=[],
        help="Known-ratio run as <ratio>:<dorado_bam>:<pod5_dir>[:run_id]. Repeat for each ratio.",
    )
    parser.add_argument(
        "--ratio-dataset",
        action="append",
        default=[],
        help="Prebuilt known-ratio chunk dataset as <ratio>:<dataset_dir>. Repeat for each ratio.",
    )
    parser.add_argument("--reference-fasta", default=None)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebuild-sources", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)

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
    parser.add_argument("--non-a-policy", choices=["ignore", "canonical", "zero"], default="ignore")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--max-per-stratum", type=int, default=0)
    parser.add_argument("--qscore-bins", default="8,10,12,14,16")
    parser.add_argument("--coverage-bins", default="0.85,0.9,0.95,0.98")
    parser.add_argument("--heldout-mode", choices=["none", "leave-run", "leave-site"], default="none")
    parser.add_argument("--heldout-run", action="append", default=[])
    parser.add_argument("--heldout-runs-file", type=Path, default=None)
    parser.add_argument("--heldout-site", action="append", default=[])
    parser.add_argument("--heldout-sites-file", type=Path, default=None)
    parser.add_argument("--leave-site-fraction", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.ratio_run and not args.ratio_dataset:
        raise ValueError("Provide at least one --ratio-run or --ratio-dataset.")
    if args.ratio_run and args.reference_fasta is None:
        raise ValueError("--reference-fasta is required when using --ratio-run.")
    if args.metadata_kmer <= 0 or args.metadata_kmer % 2 == 0:
        raise ValueError(f"--metadata-kmer must be a positive odd integer, got {args.metadata_kmer}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    datasets = [parse_ratio_dataset(spec) for spec in args.ratio_dataset]
    datasets.extend(build_ratio_dataset(args, parse_ratio_run(spec)) for spec in args.ratio_run)
    if len(datasets) < 2:
        raise ValueError("Real LLP construction expects at least two known-ratio inputs.")

    combine_cmd = [
        sys.executable,
        script_path("build_llp_mixture_dataset.py"),
        "--output-dir",
        args.output_dir,
        "--seed",
        args.seed,
        "--max-per-stratum",
        args.max_per_stratum,
        "--qscore-bins",
        args.qscore_bins,
        "--coverage-bins",
        args.coverage_bins,
        "--heldout-mode",
        args.heldout_mode,
        "--leave-site-fraction",
        args.leave_site_fraction,
        "--validation-fraction",
        args.validation_fraction,
    ]
    for run_id in args.heldout_run:
        combine_cmd.extend(["--heldout-run", run_id])
    if args.heldout_runs_file is not None:
        combine_cmd.extend(["--heldout-runs-file", args.heldout_runs_file])
    for site in args.heldout_site:
        combine_cmd.extend(["--heldout-site", site])
    if args.heldout_sites_file is not None:
        combine_cmd.extend(["--heldout-sites-file", args.heldout_sites_file])
    for dataset in datasets:
        combine_cmd.extend(["--ratio-dataset", f"{dataset.ratio_label}:{dataset.directory}"])
    run_command(combine_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
