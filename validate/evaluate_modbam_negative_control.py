#!/usr/bin/env python3
"""
Evaluate false-positive modified-base calls on an unmodified negative-control modBAM.

This script is intended for 0% IVT / unmodified controls. It aggregates MM/ML
modified-base probabilities into reference site-level scores, then reports the
fraction of covered candidate sites whose score exceeds each threshold.
"""

from __future__ import annotations

import csv
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pysam

try:
    from evaluate_modbam_gold_sites import (
        SiteKey,
        SiteStats,
        aggregate_modbam_sites,
        default_thresholds,
        motif_regex,
        optional_pyplot,
        progress_iter,
        score_for_site,
        write_tsv,
    )
except ImportError:
    from validate.evaluate_modbam_gold_sites import (
        SiteKey,
        SiteStats,
        aggregate_modbam_sites,
        default_thresholds,
        motif_regex,
        optional_pyplot,
        progress_iter,
        score_for_site,
        write_tsv,
    )


def build_negative_rows(
    stats: Dict[SiteKey, SiteStats],
    *,
    min_coverage: int,
    score_column: str,
    show_progress: bool = True,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    items = sorted(stats.items())
    for key, site_stats in progress_iter(
        items,
        desc="Building negative-control site rows",
        total=len(items),
        enabled=show_progress,
    ):
        if site_stats.coverage < int(min_coverage):
            continue
        chrom, start, strand = key
        score = score_for_site(site_stats, score_column)
        if score is None:
            score = 0.0
        rows.append({
            "chrom": chrom,
            "start": int(start),
            "end": int(start) + 1,
            "strand": strand,
            "coverage": int(site_stats.coverage),
            "num_mod_calls": int(site_stats.num_mod_calls),
            "mod_fraction": float(site_stats.mod_fraction),
            "mean_prob_zero_filled": float(site_stats.mean_prob_zero_filled),
            "mean_called_prob": site_stats.mean_called_prob,
            "median_called_prob": site_stats.median_called_prob,
            "score": float(score),
        })
    return rows


def threshold_false_positive_rows(
    rows: Sequence[Dict[str, object]],
    thresholds: Sequence[float],
) -> List[Dict[str, object]]:
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    total = int(scores.size)
    output = []
    for threshold in thresholds:
        fp = int(np.count_nonzero(scores >= float(threshold))) if total else 0
        fraction = float(fp / total) if total else 0.0
        output.append({
            "threshold": float(threshold),
            "false_positive_sites": fp,
            "total_sites": total,
            "false_positive_fraction": fraction,
            "specificity": float(1.0 - fraction),
        })
    return output


def quantile_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "q99": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def save_negative_control_plots(rows: Sequence[Dict[str, object]], output_dir: Path) -> List[str]:
    plt = optional_pyplot()
    if plt is None or not rows:
        return []

    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    coverage = np.asarray([int(row["coverage"]) for row in rows], dtype=np.int64)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(scores, bins=80, color="#636363", alpha=0.85)
    ax.set_xlabel("Site score")
    ax.set_ylabel("Site count")
    ax.set_title("Negative-control site score distribution")
    ax.grid(alpha=0.2)
    path = output_dir / "score_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sorted_scores = np.sort(scores)
    survival = 1.0 - (np.arange(sorted_scores.size, dtype=np.float64) + 1.0) / float(sorted_scores.size)
    ax.plot(sorted_scores, survival, color="#2c7fb8")
    ax.set_xlabel("Site score threshold")
    ax.set_ylabel("Fraction of sites >= threshold")
    ax.set_title("Negative-control false-positive curve")
    ax.grid(alpha=0.2)
    path = output_dir / "false_positive_curve.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(coverage, scores, s=8, alpha=0.35, color="#525252")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Site score")
    ax.set_title("Coverage vs site score")
    ax.grid(alpha=0.2)
    path = output_dir / "coverage_vs_score.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    return written


def build_text_summary(summary: Dict[str, object]) -> str:
    score = summary["score_distribution"]
    selected = summary["selected_thresholds"]
    lines = [
        "[inputs]",
        f"bam: {summary['inputs']['bam']}",
        "",
        "[counts]",
        f"covered_sites_before_filter: {summary['counts']['covered_sites_before_filter']}",
        f"evaluated_sites: {summary['counts']['evaluated_sites']}",
        "",
        "[score_distribution]",
        f"score_column: {summary['settings']['score_column']}",
        f"mean: {score['mean']}",
        f"median: {score['median']}",
        f"q90: {score['q90']}",
        f"q95: {score['q95']}",
        f"q99: {score['q99']}",
        f"max: {score['max']}",
        "",
        "[false_positive_fraction]",
    ]
    for threshold in ["0.5", "0.8", "0.9", "0.95", "0.99"]:
        lines.append(f"score>={threshold}: {selected.get(threshold)}")
    return "\n".join(lines) + "\n"


def main(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.motif and args.motif_kmer_size % 2 == 0:
        raise ValueError("--motif-kmer-size must be odd")

    motif = motif_regex(args.motif)
    fasta = pysam.FastaFile(str(args.reference)) if args.reference else None
    try:
        stats, bam_counters = aggregate_modbam_sites(
            args.bam,
            canonical_base=args.canonical_base,
            mod_code=args.mod_code,
            keep_u=args.keep_u,
            include_non_primary=args.include_non_primary,
            ignore_strand=args.ignore_strand,
            fasta=fasta,
            motif=motif,
            motif_kmer_size=args.motif_kmer_size,
            show_progress=not args.no_progress,
        )
    finally:
        if fasta is not None:
            fasta.close()

    rows = build_negative_rows(
        stats,
        min_coverage=args.min_coverage,
        score_column=args.score_column,
        show_progress=not args.no_progress,
    )
    thresholds = default_thresholds()
    threshold_rows = threshold_false_positive_rows(rows, thresholds)

    fieldnames = [
        "chrom",
        "start",
        "end",
        "strand",
        "coverage",
        "num_mod_calls",
        "mod_fraction",
        "mean_prob_zero_filled",
        "mean_called_prob",
        "median_called_prob",
        "score",
    ]
    write_tsv(output_dir / "site_level_scores.tsv", rows, fieldnames)
    write_tsv(
        output_dir / "threshold_false_positive.tsv",
        threshold_rows,
        [
            "threshold",
            "false_positive_sites",
            "total_sites",
            "false_positive_fraction",
            "specificity",
        ],
    )

    scores = [float(row["score"]) for row in rows]
    coverage = [float(row["coverage"]) for row in rows]
    selected = {
        str(threshold): next(
            row["false_positive_fraction"]
            for row in threshold_rows
            if abs(float(row["threshold"]) - threshold) < 1e-8
        )
        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]
    }
    plots = save_negative_control_plots(rows, output_dir) if not args.no_plots else []

    summary = {
        "inputs": {
            "bam": str(args.bam.resolve()),
            "reference": None if args.reference is None else str(args.reference.resolve()),
        },
        "settings": {
            "canonical_base": args.canonical_base,
            "mod_code": args.mod_code,
            "min_coverage": args.min_coverage,
            "score_column": args.score_column,
            "ignore_strand": args.ignore_strand,
            "include_non_primary": args.include_non_primary,
            "motif": args.motif,
            "motif_kmer_size": args.motif_kmer_size,
        },
        "bam_counters": bam_counters,
        "counts": {
            "covered_sites_before_filter": int(len(stats)),
            "evaluated_sites": int(len(rows)),
        },
        "coverage_distribution": quantile_summary(coverage),
        "score_distribution": quantile_summary(scores),
        "selected_thresholds": selected,
        "threshold_false_positive_tsv": str((output_dir / "threshold_false_positive.tsv").resolve()),
        "plots": plots,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    (output_dir / "summary.txt").write_text(build_text_summary(summary), encoding="utf-8")

    print(build_text_summary(summary), end="")
    print(f"artifacts written to: {output_dir}")
    if not args.no_plots and not plots:
        print("plots not written: matplotlib is not installed in this environment")


def argparser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bam", type=Path, required=True, help="Aligned modBAM/SAM from an unmodified control.")
    parser.add_argument("--reference", type=Path, default=None, help="Reference FASTA, required for motif filtering.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-base", default="A")
    parser.add_argument("--mod-code", default="a", help="SAM MM modification code, e.g. 'a' for m6A.")
    parser.add_argument("--min-coverage", type=int, default=5)
    parser.add_argument(
        "--score-column",
        choices=["mod_fraction", "mean_prob_zero_filled", "mean_called_prob", "median_called_prob"],
        default="mean_prob_zero_filled",
    )
    parser.add_argument("--ignore-strand", action="store_true", default=False)
    parser.add_argument("--include-non-primary", action="store_true", default=False)
    parser.add_argument("--keep-u", action="store_true", default=False)
    parser.add_argument("--motif", default=None, help="Optional motif filter: DRACH, RRACH, or a regex.")
    parser.add_argument("--motif-kmer-size", type=int, default=5)
    parser.add_argument("--no-progress", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
