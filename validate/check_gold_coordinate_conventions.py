#!/usr/bin/env python3
"""
Sweep gold-site coordinate and strand conventions against an aligned modBAM.

This diagnostic is meant to answer a narrow question: whether poor gold-site
agreement can be explained by a 0/1-based offset or strand convention mismatch.
It reuses the modBAM aggregation from evaluate_modbam_gold_sites.py, then scores
the same BAM against shifted/flipped gold-site keys.
"""

from __future__ import annotations

import csv
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pysam

try:
    from evaluate_modbam_gold_sites import (
        GoldInfo,
        SiteKey,
        SiteStats,
        aggregate_modbam_sites,
        build_site_rows,
        compute_metrics,
        load_gold_sites,
        motif_regex,
        reverse_complement,
        site_context,
    )
except ImportError:
    from validate.evaluate_modbam_gold_sites import (
        GoldInfo,
        SiteKey,
        SiteStats,
        aggregate_modbam_sites,
        build_site_rows,
        compute_metrics,
        load_gold_sites,
        motif_regex,
        reverse_complement,
        site_context,
    )


COMPLEMENT = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")


def flip_strand(strand: str) -> str:
    if strand == "+":
        return "-"
    if strand == "-":
        return "+"
    return "."


def shifted_info(info: GoldInfo, start: int, strand: str) -> GoldInfo:
    updated = dict(info)
    updated["start"] = int(start)
    updated["end"] = int(start) + 1
    updated["strand"] = strand
    return updated


def transform_gold(
    gold: Dict[SiteKey, GoldInfo],
    *,
    shift: int,
    convention: str,
) -> Dict[SiteKey, GoldInfo]:
    transformed: Dict[SiteKey, GoldInfo] = {}
    for (chrom, start, strand), info in gold.items():
        shifted_start = int(start) + int(shift)
        if shifted_start < 0:
            continue

        if convention == "as_is":
            out_strands = [strand if strand in {"+", "-"} else "."]
        elif convention == "flip_gold_strand":
            out_strands = [flip_strand(strand)]
        elif convention == "ignore_strand":
            out_strands = ["."]
        elif convention == "either_strand":
            out_strands = ["+", "-"] if strand in {"+", "-"} else ["+", "-"]
        else:
            raise ValueError(f"Unsupported convention: {convention}")

        for out_strand in out_strands:
            key = (chrom, shifted_start, out_strand)
            transformed[key] = shifted_info(info, shifted_start, out_strand)
    return transformed


def collapse_stats_by_strand(stats: Dict[SiteKey, SiteStats]) -> Dict[SiteKey, SiteStats]:
    collapsed: Dict[SiteKey, SiteStats] = {}
    for (chrom, start, _strand), site_stats in stats.items():
        key = (chrom, int(start), ".")
        target = collapsed.setdefault(key, SiteStats())
        target.coverage += site_stats.coverage
        target.mod_probs.extend(site_stats.mod_probs)
    return collapsed


def expected_reference_bases(canonical_base: str, strand: str) -> set[str]:
    base = str(canonical_base).upper().replace("U", "T")
    comp = base.translate(COMPLEMENT).upper().replace("U", "T")
    if strand == "+":
        return {base}
    if strand == "-":
        return {comp}
    return {base, comp}


def reference_base(fasta, chrom: str, start: int) -> Optional[str]:
    if fasta is None:
        return None
    try:
        base = fasta.fetch(chrom, int(start), int(start) + 1).upper()
    except Exception:
        return None
    if len(base) != 1 or base == "N":
        return None
    return base


def reference_base_summary(
    gold: Dict[SiteKey, GoldInfo],
    *,
    fasta,
    canonical_base: str,
) -> Dict[str, object]:
    checked = 0
    compatible = 0
    mismatches: Dict[str, int] = {}
    for chrom, start, strand in gold:
        base = reference_base(fasta, chrom, start)
        if base is None:
            continue
        checked += 1
        if base in expected_reference_bases(canonical_base, strand):
            compatible += 1
        else:
            key = f"{strand}:{base}"
            mismatches[key] = mismatches.get(key, 0) + 1

    return {
        "reference_base_checked": checked,
        "reference_base_compatible": compatible,
        "reference_base_compatible_fraction": float(compatible / checked) if checked else None,
        "reference_base_mismatches": mismatches,
    }


def motif_matches_site(fasta, chrom: str, start: int, strand: str, motif, kmer_size: int) -> Optional[bool]:
    if fasta is None or motif is None:
        return None
    if strand == ".":
        plus = site_context(fasta, chrom, start, "+", kmer_size)
        minus = site_context(fasta, chrom, start, "-", kmer_size)
        return bool((plus and motif.search(plus)) or (minus and motif.search(minus)))
    context = site_context(fasta, chrom, start, strand, kmer_size)
    return bool(context and motif.search(context))


def motif_summary(
    gold: Dict[SiteKey, GoldInfo],
    *,
    fasta,
    motif,
    kmer_size: int,
) -> Dict[str, object]:
    checked = 0
    matched = 0
    for chrom, start, strand in gold:
        result = motif_matches_site(fasta, chrom, start, strand, motif, kmer_size)
        if result is None:
            continue
        checked += 1
        matched += int(result)
    return {
        "motif_checked": checked,
        "motif_matched": matched,
        "motif_match_fraction": float(matched / checked) if checked else None,
    }


def metric_value(metrics: Dict[str, object], path: Tuple[str, ...], default=None):
    current = metrics
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def evaluate_convention(
    *,
    stats: Dict[SiteKey, SiteStats],
    collapsed_stats: Dict[SiteKey, SiteStats],
    gold: Dict[SiteKey, GoldInfo],
    convention: str,
    shift: int,
    min_coverage: int,
    score_column: str,
    threshold: float,
    fasta,
    canonical_base: str,
    motif,
    motif_kmer_size: int,
) -> Dict[str, object]:
    shifted_gold = transform_gold(gold, shift=shift, convention=convention)
    stats_for_convention = collapsed_stats if convention == "ignore_strand" else stats
    rows = build_site_rows(
        stats_for_convention,
        shifted_gold,
        min_coverage=min_coverage,
        score_column=score_column,
        threshold=threshold,
    )
    metrics = compute_metrics(rows, threshold=threshold)
    base = reference_base_summary(shifted_gold, fasta=fasta, canonical_base=canonical_base)
    motif_stats = motif_summary(shifted_gold, fasta=fasta, motif=motif, kmer_size=motif_kmer_size)
    covered_before_filter = len(set(stats_for_convention) & set(shifted_gold))

    threshold_metrics = metrics.get("threshold_metrics", {})
    top_n = metrics.get("top_n_gold_fraction", {})
    return {
        "convention": convention,
        "gold_shift": int(shift),
        "gold_sites": int(len(shifted_gold)),
        "covered_gold_before_filter": int(covered_before_filter),
        "evaluated_sites": int(metrics.get("num_sites", 0)),
        "evaluated_gold_sites": int(metrics.get("num_positive", 0)),
        "evaluated_negative_sites": int(metrics.get("num_negative", 0)),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "precision": threshold_metrics.get("precision"),
        "recall": threshold_metrics.get("recall"),
        "f1": threshold_metrics.get("f1"),
        "top100_gold_fraction": top_n.get("100"),
        "top500_gold_fraction": top_n.get("500"),
        "top1000_gold_fraction": top_n.get("1000"),
        **base,
        **motif_stats,
    }


def sort_key(record: Dict[str, object]) -> Tuple[float, float, float, int]:
    roc = record.get("roc_auc")
    pr = record.get("pr_auc")
    top = record.get("top1000_gold_fraction")
    covered = int(record.get("evaluated_gold_sites", 0))
    return (
        float(-1 if roc is None else roc),
        float(-1 if pr is None else pr),
        float(-1 if top is None else top),
        covered,
    )


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "convention",
        "gold_shift",
        "gold_sites",
        "covered_gold_before_filter",
        "evaluated_sites",
        "evaluated_gold_sites",
        "evaluated_negative_sites",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "top100_gold_fraction",
        "top500_gold_fraction",
        "top1000_gold_fraction",
        "reference_base_checked",
        "reference_base_compatible",
        "reference_base_compatible_fraction",
        "motif_checked",
        "motif_matched",
        "motif_match_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_shifts(text: str) -> List[int]:
    shifts = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            shifts.append(int(item))
    return shifts


def main(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motif = motif_regex(args.motif)
    gold = load_gold_sites(args.gold_bed, gold_format=args.gold_format, ignore_strand=False)
    stats, counters = aggregate_modbam_sites(
        args.bam,
        canonical_base=args.canonical_base,
        mod_code=args.mod_code,
        keep_u=args.keep_u,
        include_non_primary=args.include_non_primary,
        ignore_strand=False,
        fasta=None,
        motif=None,
    )
    collapsed_stats = collapse_stats_by_strand(stats)

    fasta = pysam.FastaFile(str(args.reference)) if args.reference else None
    try:
        records = []
        for convention in args.conventions:
            for shift in parse_shifts(args.shifts):
                records.append(
                    evaluate_convention(
                        stats=stats,
                        collapsed_stats=collapsed_stats,
                        gold=gold,
                        convention=convention,
                        shift=shift,
                        min_coverage=args.min_coverage,
                        score_column=args.score_column,
                        threshold=args.prob_threshold,
                        fasta=fasta,
                        canonical_base=args.canonical_base,
                        motif=motif,
                        motif_kmer_size=args.motif_kmer_size,
                    )
                )
    finally:
        if fasta is not None:
            fasta.close()

    ranked = sorted(records, key=sort_key, reverse=True)
    summary = {
        "inputs": {
            "bam": str(args.bam.resolve()),
            "gold": str(args.gold_bed.resolve()),
            "reference": None if args.reference is None else str(args.reference.resolve()),
        },
        "settings": {
            "gold_format": args.gold_format,
            "canonical_base": args.canonical_base,
            "mod_code": args.mod_code,
            "min_coverage": args.min_coverage,
            "score_column": args.score_column,
            "prob_threshold": args.prob_threshold,
            "shifts": parse_shifts(args.shifts),
            "conventions": args.conventions,
            "motif": args.motif,
            "motif_kmer_size": args.motif_kmer_size,
            "include_non_primary": args.include_non_primary,
        },
        "bam_counters": counters,
        "best_by_roc_auc": ranked[0] if ranked else None,
        "records": ranked,
    }
    write_tsv(output_dir / "coordinate_convention_summary.tsv", ranked)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Top coordinate/strand conventions:")
    for record in ranked[: min(args.show_top, len(ranked))]:
        print(
            "{convention:>16} shift={gold_shift:+d} "
            "roc_auc={roc_auc} pr_auc={pr_auc} "
            "eval_gold={evaluated_gold_sites} ref_base_ok={reference_base_compatible_fraction} "
            "motif_ok={motif_match_fraction}".format(**record)
        )
    print(f"artifacts written to: {output_dir}")


def argparser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bam", type=Path, required=True, help="Aligned modBAM/SAM produced by tetramod basecaller.")
    parser.add_argument("--gold-bed", type=Path, required=True, help="Gold m6A sites in BED6 or m6A-Atlas table format.")
    parser.add_argument("--reference", type=Path, default=None, help="Reference FASTA for base/motif sanity checks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-format", choices=["auto", "bed", "m6aatlas"], default="auto")
    parser.add_argument("--canonical-base", default="A")
    parser.add_argument("--mod-code", default="a")
    parser.add_argument("--min-coverage", type=int, default=5)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument(
        "--score-column",
        choices=["mod_fraction", "mean_prob_zero_filled", "mean_called_prob", "median_called_prob"],
        default="mod_fraction",
    )
    parser.add_argument("--shifts", default="-2,-1,0,1,2", help="Comma-separated shifts added to gold starts.")
    parser.add_argument(
        "--conventions",
        nargs="+",
        choices=["as_is", "flip_gold_strand", "ignore_strand", "either_strand"],
        default=["as_is", "flip_gold_strand", "ignore_strand", "either_strand"],
    )
    parser.add_argument("--motif", default="DRACH", help="Optional motif sanity check; set empty string to disable.")
    parser.add_argument("--motif-kmer-size", type=int, default=5)
    parser.add_argument("--include-non-primary", action="store_true", default=False)
    parser.add_argument("--keep-u", action="store_true", default=False)
    parser.add_argument("--show-top", type=int, default=8)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
