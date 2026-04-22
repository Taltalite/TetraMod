#!/usr/bin/env python3
"""
Compare TetraMod and Bonito basecaller BAM/SAM outputs on shared read ids.

Outputs:
- per_read_comparison.tsv
- worst_reads.tsv
- summary.json
- summary.txt
"""

from __future__ import annotations

import csv
import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional

import numpy as np
import pysam
from edlib import align as edlib_align


CIGAR_RE = re.compile(r"(\d+)([=XID])")


def normalize_seq(seq: str, *, keep_u: bool) -> str:
    text = str(seq or "").upper()
    return text if keep_u else text.replace("U", "T")


def open_alignment(path: Path):
    mode = "r" if path.suffix.lower() == ".sam" else "rb"
    return pysam.AlignmentFile(str(path), mode, check_sq=False)


def mean_query_quality(record) -> Optional[float]:
    qualities = record.query_qualities
    if qualities is None:
        return None
    values = list(qualities)
    if not values:
        return None
    return float(sum(values) / len(values))


def read_tag(record, tag: str):
    try:
        return record.get_tag(tag)
    except KeyError:
        return None


def load_bam_records(
    path: Path,
    *,
    keep_u: bool,
    primary_only: bool,
) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    duplicate_ids = 0
    skipped_non_primary = 0
    skipped_without_sequence = 0

    with open_alignment(path) as bam:
        for record in bam:
            if primary_only and (record.is_secondary or record.is_supplementary):
                skipped_non_primary += 1
                continue
            if record.query_sequence is None:
                skipped_without_sequence += 1
                continue

            read_id = str(record.query_name)
            if read_id in records:
                duplicate_ids += 1
                continue

            records[read_id] = {
                "sequence": normalize_seq(record.query_sequence, keep_u=keep_u),
                "mean_qscore": mean_query_quality(record),
                "mapq": int(record.mapping_quality),
                "is_unmapped": bool(record.is_unmapped),
                "is_reverse": bool(record.is_reverse),
                "reference_name": None if record.reference_id < 0 else record.reference_name,
                "reference_start": int(record.reference_start) if record.reference_start >= 0 else None,
                "qs_tag": read_tag(record, "qs"),
            }

    records["_meta"] = {
        "path": str(path.resolve()),
        "num_indexed_reads": len(records),
        "num_duplicate_read_ids_skipped": duplicate_ids,
        "num_non_primary_skipped": skipped_non_primary,
        "num_without_sequence_skipped": skipped_without_sequence,
    }
    return records


def parse_cigar_counts(cigar: str) -> Dict[str, int]:
    counts = {"=": 0, "X": 0, "I": 0, "D": 0}
    for count, op in CIGAR_RE.findall(cigar or ""):
        counts[op] += int(count)
    return counts


def compare_sequences(query_seq: str, target_seq: str) -> Dict[str, object]:
    result = edlib_align(query_seq, target_seq, mode="NW", task="path")
    counts = parse_cigar_counts(result.get("cigar", ""))
    aligned = counts["="] + counts["X"] + counts["I"] + counts["D"]
    identity = float(counts["="] / aligned) if aligned else 0.0
    return {
        "identity": identity,
        "edit_distance": int(result.get("editDistance", -1)),
        "matches": int(counts["="]),
        "mismatches": int(counts["X"]),
        "insertions_vs_bonito": int(counts["I"]),
        "deletions_vs_bonito": int(counts["D"]),
    }


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def numeric_delta(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def build_text_summary(summary: Dict[str, object]) -> str:
    metrics = summary["metrics"]
    counts = summary["counts"]
    lines = [
        "[inputs]",
        f"tetramod_bam: {summary['tetramod_bam']}",
        f"bonito_bam: {summary['bonito_bam']}",
        "",
        "[counts]",
        f"tetramod_reads: {counts['tetramod_reads']}",
        f"bonito_reads: {counts['bonito_reads']}",
        f"shared_reads: {counts['shared_reads']}",
        f"tetramod_only_reads: {counts['tetramod_only_reads']}",
        f"bonito_only_reads: {counts['bonito_only_reads']}",
        "",
        "[sequence_comparison]",
        f"mean_identity: {metrics['mean_identity']:.6f}",
        f"median_identity: {metrics['median_identity']:.6f}",
        f"mean_edit_distance: {metrics['mean_edit_distance']:.3f}",
        f"same_length_fraction: {metrics['same_length_fraction']:.6f}",
        f"exact_match_fraction: {metrics['exact_match_fraction']:.6f}",
        f"mean_tetramod_length: {metrics['mean_tetramod_length']:.2f}",
        f"mean_bonito_length: {metrics['mean_bonito_length']:.2f}",
        f"mean_length_delta_tetramod_minus_bonito: {metrics['mean_length_delta']:.2f}",
        "",
        "[quality]",
        f"mean_tetramod_qscore: {metrics['mean_tetramod_qscore']}",
        f"mean_bonito_qscore: {metrics['mean_bonito_qscore']}",
        f"mean_qscore_delta_tetramod_minus_bonito: {metrics['mean_qscore_delta']}",
    ]
    return "\n".join(lines) + "\n"


def main(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tetramod_records = load_bam_records(
        Path(args.tetramod_bam),
        keep_u=args.keep_u,
        primary_only=not args.include_non_primary,
    )
    tetramod_meta = tetramod_records.pop("_meta")
    bonito_records = load_bam_records(
        Path(args.bonito_bam),
        keep_u=args.keep_u,
        primary_only=not args.include_non_primary,
    )
    bonito_meta = bonito_records.pop("_meta")

    shared_ids = sorted(set(tetramod_records) & set(bonito_records))
    rows: List[Dict[str, object]] = []

    identities: List[float] = []
    edit_distances: List[int] = []
    exact_matches = 0
    same_lengths = 0
    length_deltas: List[int] = []
    tetramod_lengths: List[int] = []
    bonito_lengths: List[int] = []
    qscore_deltas: List[Optional[float]] = []

    for read_id in shared_ids:
        tetra = tetramod_records[read_id]
        bonito = bonito_records[read_id]
        tetra_seq = str(tetra["sequence"])
        bonito_seq = str(bonito["sequence"])
        comparison = compare_sequences(tetra_seq, bonito_seq)

        tetra_len = len(tetra_seq)
        bonito_len = len(bonito_seq)
        length_delta = tetra_len - bonito_len
        qscore_delta = numeric_delta(tetra["mean_qscore"], bonito["mean_qscore"])

        identities.append(float(comparison["identity"]))
        edit_distances.append(int(comparison["edit_distance"]))
        exact_matches += int(comparison["edit_distance"] == 0)
        same_lengths += int(tetra_len == bonito_len)
        length_deltas.append(length_delta)
        tetramod_lengths.append(tetra_len)
        bonito_lengths.append(bonito_len)
        qscore_deltas.append(qscore_delta)

        rows.append({
            "read_id": read_id,
            "identity": comparison["identity"],
            "edit_distance": comparison["edit_distance"],
            "matches": comparison["matches"],
            "mismatches": comparison["mismatches"],
            "insertions_vs_bonito": comparison["insertions_vs_bonito"],
            "deletions_vs_bonito": comparison["deletions_vs_bonito"],
            "tetramod_length": tetra_len,
            "bonito_length": bonito_len,
            "length_delta_tetramod_minus_bonito": length_delta,
            "same_length": int(tetra_len == bonito_len),
            "exact_match": int(comparison["edit_distance"] == 0),
            "tetramod_mean_qscore": tetra["mean_qscore"],
            "bonito_mean_qscore": bonito["mean_qscore"],
            "qscore_delta_tetramod_minus_bonito": qscore_delta,
            "tetramod_qs_tag": tetra["qs_tag"],
            "bonito_qs_tag": bonito["qs_tag"],
            "tetramod_mapq": tetra["mapq"],
            "bonito_mapq": bonito["mapq"],
            "tetramod_is_unmapped": int(tetra["is_unmapped"]),
            "bonito_is_unmapped": int(bonito["is_unmapped"]),
            "tetramod_is_reverse": int(tetra["is_reverse"]),
            "bonito_is_reverse": int(bonito["is_reverse"]),
            "tetramod_reference_name": tetra["reference_name"],
            "bonito_reference_name": bonito["reference_name"],
            "tetramod_reference_start": tetra["reference_start"],
            "bonito_reference_start": bonito["reference_start"],
            "tetramod_sequence_prefix": tetra_seq[:120],
            "bonito_sequence_prefix": bonito_seq[:120],
        })

    rows.sort(key=lambda row: (row["identity"], row["edit_distance"], row["read_id"]))

    fieldnames = [
        "read_id",
        "identity",
        "edit_distance",
        "matches",
        "mismatches",
        "insertions_vs_bonito",
        "deletions_vs_bonito",
        "tetramod_length",
        "bonito_length",
        "length_delta_tetramod_minus_bonito",
        "same_length",
        "exact_match",
        "tetramod_mean_qscore",
        "bonito_mean_qscore",
        "qscore_delta_tetramod_minus_bonito",
        "tetramod_qs_tag",
        "bonito_qs_tag",
        "tetramod_mapq",
        "bonito_mapq",
        "tetramod_is_unmapped",
        "bonito_is_unmapped",
        "tetramod_is_reverse",
        "bonito_is_reverse",
        "tetramod_reference_name",
        "bonito_reference_name",
        "tetramod_reference_start",
        "bonito_reference_start",
        "tetramod_sequence_prefix",
        "bonito_sequence_prefix",
    ]

    per_read_path = out_dir / "per_read_comparison.tsv"
    with per_read_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    worst_path = out_dir / "worst_reads.tsv"
    with worst_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows[: args.num_examples])

    num_shared = len(shared_ids)
    summary = {
        "tetramod_bam": str(Path(args.tetramod_bam).resolve()),
        "bonito_bam": str(Path(args.bonito_bam).resolve()),
        "settings": {
            "include_non_primary": bool(args.include_non_primary),
            "keep_u": bool(args.keep_u),
        },
        "input_metadata": {
            "tetramod": tetramod_meta,
            "bonito": bonito_meta,
        },
        "counts": {
            "tetramod_reads": int(len(tetramod_records)),
            "bonito_reads": int(len(bonito_records)),
            "shared_reads": int(num_shared),
            "tetramod_only_reads": int(len(set(tetramod_records) - set(bonito_records))),
            "bonito_only_reads": int(len(set(bonito_records) - set(tetramod_records))),
        },
        "metrics": {
            "mean_identity": float(sum(identities) / num_shared) if num_shared else 0.0,
            "median_identity": float(median(identities)) if identities else 0.0,
            "mean_edit_distance": float(sum(edit_distances) / num_shared) if num_shared else 0.0,
            "same_length_fraction": float(same_lengths / num_shared) if num_shared else 0.0,
            "exact_match_fraction": float(exact_matches / num_shared) if num_shared else 0.0,
            "mean_tetramod_length": float(sum(tetramod_lengths) / num_shared) if num_shared else 0.0,
            "mean_bonito_length": float(sum(bonito_lengths) / num_shared) if num_shared else 0.0,
            "mean_length_delta": float(sum(length_deltas) / num_shared) if num_shared else 0.0,
            "mean_tetramod_qscore": mean_or_none(row["tetramod_mean_qscore"] for row in rows),
            "mean_bonito_qscore": mean_or_none(row["bonito_mean_qscore"] for row in rows),
            "mean_qscore_delta": mean_or_none(qscore_deltas),
        },
        "artifacts": {
            "per_read_comparison_tsv": str(per_read_path.resolve()),
            "worst_reads_tsv": str(worst_path.resolve()),
        },
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    (out_dir / "summary.txt").write_text(build_text_summary(summary), encoding="utf-8")

    print(build_text_summary(summary), end="")
    print(f"artifacts written to: {out_dir}")


def argparser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=True)
    parser.add_argument("--tetramod-bam", type=Path, required=True, help="BAM/SAM output from tetramod basecaller.")
    parser.add_argument("--bonito-bam", type=Path, required=True, help="BAM/SAM output from bonito basecaller.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=100, help="Number of lowest-identity reads to write to worst_reads.tsv.")
    parser.add_argument("--include-non-primary", action="store_true", default=False)
    parser.add_argument("--keep-u", action="store_true", default=False, help="Do not normalize U to T before comparing sequences.")
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
