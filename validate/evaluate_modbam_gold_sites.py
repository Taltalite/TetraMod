#!/usr/bin/env python3
"""
Evaluate site-level modified-base calls from a modBAM against gold m6A sites.

The script aggregates per-read MM/ML modified-base calls into reference
site-level scores, labels covered candidate sites with a gold BED/TXT file, and
writes metrics plus optional matplotlib plots.
"""

from __future__ import annotations

import csv
import json
import math
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pysam


SiteKey = Tuple[str, int, str]
GoldInfo = Dict[str, object]


@dataclass
class SiteStats:
    coverage: int = 0
    mod_probs: List[float] = field(default_factory=list)

    def add_coverage(self, prob: Optional[float]) -> None:
        self.coverage += 1
        if prob is not None:
            self.mod_probs.append(float(prob))

    @property
    def num_mod_calls(self) -> int:
        return len(self.mod_probs)

    @property
    def mod_fraction(self) -> float:
        return safe_div(self.num_mod_calls, self.coverage)

    @property
    def mean_prob_zero_filled(self) -> float:
        return safe_div(sum(self.mod_probs), self.coverage)

    @property
    def mean_called_prob(self) -> Optional[float]:
        if not self.mod_probs:
            return None
        return float(sum(self.mod_probs) / len(self.mod_probs))

    @property
    def median_called_prob(self) -> Optional[float]:
        if not self.mod_probs:
            return None
        return float(median(self.mod_probs))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def normalize_base(base: str, *, keep_u: bool = False) -> str:
    text = str(base or "").upper()
    return text if keep_u else text.replace("U", "T")


def site_key(chrom: str, start: int, strand: str, *, ignore_strand: bool = False) -> SiteKey:
    return str(chrom), int(start), "." if ignore_strand else str(strand or ".")


def parse_number(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        number = float(text)
    except ValueError:
        return default
    if math.isnan(number):
        return default
    return number


def parse_bed_gold(path: Path, *, ignore_strand: bool = False) -> Dict[SiteKey, GoldInfo]:
    gold: Dict[SiteKey, GoldInfo] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#") or text.startswith("track ") or text.startswith("browser "):
                continue
            fields = re.split(r"\t+|\s+", text)
            if len(fields) < 3:
                raise ValueError(f"Invalid BED line {line_number}: expected at least 3 columns")
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            name = fields[3] if len(fields) >= 4 else f"{chrom}:{start}-{end}"
            score = parse_number(fields[4], default=None) if len(fields) >= 5 else None
            strand = fields[5] if len(fields) >= 6 and fields[5] in {"+", "-"} else "."
            key = site_key(chrom, start, strand, ignore_strand=ignore_strand)
            gold[key] = {
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": "." if ignore_strand else strand,
                "name": name,
                "support": score,
                "source_line": line_number,
            }
    return gold


def _normalise_header(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _pick_column(header: Sequence[str], names: Sequence[str]) -> Optional[str]:
    lookup = {_normalise_header(column): column for column in header}
    for name in names:
        column = lookup.get(_normalise_header(name))
        if column is not None:
            return column
    return None


def parse_m6a_atlas_gold(path: Path, *, ignore_strand: bool = False) -> Dict[SiteKey, GoldInfo]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("m6A-Atlas input is missing a header row.")

        chrom_col = _pick_column(reader.fieldnames, ["chrom", "chr", "chromosome"])
        pos_col = _pick_column(reader.fieldnames, ["position", "pos", "site", "genomicposition"])
        start_col = _pick_column(reader.fieldnames, ["start", "chromstart"])
        strand_col = _pick_column(reader.fieldnames, ["strand"])
        support_col = _pick_column(
            reader.fieldnames,
            ["techniquenum", "technique num", "conditionnum", "condition num", "support", "score"],
        )
        id_col = _pick_column(reader.fieldnames, ["id", "siteid", "name"])

        if chrom_col is None:
            raise ValueError("Unable to detect chromosome column in m6A-Atlas file.")
        if pos_col is None and start_col is None:
            raise ValueError("Unable to detect position/start column in m6A-Atlas file.")

        gold: Dict[SiteKey, GoldInfo] = {}
        for line_number, row in enumerate(reader, start=2):
            chrom = str(row.get(chrom_col, "")).strip()
            if not chrom:
                continue
            if pos_col is not None:
                start = int(float(str(row[pos_col]).strip())) - 1
            else:
                start = int(float(str(row[start_col]).strip()))
            strand = str(row.get(strand_col, ".")).strip() if strand_col else "."
            if strand not in {"+", "-"}:
                strand = "."
            name = str(row.get(id_col, "")).strip() if id_col else ""
            if not name:
                name = f"{chrom}:{start}-{start + 1}:{strand}"
            support = parse_number(row.get(support_col), default=None) if support_col else None
            key = site_key(chrom, start, strand, ignore_strand=ignore_strand)
            gold[key] = {
                "chrom": chrom,
                "start": start,
                "end": start + 1,
                "strand": "." if ignore_strand else strand,
                "name": name,
                "support": support,
                "source_line": line_number,
            }
    return gold


def detect_gold_format(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = re.split(r"\t+|\s+", text)
            if len(fields) >= 3:
                try:
                    int(fields[1])
                    int(fields[2])
                except ValueError:
                    return "m6aatlas"
                return "bed"
            return "m6aatlas"
    raise ValueError(f"Gold file is empty: {path}")


def load_gold_sites(path: Path, *, gold_format: str = "auto", ignore_strand: bool = False) -> Dict[SiteKey, GoldInfo]:
    fmt = detect_gold_format(path) if gold_format == "auto" else gold_format
    if fmt == "bed":
        return parse_bed_gold(path, ignore_strand=ignore_strand)
    if fmt == "m6aatlas":
        return parse_m6a_atlas_gold(path, ignore_strand=ignore_strand)
    raise ValueError(f"Unsupported gold format: {gold_format}")


def query_to_reference_map(record) -> Dict[int, int]:
    mapping = {}
    for query_pos, reference_pos in record.get_aligned_pairs(matches_only=False):
        if query_pos is not None and reference_pos is not None:
            mapping[int(query_pos)] = int(reference_pos)
    return mapping


def fallback_parse_mm_ml(record, *, canonical_base: str, mod_code: str) -> Dict[int, float]:
    try:
        mm_tag = record.get_tag("MM")
        ml_tag = list(record.get_tag("ML"))
    except KeyError:
        return {}

    canonical_base = canonical_base.upper()
    mod_code = mod_code.lower()
    sequence = str(record.query_sequence or "").upper()
    canonical_positions = [
        idx for idx, base in enumerate(sequence)
        if normalize_base(base) == canonical_base
    ]

    calls: Dict[int, float] = {}
    ml_index = 0
    for group in str(mm_tag).split(";"):
        if not group:
            continue
        match = re.match(r"^([A-Za-z])([+-])([^,.?;]+)[.?]?,(.*)$", group)
        if not match:
            continue
        canonical, _strand, code, deltas_text = match.groups()
        deltas = [item for item in deltas_text.split(",") if item != ""]
        if canonical.upper() != canonical_base or code.lower() != mod_code:
            ml_index += len(deltas)
            continue
        canonical_index = -1
        for delta_text in deltas:
            canonical_index += int(delta_text) + 1
            if canonical_index >= len(canonical_positions) or ml_index >= len(ml_tag):
                break
            calls[canonical_positions[canonical_index]] = float(ml_tag[ml_index]) / 255.0
            ml_index += 1
    return calls


def read_modified_base_probs(record, *, canonical_base: str, mod_code: str) -> Dict[int, float]:
    canonical_base = canonical_base.upper()
    mod_code = mod_code.lower()
    calls: Dict[int, float] = {}

    try:
        modified_bases = record.modified_bases or {}
    except (KeyError, ValueError, AttributeError):
        modified_bases = {}

    for key, values in modified_bases.items():
        base, _strand, code = key
        if str(base).upper() != canonical_base or str(code).lower() != mod_code:
            continue
        for query_pos, qual in values:
            prob = None if qual is None or int(qual) < 0 else float(qual) / 255.0
            if prob is not None:
                previous = calls.get(int(query_pos))
                calls[int(query_pos)] = prob if previous is None else max(previous, prob)

    if calls:
        return calls
    return fallback_parse_mm_ml(record, canonical_base=canonical_base, mod_code=mod_code)


def record_strand(record) -> str:
    return "-" if record.is_reverse else "+"


def motif_regex(motif: Optional[str]) -> Optional[re.Pattern]:
    if motif is None or str(motif).strip() == "":
        return None
    text = str(motif).upper()
    if text == "DRACH":
        return re.compile(r"^[AGT][AG]AC[ACT]$")
    if text == "RRACH":
        return re.compile(r"^[AG][AG]AC[ACT]$")
    return re.compile(text)


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")
    return str(seq).translate(table)[::-1].upper()


def site_context(fasta, chrom: str, start: int, strand: str, kmer_size: int) -> Optional[str]:
    if fasta is None or kmer_size <= 1:
        return None
    half = kmer_size // 2
    left = start - half
    right = start + half + 1
    if left < 0:
        return None
    try:
        seq = fasta.fetch(chrom, left, right).upper()
    except Exception:
        return None
    if len(seq) != kmer_size or "N" in seq:
        return None
    if strand == "-":
        seq = reverse_complement(seq)
    return seq


def motif_allowed(
    fasta,
    chrom: str,
    start: int,
    strand: str,
    motif: Optional[re.Pattern],
    kmer_size: int,
) -> bool:
    if motif is None:
        return True
    context = site_context(fasta, chrom, start, strand, kmer_size)
    return bool(context and motif.search(context))


def aggregate_modbam_sites(
    bam_path: Path,
    *,
    canonical_base: str,
    mod_code: str,
    keep_u: bool = False,
    include_non_primary: bool = False,
    ignore_strand: bool = False,
    fasta=None,
    motif: Optional[re.Pattern] = None,
    motif_kmer_size: int = 5,
) -> Tuple[Dict[SiteKey, SiteStats], Dict[str, int]]:
    stats: Dict[SiteKey, SiteStats] = defaultdict(SiteStats)
    counters = {
        "records_seen": 0,
        "records_used": 0,
        "records_skipped_unmapped": 0,
        "records_skipped_non_primary": 0,
        "covered_candidate_bases": 0,
        "modified_calls_projected": 0,
    }
    canonical_base = canonical_base.upper()

    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for record in bam.fetch(until_eof=True):
            counters["records_seen"] += 1
            if record.is_unmapped:
                counters["records_skipped_unmapped"] += 1
                continue
            if not include_non_primary and (record.is_secondary or record.is_supplementary):
                counters["records_skipped_non_primary"] += 1
                continue
            if record.query_sequence is None or record.reference_name is None:
                continue

            counters["records_used"] += 1
            qpos_to_ref = query_to_reference_map(record)
            mod_probs = read_modified_base_probs(
                record,
                canonical_base=canonical_base,
                mod_code=mod_code,
            )
            chrom = str(record.reference_name)
            strand = record_strand(record)
            sequence = str(record.query_sequence)

            for query_pos, reference_pos in qpos_to_ref.items():
                if query_pos >= len(sequence):
                    continue
                base = normalize_base(sequence[query_pos], keep_u=keep_u)
                if base != canonical_base:
                    continue
                if not motif_allowed(fasta, chrom, reference_pos, strand, motif, motif_kmer_size):
                    continue
                key = site_key(chrom, reference_pos, strand, ignore_strand=ignore_strand)
                prob = mod_probs.get(query_pos)
                stats[key].add_coverage(prob)
                counters["covered_candidate_bases"] += 1
                if prob is not None:
                    counters["modified_calls_projected"] += 1

    return stats, counters


def score_for_site(stats: SiteStats, score_column: str) -> Optional[float]:
    if score_column == "mod_fraction":
        return stats.mod_fraction
    if score_column == "mean_prob_zero_filled":
        return stats.mean_prob_zero_filled
    if score_column == "mean_called_prob":
        return stats.mean_called_prob
    if score_column == "median_called_prob":
        return stats.median_called_prob
    raise ValueError(f"Unsupported score column: {score_column}")


def build_site_rows(
    stats: Dict[SiteKey, SiteStats],
    gold: Dict[SiteKey, GoldInfo],
    *,
    min_coverage: int,
    score_column: str,
    threshold: float,
) -> List[Dict[str, object]]:
    rows = []
    for key, site_stats in sorted(stats.items()):
        if site_stats.coverage < min_coverage:
            continue
        chrom, start, strand = key
        gold_info = gold.get(key)
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
            "is_gold": int(gold_info is not None),
            "gold_name": "" if gold_info is None else str(gold_info.get("name", "")),
            "gold_support": "" if gold_info is None or gold_info.get("support") is None else gold_info.get("support"),
            "predicted_modified": int(float(score) >= threshold),
        })
    return rows


def require_sklearn_metrics():
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required for gold-site metric evaluation. "
            "Install the project dependencies, then rerun this script."
        ) from exc

    return {
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_recall_curve": precision_recall_curve,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
    }


def sklearn_curves(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, object]:
    metrics = require_sklearn_metrics()
    y_true = y_true.astype(np.int64)
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "fpr": np.array([], dtype=np.float32),
            "tpr": np.array([], dtype=np.float32),
            "precision": np.array([], dtype=np.float32),
            "recall": np.array([], dtype=np.float32),
            "thresholds": np.array([], dtype=np.float32),
        }

    fpr, tpr, roc_thresholds = metrics["roc_curve"](y_true, y_score)
    precision, recall, pr_thresholds = metrics["precision_recall_curve"](y_true, y_score)
    return {
        "roc_auc": float(metrics["roc_auc_score"](y_true, y_score)),
        "pr_auc": float(metrics["average_precision_score"](y_true, y_score)),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "thresholds": {
            "roc": roc_thresholds,
            "pr": pr_thresholds,
        },
    }


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    metrics = require_sklearn_metrics()
    y_pred = (y_score >= threshold).astype(np.int64)
    matrix = metrics["confusion_matrix"](y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(metrics["accuracy_score"](y_true, y_pred)),
        "precision": float(metrics["precision_score"](y_true, y_pred, zero_division=0)),
        "recall": float(metrics["recall_score"](y_true, y_pred, zero_division=0)),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(metrics["f1_score"](y_true, y_pred, zero_division=0)),
        "fdr": safe_div(fp, tp + fp),
    }


def sensitivity_at_fdr(y_true: np.ndarray, y_score: np.ndarray, fdr_levels: Sequence[float]) -> Dict[str, float]:
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order].astype(np.int64)
    positives = int(y_true.sum())
    if positives == 0:
        return {str(level): 0.0 for level in fdr_levels}

    tps = np.cumsum(sorted_true == 1)
    fps = np.cumsum(sorted_true == 0)
    fdr = fps / np.maximum(tps + fps, 1)
    recall = tps / positives
    result = {}
    for level in fdr_levels:
        keep = recall[fdr <= float(level)]
        result[str(level)] = float(keep.max()) if keep.size else 0.0
    return result


def top_n_hits(y_true: np.ndarray, y_score: np.ndarray, top_ns: Sequence[int]) -> Dict[str, float]:
    order = np.argsort(-y_score, kind="mergesort")
    result = {}
    for n in top_ns:
        if n <= 0:
            continue
        subset = y_true[order[: min(int(n), len(order))]]
        result[str(n)] = safe_div(int(subset.sum()), int(subset.size))
    return result


def compute_metrics(rows: List[Dict[str, object]], threshold: float) -> Dict[str, object]:
    y_true = np.asarray([int(row["is_gold"]) for row in rows], dtype=np.int64)
    y_score = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    curves = sklearn_curves(y_true, y_score) if len(rows) else {
        "roc_auc": None,
        "pr_auc": None,
    }
    metrics = {
        "num_sites": int(len(rows)),
        "num_positive": int(y_true.sum()),
        "num_negative": int((1 - y_true).sum()),
        "roc_auc": curves["roc_auc"],
        "pr_auc": curves["pr_auc"],
        "threshold_metrics": threshold_metrics(y_true, y_score, threshold) if len(rows) else {},
        "sensitivity_at_fdr": sensitivity_at_fdr(y_true, y_score, [0.01, 0.05, 0.10]),
        "top_n_gold_fraction": top_n_hits(y_true, y_score, [100, 500, 1000, 5000]),
    }
    return metrics


def write_tsv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def optional_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def save_plots(rows: List[Dict[str, object]], output_dir: Path, threshold: float) -> List[str]:
    plt = optional_pyplot()
    if plt is None or not rows:
        return []

    y_true = np.asarray([int(row["is_gold"]) for row in rows], dtype=np.int64)
    y_score = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    coverage = np.asarray([int(row["coverage"]) for row in rows], dtype=np.int64)
    curves = sklearn_curves(y_true, y_score)
    written = []

    if curves["roc_auc"] is not None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(curves["fpr"], curves["tpr"], color="#2c7fb8", label=f"AUC={curves['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend()
        ax.grid(alpha=0.2)
        path = output_dir / "roc_curve.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(curves["recall"], curves["precision"], color="#31a354", label=f"AUC={curves['pr_auc']:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-recall curve")
        ax.legend()
        ax.grid(alpha=0.2)
        path = output_dir / "pr_curve.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(y_score[y_true == 0], bins=50, alpha=0.65, label="nongold", color="#636363")
    ax.hist(y_score[y_true == 1], bins=50, alpha=0.65, label="gold", color="#d62728")
    ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1, label=f"threshold={threshold:g}")
    ax.set_xlabel("Site score")
    ax.set_ylabel("Site count")
    ax.set_title("Site score distribution")
    ax.legend()
    ax.grid(alpha=0.2)
    path = output_dir / "score_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = np.where(y_true == 1, "#d62728", "#636363")
    ax.scatter(coverage, y_score, s=8, alpha=0.45, c=colors)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Site score")
    ax.set_title("Coverage vs site score")
    ax.grid(alpha=0.2)
    path = output_dir / "coverage_vs_score.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    tm = threshold_metrics(y_true, y_score, threshold)
    matrix = np.asarray([[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#111111")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    return written


def build_text_summary(summary: Dict[str, object]) -> str:
    metrics = summary["metrics"]
    tm = metrics["threshold_metrics"]
    lines = [
        "[inputs]",
        f"bam: {summary['inputs']['bam']}",
        f"gold: {summary['inputs']['gold']}",
        "",
        "[counts]",
        f"gold_sites: {summary['counts']['gold_sites']}",
        f"covered_sites_before_filter: {summary['counts']['covered_sites_before_filter']}",
        f"evaluated_sites: {metrics['num_sites']}",
        f"evaluated_gold_sites: {metrics['num_positive']}",
        f"evaluated_negative_sites: {metrics['num_negative']}",
        f"gold_sites_covered_after_filter: {summary['counts']['gold_sites_covered_after_filter']}",
        "",
        "[metrics]",
        f"score_column: {summary['settings']['score_column']}",
        f"roc_auc: {metrics['roc_auc']}",
        f"pr_auc: {metrics['pr_auc']}",
        f"threshold: {tm.get('threshold')}",
        f"precision: {tm.get('precision')}",
        f"recall: {tm.get('recall')}",
        f"f1: {tm.get('f1')}",
        f"fdr: {tm.get('fdr')}",
    ]
    return "\n".join(lines) + "\n"


def main(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.motif and args.motif_kmer_size % 2 == 0:
        raise ValueError("--motif-kmer-size must be odd")

    gold = load_gold_sites(args.gold_bed, gold_format=args.gold_format, ignore_strand=args.ignore_strand)
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
        )
    finally:
        if fasta is not None:
            fasta.close()

    rows = build_site_rows(
        stats,
        gold,
        min_coverage=args.min_coverage,
        score_column=args.score_column,
        threshold=args.prob_threshold,
    )
    metrics = compute_metrics(rows, args.prob_threshold)

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
        "is_gold",
        "gold_name",
        "gold_support",
        "predicted_modified",
    ]
    write_tsv(output_dir / "site_level_predictions.tsv", rows, fieldnames)
    write_tsv(output_dir / "gold_overlap.tsv", [row for row in rows if int(row["is_gold"]) == 1], fieldnames)
    write_tsv(output_dir / "negative_sites.tsv", [row for row in rows if int(row["is_gold"]) == 0], fieldnames)

    covered_gold_keys = set(stats) & set(gold)
    covered_gold_after_filter = {
        site_key(str(row["chrom"]), int(row["start"]), str(row["strand"]), ignore_strand=args.ignore_strand)
        for row in rows
        if int(row["is_gold"]) == 1
    }
    plots = save_plots(rows, output_dir, args.prob_threshold) if not args.no_plots else []

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
            "ignore_strand": args.ignore_strand,
            "include_non_primary": args.include_non_primary,
            "motif": args.motif,
            "motif_kmer_size": args.motif_kmer_size,
        },
        "bam_counters": bam_counters,
        "counts": {
            "gold_sites": int(len(gold)),
            "covered_sites_before_filter": int(len(stats)),
            "covered_sites_after_filter": int(len(rows)),
            "gold_sites_covered_before_filter": int(len(covered_gold_keys)),
            "gold_sites_covered_after_filter": int(len(covered_gold_after_filter)),
            "gold_sites_uncovered_before_filter": int(len(set(gold) - covered_gold_keys)),
        },
        "metrics": metrics,
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
    parser.add_argument("--bam", type=Path, required=True, help="Aligned modBAM/SAM produced by tetramod basecaller.")
    parser.add_argument("--gold-bed", type=Path, required=True, help="Gold m6A sites in BED6 or m6A-Atlas table format.")
    parser.add_argument("--reference", type=Path, default=None, help="Reference FASTA, required for motif filtering.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-format", choices=["auto", "bed", "m6aatlas"], default="auto")
    parser.add_argument("--canonical-base", default="A")
    parser.add_argument("--mod-code", default="a", help="SAM MM modification code, e.g. 'a' for m6A.")
    parser.add_argument("--min-coverage", type=int, default=5)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument(
        "--score-column",
        choices=["mod_fraction", "mean_prob_zero_filled", "mean_called_prob", "median_called_prob"],
        default="mod_fraction",
    )
    parser.add_argument("--ignore-strand", action="store_true", default=False)
    parser.add_argument("--include-non-primary", action="store_true", default=False)
    parser.add_argument("--keep-u", action="store_true", default=False)
    parser.add_argument("--motif", default=None, help="Optional motif filter: DRACH, RRACH, or a regex.")
    parser.add_argument("--motif-kmer-size", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main(argparser().parse_args())
