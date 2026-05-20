#!/usr/bin/env python3
"""
Aggregate mAFiA leave-one-motif-out evaluation outputs.

The script expects per-motif evaluation directories produced by
validate/evaluate_mafia_stage1.py with --write-sites, for example:

    eval_root/leave_GAACT/heldout_Mix_1_A_RTA/site_predictions.tsv
    eval_root/leave_GAACT/heldout_Mix_2_m6A_RTA/site_predictions.tsv

It filters rows to the held-out motif and computes combined binary metrics
across all heldout runs for each LOMO model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate.evaluate_mafia_stage1 import binary_metrics


DEFAULT_MOTIFS = ("AGACT", "GAACT", "GGACA", "GGACC", "GGACT", "TGACT")
METRIC_FIELDS = (
    "num_sites",
    "num_positive",
    "num_negative",
    "positive_rate",
    "predicted_positive_rate",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "bce",
    "mean_prob",
    "median_prob",
    "mean_positive_prob",
    "mean_negative_prob",
    "median_positive_prob",
    "median_negative_prob",
    "q05_positive_prob",
    "q95_negative_prob",
)


def parse_motifs(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return list(DEFAULT_MOTIFS)
    return [item.strip().upper().replace("U", "T") for item in str(value).split(",") if item.strip()]


def safe_name(motif: str) -> str:
    return str(motif).strip().upper().replace("U", "T").replace("/", "_")


def read_site_predictions(path: Path, motif: str) -> tuple[list[int], list[float], int]:
    motif = motif.upper().replace("U", "T")
    y_true: list[int] = []
    y_prob: list[float] = []
    total_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"target", "prob_m6a", "motif_context"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        for row in reader:
            total_rows += 1
            row_motif = str(row["motif_context"]).upper().replace("U", "T")
            if row_motif != motif:
                continue
            y_true.append(int(row["target"]))
            y_prob.append(float(row["prob_m6a"]))
    return y_true, y_prob, total_rows


def iter_eval_dirs(root: Path, motif: str, pattern: str) -> Iterable[Path]:
    motif_dir = root / f"leave_{safe_name(motif)}"
    if not motif_dir.exists():
        return []
    return sorted(path for path in motif_dir.glob(pattern) if path.is_dir())


def aggregate_motif(root: Path, motif: str, pattern: str, threshold: float) -> dict[str, object]:
    eval_dirs = list(iter_eval_dirs(root, motif, pattern))
    y_true: list[int] = []
    y_prob: list[float] = []
    used_dirs = []
    scanned_rows = 0
    missing_site_predictions = []

    for directory in eval_dirs:
        site_path = directory / "site_predictions.tsv"
        if not site_path.exists():
            missing_site_predictions.append(str(directory))
            continue
        labels, probs, total_rows = read_site_predictions(site_path, motif)
        scanned_rows += int(total_rows)
        if labels:
            used_dirs.append(str(directory))
            y_true.extend(labels)
            y_prob.extend(probs)

    if not y_true:
        metrics = {field: None for field in METRIC_FIELDS}
        metrics.update({"num_sites": 0, "num_positive": 0, "num_negative": 0, "threshold": float(threshold)})
    else:
        metrics = binary_metrics(
            np.asarray(y_true, dtype=np.int64),
            np.asarray(y_prob, dtype=np.float32),
            threshold,
        )

    return {
        "heldout_motif": motif,
        "eval_root": str((root / f"leave_{safe_name(motif)}").resolve()),
        "eval_dirs_found": int(len(eval_dirs)),
        "eval_dirs_used": int(len(used_dirs)),
        "scanned_site_rows": int(scanned_rows),
        "used_dirs": used_dirs,
        "missing_site_predictions": missing_site_predictions,
        **metrics,
    }


def write_tsv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "heldout_motif",
        "eval_dirs_found",
        "eval_dirs_used",
        "scanned_site_rows",
        "num_sites",
        "num_positive",
        "num_negative",
        "positive_rate",
        "predicted_positive_rate",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "bce",
        "mean_prob",
        "median_prob",
        "mean_positive_prob",
        "mean_negative_prob",
        "q05_positive_prob",
        "q95_negative_prob",
        "eval_root",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    motifs = parse_motifs(args.motifs)
    rows = [aggregate_motif(root, motif, args.eval_glob, float(args.threshold)) for motif in motifs]
    summary = {
        "eval_root": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "motifs": motifs,
        "eval_glob": str(args.eval_glob),
        "threshold": float(args.threshold),
        "rows": rows,
    }
    write_tsv(output_dir / "lomo_summary.tsv", rows)
    (output_dir / "lomo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote LOMO benchmark summary to: {output_dir}")
    for row in rows:
        print(
            f"{row['heldout_motif']}: sites={row['num_sites']} "
            f"roc_auc={row['roc_auc']} pr_auc={row['pr_auc']} "
            f"recall={row['recall']} specificity={row['specificity']} bce={row['bce']}"
        )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    parser.add_argument("--eval-glob", default="heldout_*")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
