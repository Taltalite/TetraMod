#!/usr/bin/env python3
"""
Plot bag-level ROC curves from validate/evaluate_llp_bags.py outputs.

Stage 1 ROC is treated as a standard binary ROC and requires 0% and 100%
control bags:
    negative = 0% control bag
    positive = 100% full-mod bag

Stage 2 ROC is diagnostic only. Ratio-IVT labels are proportions, not native
binary labels, so each curve compares one lower ratio against one higher ratio.

Examples:
    python vis/plot_bag_level_roc.py \
        --stage1-bags val_res/stage1_control_bags/bag_scores.tsv \
        --stage2-bags val_res/stage2_llp_run2_w9_valid/bag_scores.tsv \
        --output-dir vis_bag_roc

    python vis/plot_bag_level_roc.py \
        --stage2-bags val_res/stage2_llp_run2_w9_valid \
        --stage2-pair 12.5,75 \
        --stage2-pair 25,75 \
        --output-dir vis_bag_roc
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def resolve_bag_scores_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "bag_scores.tsv"
    return require_file(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_bag_scores(path: Path) -> list[dict[str, float | int]]:
    rows = []
    for row in read_tsv(resolve_bag_scores_path(path)):
        rows.append(
            {
                "bag_key": int(row["bag_key"]),
                "target_ratio": float(row["target_ratio"]),
                "target_fraction": float(row["target_fraction"]),
                "bag_score": float(row["bag_score"]),
                "num_reads": int(row["num_reads"]),
            }
        )
    if not rows:
        raise ValueError(f"{path}: no bag score rows found.")
    return rows


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    return plt


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape:
        raise ValueError(f"y_true and scores must share shape, got {y_true.shape} and {scores.shape}")
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError(f"ROC requires at least one positive and one negative, got pos={positives}, neg={negatives}")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_true = y_true[order]
    tps = np.cumsum(sorted_true == 1)
    fps = np.cumsum(sorted_true == 0)

    threshold_idx = np.where(np.diff(sorted_scores))[0]
    threshold_idx = np.r_[threshold_idx, len(sorted_scores) - 1]
    tpr = tps[threshold_idx] / positives
    fpr = fps[threshold_idx] / negatives
    thresholds = sorted_scores[threshold_idx]

    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    thresholds = np.r_[np.inf, thresholds, -np.inf]
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, thresholds, auc


def stage1_binary_arrays(rows: list[dict[str, float | int]]) -> tuple[np.ndarray, np.ndarray]:
    selected_labels = []
    selected_scores = []
    skipped = 0
    for row in rows:
        target = float(row["target_fraction"])
        if np.isclose(target, 0.0):
            selected_labels.append(0)
        elif np.isclose(target, 1.0):
            selected_labels.append(1)
        else:
            skipped += 1
            continue
        selected_scores.append(float(row["bag_score"]))
    if skipped:
        raise ValueError(
            "Stage 1 legal ROC expects only 0% and 100% bags. "
            f"Found and skipped {skipped} non-binary ratio bags."
        )
    return np.asarray(selected_labels, dtype=np.int64), np.asarray(selected_scores, dtype=np.float64)


def parse_pair(value: str) -> tuple[float, float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"Stage 2 pair must be LOW,HIGH, got {value!r}")
    low, high = (float(parts[0]), float(parts[1]))
    if np.isclose(low, high):
        raise ValueError(f"Stage 2 pair ratios must differ, got {value!r}")
    return (min(low, high), max(low, high))


def available_ratios(rows: list[dict[str, float | int]]) -> list[float]:
    return sorted({float(row["target_ratio"]) for row in rows})


def default_stage2_pairs(rows: list[dict[str, float | int]]) -> list[tuple[float, float]]:
    ratios = available_ratios(rows)
    if len(ratios) < 2:
        raise ValueError("Stage 2 diagnostic ROC requires at least two target ratios.")
    return [(ratios[i], ratios[j]) for i in range(len(ratios)) for j in range(i + 1, len(ratios))]


def stage2_pair_arrays(
    rows: list[dict[str, float | int]],
    low_ratio: float,
    high_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    for row in rows:
        ratio = float(row["target_ratio"])
        if np.isclose(ratio, low_ratio):
            labels.append(0)
        elif np.isclose(ratio, high_ratio):
            labels.append(1)
        else:
            continue
        scores.append(float(row["bag_score"]))
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def plot_single_roc(plt, fpr: np.ndarray, tpr: np.ndarray, auc: float, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(fpr, tpr, linewidth=2.2, color="#3b6ea8", label=f"AUC={auc:.4f}")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.1, color="#555555", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_multi_roc(plt, curves: list[dict], title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    cmap = plt.get_cmap("tab10")
    for idx, curve in enumerate(curves):
        label = f"{curve['label']} AUC={curve['auc']:.4f}"
        ax.plot(curve["fpr"], curve["tpr"], linewidth=2.0, color=cmap(idx % 10), label=label)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.1, color="#555555", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def append_curve_rows(rows: list[dict], *, group: str, label: str, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> None:
    for idx, (x, y, threshold) in enumerate(zip(fpr, tpr, thresholds)):
        rows.append(
            {
                "group": group,
                "label": label,
                "point_index": idx,
                "fpr": float(x),
                "tpr": float(y),
                "threshold": None if not np.isfinite(threshold) else float(threshold),
            }
        )


def run(args) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib()

    summary_rows = []
    curve_rows = []
    summary_json = {"stage1": None, "stage2": []}

    if args.stage1_bags is not None:
        stage1_rows = load_bag_scores(args.stage1_bags)
        y_true, scores = stage1_binary_arrays(stage1_rows)
        fpr, tpr, thresholds, auc = roc_curve(y_true, scores)
        pos = int(np.sum(y_true == 1))
        neg = int(np.sum(y_true == 0))
        plot_single_roc(
            plt,
            fpr,
            tpr,
            auc,
            "Stage 1 Legal Bag-Level ROC: 0% vs 100%",
            output_dir / "stage1_legal_bag_roc.png",
        )
        append_curve_rows(curve_rows, group="stage1", label="0_vs_100", fpr=fpr, tpr=tpr, thresholds=thresholds)
        record = {
            "group": "stage1",
            "comparison": "0_vs_100",
            "negative_ratio": 0.0,
            "positive_ratio": 100.0,
            "auc": auc,
            "num_negative_bags": neg,
            "num_positive_bags": pos,
            "num_bags": int(y_true.size),
            "interpretation": "legal_control_roc",
        }
        summary_rows.append(record)
        summary_json["stage1"] = record

    if args.stage2_bags is not None:
        stage2_rows = load_bag_scores(args.stage2_bags)
        pairs = [parse_pair(item) for item in args.stage2_pair] if args.stage2_pair else default_stage2_pairs(stage2_rows)
        curves = []
        for low, high in pairs:
            y_true, scores = stage2_pair_arrays(stage2_rows, low, high)
            fpr, tpr, thresholds, auc = roc_curve(y_true, scores)
            pos = int(np.sum(y_true == 1))
            neg = int(np.sum(y_true == 0))
            label = f"{low:g}_vs_{high:g}"
            curves.append({"label": label.replace("_", " "), "fpr": fpr, "tpr": tpr, "auc": auc})
            append_curve_rows(curve_rows, group="stage2", label=label, fpr=fpr, tpr=tpr, thresholds=thresholds)
            record = {
                "group": "stage2",
                "comparison": label,
                "negative_ratio": low,
                "positive_ratio": high,
                "auc": auc,
                "num_negative_bags": neg,
                "num_positive_bags": pos,
                "num_bags": int(y_true.size),
                "interpretation": "diagnostic_ratio_separation_only",
            }
            summary_rows.append(record)
            summary_json["stage2"].append(record)
        plot_multi_roc(
            plt,
            curves,
            "Stage 2 Diagnostic Bag-Level ROC: Ratio Pair Comparisons",
            output_dir / "stage2_diagnostic_pairwise_bag_roc.png",
        )

    if not summary_rows:
        raise ValueError("Provide --stage1-bags, --stage2-bags, or both.")

    write_tsv(
        output_dir / "roc_summary.tsv",
        summary_rows,
        (
            "group",
            "comparison",
            "negative_ratio",
            "positive_ratio",
            "auc",
            "num_negative_bags",
            "num_positive_bags",
            "num_bags",
            "interpretation",
        ),
    )
    write_tsv(
        output_dir / "roc_curve_points.tsv",
        curve_rows,
        ("group", "label", "point_index", "fpr", "tpr", "threshold"),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--stage1-bags",
        type=Path,
        default=None,
        help="Stage 1 bag_scores.tsv or directory containing bag_scores.tsv. Must contain only 0%/100% bags.",
    )
    parser.add_argument(
        "--stage2-bags",
        type=Path,
        default=None,
        help="Stage 2 bag_scores.tsv or directory containing bag_scores.tsv.",
    )
    parser.add_argument(
        "--stage2-pair",
        action="append",
        default=[],
        help="Diagnostic Stage 2 pair as LOW,HIGH ratio percentages. Can be repeated. Defaults to all pairs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(argparser().parse_args())
