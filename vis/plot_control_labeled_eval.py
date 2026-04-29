#!/usr/bin/env python3
"""
Visualize Stage 1 vs Stage 2 performance on precisely labeled 0%/100% controls.

Primary inputs are directories produced by validate/evaluate_promote_control.py:
- dataset_metrics.tsv
- summary.json

Optional inputs are mod_site_examples.tsv files produced by
validate/evaluate_train_mod.py. Those files allow probability-distribution plots;
evaluate_promote_control.py currently stores aggregate metrics only.

Example with the current repository layout:
    python vis/plot_control_labeled_eval.py \
        --stage1-control-dir val_res/stage1_control_run1_control_eval \
        --stage2-control-dir val_res/stage2_llp_run1_control_eval \
        --output-dir vis_control_out
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


DEFAULT_STAGE1_CONTROL = Path("val_res/stage1_control_run1_control_eval")
DEFAULT_STAGE2_CONTROL = Path("val_res/stage2_llp_run1_control_eval")
DEFAULT_POSITIVE_RE = r"m6a|modified|mod"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_control_metrics(directory: Path) -> dict[str, dict[str, float | str | None]]:
    require_file(directory / "summary.json")
    read_json(directory / "summary.json")
    rows = read_tsv(require_file(directory / "dataset_metrics.tsv"))
    metrics = {}
    for row in rows:
        parsed: dict[str, float | str | None] = {"name": row["name"]}
        for key, value in row.items():
            if key == "name":
                continue
            parsed[key] = parse_float(value)
        metrics[row["name"]] = parsed
    missing = {"ivt", "full_mod"} - set(metrics)
    if missing:
        raise ValueError(f"{directory}: expected dataset rows named 'ivt' and 'full_mod', missing {sorted(missing)}")
    return metrics


def control_stats(metrics: dict[str, dict[str, float | str | None]]) -> dict[str, float]:
    ivt = metrics["ivt"]
    full = metrics["full_mod"]
    neg = float(ivt["num_sites"])
    pos = float(full["num_sites"])
    ivt_accuracy = float(ivt["accuracy"])
    full_recall = float(full["recall"])

    tn = ivt_accuracy * neg
    fp = neg - tn
    tp = full_recall * pos
    fn = pos - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (pos + neg) if pos + neg else 0.0

    return {
        "ivt_mean_prob": float(ivt["mean_pred_mod_prob"]),
        "ivt_median_prob": float(ivt["median_pred_mod_prob"]),
        "full_mean_prob": float(full["mean_pred_mod_prob"]),
        "full_median_prob": float(full["median_pred_mod_prob"]),
        "gap_mean": float(full["mean_pred_mod_prob"]) - float(ivt["mean_pred_mod_prob"]),
        "gap_median": float(full["median_pred_mod_prob"]) - float(ivt["median_pred_mod_prob"]),
        "tn": tn,
        "fp": fp,
        "tp": tp,
        "fn": fn,
        "fpr": fp / neg if neg else 0.0,
        "fnr": fn / pos if pos else 0.0,
        "specificity": tn / neg if neg else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "num_negative_sites": neg,
        "num_positive_sites": pos,
    }


def infer_positive_probability(row: dict[str, str], positive_re: re.Pattern[str]) -> tuple[int, float] | None:
    true_label = row.get("true_mod_label", "")
    pred_label = row.get("pred_mod_label", "")
    score = parse_float(row.get("score"))
    if score is None:
        return None
    true_is_pos = bool(positive_re.search(str(true_label)))
    pred_is_pos = bool(positive_re.search(str(pred_label)))
    prob = float(score) if pred_is_pos else 1.0 - float(score)
    return int(true_is_pos), min(max(prob, 0.0), 1.0)


def load_site_probabilities(path: Path, positive_label_re: str) -> tuple[np.ndarray, np.ndarray]:
    positive_re = re.compile(positive_label_re, flags=re.IGNORECASE)
    y_true = []
    y_prob = []
    for row in read_tsv(require_file(path)):
        inferred = infer_positive_probability(row, positive_re)
        if inferred is None:
            continue
        label, prob = inferred
        y_true.append(label)
        y_prob.append(prob)
    if not y_true:
        raise ValueError(f"{path}: no usable site probability rows found.")
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float64)


def binary_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-y_prob, kind="mergesort")
    scores = y_prob[order]
    truth = y_true[order]
    positives = max(int(np.sum(truth == 1)), 1)
    negatives = max(int(np.sum(truth == 0)), 1)
    tps = np.cumsum(truth == 1)
    fps = np.cumsum(truth == 0)
    threshold_idx = np.where(np.diff(scores))[0]
    threshold_idx = np.r_[threshold_idx, len(scores) - 1]
    tps = tps[threshold_idx]
    fps = fps[threshold_idx]
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives
    fpr = fps / negatives
    tpr = recall
    return {
        "precision": np.r_[1.0, precision],
        "recall": np.r_[0.0, recall],
        "fpr": np.r_[0.0, fpr, 1.0],
        "tpr": np.r_[0.0, tpr, 1.0],
    }


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


def add_bar_labels(ax, bars, *, fmt="{:.2f}", scale=1.0) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            fmt.format(height / scale),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_probability_separation(plt, stats1: dict[str, float], stats2: dict[str, float], labels: tuple[str, str], output: Path) -> None:
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ivt = [stats1["ivt_mean_prob"] * 100.0, stats2["ivt_mean_prob"] * 100.0]
    full = [stats1["full_mean_prob"] * 100.0, stats2["full_mean_prob"] * 100.0]
    bars1 = ax.bar(x - width / 2, ivt, width, label="mod0 / IVT", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, full, width, label="mod100 / full-mod", color="#dd8452")
    add_bar_labels(ax, bars1, fmt="{:.2f}")
    add_bar_labels(ax, bars2, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean predicted m6A probability (%)")
    ax.set_title("Precise Control Separation: mod0 vs mod100")
    ax.set_ylim(0.0, 108.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "control_precise_probability_separation.png")
    plt.close(fig)


def plot_error_tradeoff(plt, stats1: dict[str, float], stats2: dict[str, float], labels: tuple[str, str], output: Path) -> None:
    metric_keys = ["fpr", "fnr", "precision", "recall", "f1", "accuracy"]
    metric_labels = ["mod0 FPR", "mod100 FNR", "Precision", "Recall", "F1", "Accuracy"]
    x = np.arange(len(metric_keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    values1 = [stats1[key] * 100.0 for key in metric_keys]
    values2 = [stats2[key] * 100.0 for key in metric_keys]
    ax.bar(x - width / 2, values1, width, label=labels[0], color="#3b6ea8")
    ax.bar(x + width / 2, values2, width, label=labels[1], color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Percent (%)")
    ax.set_title("Threshold 0.5 Performance on Precisely Labeled Controls")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "control_precise_error_tradeoff.png")
    plt.close(fig)


def plot_confusion_counts(plt, stats1: dict[str, float], stats2: dict[str, float], labels: tuple[str, str], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9))
    for ax, stats, label in zip(axes, (stats1, stats2), labels):
        negative_total = stats["tn"] + stats["fp"]
        positive_total = stats["tp"] + stats["fn"]
        ax.bar([0], [stats["tn"] / negative_total * 100.0], color="#4c72b0", label="TN")
        ax.bar([0], [stats["fp"] / negative_total * 100.0], bottom=[stats["tn"] / negative_total * 100.0], color="#c44e52", label="FP")
        ax.bar([1], [stats["tp"] / positive_total * 100.0], color="#55a868", label="TP")
        ax.bar([1], [stats["fn"] / positive_total * 100.0], bottom=[stats["tp"] / positive_total * 100.0], color="#dd8452", label="FN")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["mod0", "mod100"])
        ax.set_ylim(0.0, 100.0)
        ax.set_ylabel("Within-class fraction (%)")
        ax.set_title(label)
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Control Confusion Composition at Threshold 0.5", y=0.995)
    fig.tight_layout()
    fig.savefig(output / "control_precise_confusion_composition.png")
    plt.close(fig)


def plot_stage_delta(plt, stats1: dict[str, float], stats2: dict[str, float], labels: tuple[str, str], output: Path) -> None:
    deltas = {
        "mod0 mean prob": (stats2["ivt_mean_prob"] - stats1["ivt_mean_prob"]) * 100.0,
        "mod100 mean prob": (stats2["full_mean_prob"] - stats1["full_mean_prob"]) * 100.0,
        "mean gap": (stats2["gap_mean"] - stats1["gap_mean"]) * 100.0,
        "mod0 FPR": (stats2["fpr"] - stats1["fpr"]) * 100.0,
        "mod100 FNR": (stats2["fnr"] - stats1["fnr"]) * 100.0,
        "F1": (stats2["f1"] - stats1["f1"]) * 100.0,
    }
    names = list(deltas)
    values = [deltas[name] for name in names]
    colors = ["#55a868" if value >= 0 else "#c44e52" for value in values]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.bar(np.arange(len(names)), values, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel(f"{labels[1]} - {labels[0]} (percentage points)")
    ax.set_title("Stage 2 Change on Precisely Labeled Controls")
    fig.tight_layout()
    fig.savefig(output / "control_precise_stage_delta.png")
    plt.close(fig)


def plot_optional_site_distributions(
    plt,
    stage1_sites: tuple[np.ndarray, np.ndarray] | None,
    stage2_sites: tuple[np.ndarray, np.ndarray] | None,
    labels: tuple[str, str],
    output: Path,
) -> None:
    if stage1_sites is None and stage2_sites is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=True)
    for ax, site_data, label in zip(axes, (stage1_sites, stage2_sites), labels):
        if site_data is None:
            ax.text(0.5, 0.5, "mod_site_examples.tsv not provided", ha="center", va="center")
            ax.set_axis_off()
            continue
        y_true, y_prob = site_data
        neg = y_prob[y_true == 0]
        pos = y_prob[y_true == 1]
        if neg.size:
            ax.hist(neg, bins=40, alpha=0.65, color="#4c72b0", label="true mod0")
        if pos.size:
            ax.hist(pos, bins=40, alpha=0.65, color="#dd8452", label="true mod100")
        ax.axvline(0.5, linestyle="--", color="#333333", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("Inferred m6A probability")
        ax.set_ylabel("A-sites")
        ax.legend(frameon=False)
    fig.suptitle("A-site Probability Distributions from mod_site_examples.tsv", y=0.995)
    fig.tight_layout()
    fig.savefig(output / "control_precise_site_probability_hist.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    for ax, site_data, label in zip(axes, (stage1_sites, stage2_sites), labels):
        if site_data is None or len(np.unique(site_data[0])) < 2:
            ax.text(0.5, 0.5, "Need both classes for ROC/PR", ha="center", va="center")
            ax.set_axis_off()
            continue
        y_true, y_prob = site_data
        curves = binary_curve_points(y_true, y_prob)
        ax.plot(curves["fpr"], curves["tpr"], label="ROC", color="#4c72b0")
        ax.plot(curves["recall"], curves["precision"], label="PR", color="#dd8452")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle=":", color="#555555", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("FPR / Recall")
        ax.set_ylabel("TPR / Precision")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False)
    fig.suptitle("A-site ROC and PR Curves from mod_site_examples.tsv", y=0.995)
    fig.tight_layout()
    fig.savefig(output / "control_precise_site_roc_pr.png")
    plt.close(fig)


def write_summary(output: Path, stats1: dict[str, float], stats2: dict[str, float], labels: tuple[str, str]) -> None:
    fieldnames = [
        "model",
        "ivt_mean_prob",
        "ivt_median_prob",
        "full_mean_prob",
        "full_median_prob",
        "gap_mean",
        "fpr",
        "fnr",
        "specificity",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "num_negative_sites",
        "num_positive_sites",
    ]
    with (output / "control_precise_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerow({"model": labels[0], **stats1})
        writer.writerow({"model": labels[1], **stats2})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Stage 1 vs Stage 2 on precise mod0/mod100 control labels.")
    parser.add_argument("--stage1-control-dir", type=Path, default=DEFAULT_STAGE1_CONTROL)
    parser.add_argument("--stage2-control-dir", type=Path, default=DEFAULT_STAGE2_CONTROL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-label", default="Stage 1 control")
    parser.add_argument("--stage2-label", default="Stage 2 LLP")
    parser.add_argument("--stage1-sites-tsv", type=Path, default=None, help="Optional evaluate_train_mod mod_site_examples.tsv for Stage 1.")
    parser.add_argument("--stage2-sites-tsv", type=Path, default=None, help="Optional evaluate_train_mod mod_site_examples.tsv for Stage 2.")
    parser.add_argument("--positive-label-re", default=DEFAULT_POSITIVE_RE, help="Regex used to identify modified labels in mod_site_examples.tsv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    labels = (args.stage1_label, args.stage2_label)

    stage1_metrics = load_control_metrics(args.stage1_control_dir)
    stage2_metrics = load_control_metrics(args.stage2_control_dir)
    stats1 = control_stats(stage1_metrics)
    stats2 = control_stats(stage2_metrics)

    stage1_sites = load_site_probabilities(args.stage1_sites_tsv, args.positive_label_re) if args.stage1_sites_tsv else None
    stage2_sites = load_site_probabilities(args.stage2_sites_tsv, args.positive_label_re) if args.stage2_sites_tsv else None

    plt = configure_matplotlib()
    plot_probability_separation(plt, stats1, stats2, labels, output)
    plot_error_tradeoff(plt, stats1, stats2, labels, output)
    plot_confusion_counts(plt, stats1, stats2, labels, output)
    plot_stage_delta(plt, stats1, stats2, labels, output)
    plot_optional_site_distributions(plt, stage1_sites, stage2_sites, labels, output)
    write_summary(output, stats1, stats2, labels)

    print(f"Wrote precise-control visualizations to: {output.resolve()}")


if __name__ == "__main__":
    main()
