#!/usr/bin/env python3
"""
Visualize Stage 1 vs Stage 2 promoted evaluation results.

Inputs are the TSV/JSON files emitted by:
- validate/evaluate_llp_bags.py
- validate/evaluate_promote_control.py

Example:
    python vis/plot_eval_results.py \
        --val-res val_res \
        --output-dir vis_out
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_STAGE1_LLP = "stage1_control_run1_evaluate_baseline_bags"
DEFAULT_STAGE2_LLP = "stage2_llp_run1_evaluate_llp_bags"
DEFAULT_STAGE1_CONTROL = "stage1_control_run1_control_eval"
DEFAULT_STAGE2_CONTROL = "stage2_llp_run1_control_eval"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_bag_scores(directory: Path) -> list[dict[str, float]]:
    path = require_file(directory / "bag_scores.tsv")
    rows = []
    for row in read_tsv(path):
        rows.append(
            {
                "bag_key": int(row["bag_key"]),
                "target_ratio": float(row["target_ratio"]),
                "target_fraction": float(row["target_fraction"]),
                "bag_score": float(row["bag_score"]),
                "num_reads": int(row["num_reads"]),
            }
        )
    return rows


def load_control_metrics(directory: Path) -> dict[str, dict[str, float | str | None]]:
    path = require_file(directory / "dataset_metrics.tsv")
    metrics = {}
    for row in read_tsv(path):
        name = row["name"]
        parsed: dict[str, float | str | None] = {"name": name}
        for key, value in row.items():
            if key == "name":
                continue
            parsed[key] = parse_float(value)
        metrics[name] = parsed
    return metrics


def bag_arrays(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratios = np.asarray([row["target_ratio"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target_fraction"] for row in rows], dtype=np.float64)
    scores = np.asarray([row["bag_score"] for row in rows], dtype=np.float64)
    return ratios, targets, scores


def ratio_stats(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    ratios, targets, scores = bag_arrays(rows)
    stats = []
    for ratio in sorted(np.unique(ratios)):
        mask = ratios == ratio
        ratio_scores = scores[mask]
        ratio_targets = targets[mask]
        errors = ratio_scores - ratio_targets
        stats.append(
            {
                "ratio": float(ratio),
                "target_percent": float(np.mean(ratio_targets) * 100.0),
                "mean_score_percent": float(np.mean(ratio_scores) * 100.0),
                "median_score_percent": float(np.median(ratio_scores) * 100.0),
                "p10_score_percent": float(np.percentile(ratio_scores, 10) * 100.0),
                "p90_score_percent": float(np.percentile(ratio_scores, 90) * 100.0),
                "mae_percent": float(np.mean(np.abs(errors)) * 100.0),
                "rmse_percent": float(np.sqrt(np.mean(errors**2)) * 100.0),
                "bias_percent": float(np.mean(errors) * 100.0),
                "num_bags": int(np.sum(mask)),
            }
        )
    return stats


def overall_bag_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    _, targets, scores = bag_arrays(rows)
    errors = scores - targets
    corr = np.corrcoef(scores, targets)[0, 1] if scores.size > 1 else np.nan
    return {
        "mae_percent": float(np.mean(np.abs(errors)) * 100.0),
        "rmse_percent": float(np.sqrt(np.mean(errors**2)) * 100.0),
        "bias_percent": float(np.mean(errors) * 100.0),
        "corr": float(corr),
        "mean_score_percent": float(np.mean(scores) * 100.0),
        "mean_target_percent": float(np.mean(targets) * 100.0),
    }


def paired_bag_delta(
    stage1_rows: list[dict[str, float]],
    stage2_rows: list[dict[str, float]],
) -> dict[str, np.ndarray]:
    first = {int(row["bag_key"]): row for row in stage1_rows}
    second = {int(row["bag_key"]): row for row in stage2_rows}
    keys = sorted(set(first) & set(second))
    targets = np.asarray([first[key]["target_fraction"] for key in keys], dtype=np.float64)
    ratios = np.asarray([first[key]["target_ratio"] for key in keys], dtype=np.float64)
    score1 = np.asarray([first[key]["bag_score"] for key in keys], dtype=np.float64)
    score2 = np.asarray([second[key]["bag_score"] for key in keys], dtype=np.float64)
    abs_err1 = np.abs(score1 - targets)
    abs_err2 = np.abs(score2 - targets)
    return {
        "keys": np.asarray(keys, dtype=np.int64),
        "ratios": ratios,
        "targets": targets,
        "score1": score1,
        "score2": score2,
        "score_delta": score2 - score1,
        "abs_error_delta": abs_err2 - abs_err1,
    }


def combined_control_stats(metrics: dict[str, dict[str, float | str | None]]) -> dict[str, float]:
    ivt = metrics["ivt"]
    full = metrics["full_mod"]
    neg = float(ivt["num_sites"])
    pos = float(full["num_sites"])
    ivt_acc = float(ivt["accuracy"])
    full_recall = float(full["recall"])
    tn = ivt_acc * neg
    fp = neg - tn
    tp = full_recall * pos
    fn = pos - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "ivt_mean_prob": float(ivt["mean_pred_mod_prob"]),
        "full_mean_prob": float(full["mean_pred_mod_prob"]),
        "gap": float(full["mean_pred_mod_prob"]) - float(ivt["mean_pred_mod_prob"]),
        "ivt_fpr": fp / neg if neg else 0.0,
        "full_fnr": fn / pos if pos else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / (pos + neg) if pos + neg else 0.0,
        "num_negative_sites": neg,
        "num_positive_sites": pos,
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


def plot_llp_calibration(plt, stage1_rows, stage2_rows, output: Path, labels: tuple[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = ["#3b6ea8", "#c44e52"]
    for rows, label, color in zip((stage1_rows, stage2_rows), labels, colors):
        stats = ratio_stats(rows)
        x = np.asarray([item["target_percent"] for item in stats])
        y = np.asarray([item["mean_score_percent"] for item in stats])
        p10 = np.asarray([item["p10_score_percent"] for item in stats])
        p90 = np.asarray([item["p90_score_percent"] for item in stats])
        yerr = np.vstack([y - p10, p90 - y])
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.0, capsize=4, color=color, label=label)

    all_targets = sorted({row["target_ratio"] for row in stage1_rows + stage2_rows})
    low, high = min(all_targets), max(all_targets)
    ax.plot([low, high], [low, high], linestyle="--", color="#333333", linewidth=1.2, label="ideal")
    ax.set_xlabel("Known bag ratio (%)")
    ax.set_ylabel("Predicted bag score (%)")
    ax.set_title("LLP Bag-Level Calibration")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "llp_ratio_calibration.png")
    plt.close(fig)


def plot_llp_errors(plt, stage1_rows, stage2_rows, output: Path, labels: tuple[str, str]) -> None:
    stats1 = ratio_stats(stage1_rows)
    stats2 = ratio_stats(stage2_rows)
    ratios = np.asarray([item["ratio"] for item in stats1])
    width = 0.36
    x = np.arange(len(ratios))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    for ax, key, title, ylabel in (
        (axes[0], "mae_percent", "Mean Absolute Error by Ratio", "MAE (percentage points)"),
        (axes[1], "bias_percent", "Signed Bias by Ratio", "Bias (percentage points)"),
    ):
        values1 = [item[key] for item in stats1]
        values2 = [item[key] for item in stats2]
        ax.bar(x - width / 2, values1, width, label=labels[0], color="#3b6ea8")
        ax.bar(x + width / 2, values2, width, label=labels[1], color="#c44e52")
        ax.axhline(0.0, color="#333333", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{ratio:g}" for ratio in ratios])
        ax.set_xlabel("Known bag ratio (%)")
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output / "llp_error_by_ratio.png")
    plt.close(fig)


def plot_llp_distributions(plt, stage1_rows, stage2_rows, output: Path, labels: tuple[str, str]) -> None:
    ratios1, _, scores1 = bag_arrays(stage1_rows)
    ratios2, _, scores2 = bag_arrays(stage2_rows)
    ratios = sorted(np.unique(np.concatenate([ratios1, ratios2])))
    data1 = [scores1[ratios1 == ratio] * 100.0 for ratio in ratios]
    data2 = [scores2[ratios2 == ratio] * 100.0 for ratio in ratios]
    positions1 = np.arange(len(ratios)) * 3.0
    positions2 = positions1 + 0.9

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    bp1 = ax.boxplot(data1, positions=positions1, widths=0.7, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data2, positions=positions2, widths=0.7, patch_artist=True, showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#8fb4dc")
        patch.set_edgecolor("#3b6ea8")
    for patch in bp2["boxes"]:
        patch.set_facecolor("#e1a0a3")
        patch.set_edgecolor("#c44e52")
    for ratio, pos in zip(ratios, positions1 + 0.45):
        ax.hlines(ratio, pos - 1.0, pos + 1.0, colors="#333333", linestyles="--", linewidth=1.0)

    ax.set_xticks(positions1 + 0.45)
    ax.set_xticklabels([f"{ratio:g}" for ratio in ratios])
    ax.set_xlabel("Known bag ratio (%)")
    ax.set_ylabel("Predicted bag score (%)")
    ax.set_title("LLP Bag Score Distributions")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], labels, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "llp_bag_score_distributions.png")
    plt.close(fig)


def plot_paired_delta(plt, stage1_rows, stage2_rows, output: Path, labels: tuple[str, str]) -> None:
    paired = paired_bag_delta(stage1_rows, stage2_rows)
    ratios = sorted(np.unique(paired["ratios"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].hist(paired["score_delta"] * 100.0, bins=50, color="#8172b2", alpha=0.85)
    axes[0].axvline(0.0, color="#333333", linewidth=1.0)
    axes[0].set_title(f"Paired Bag Score Shift: {labels[1]} - {labels[0]}")
    axes[0].set_xlabel("Score shift (percentage points)")
    axes[0].set_ylabel("Number of bags")

    data = [paired["abs_error_delta"][paired["ratios"] == ratio] * 100.0 for ratio in ratios]
    bp = axes[1].boxplot(data, labels=[f"{ratio:g}" for ratio in ratios], showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#b7c9a8")
        patch.set_edgecolor("#4d7f3f")
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_title("Paired Absolute Error Change")
    axes[1].set_xlabel("Known bag ratio (%)")
    axes[1].set_ylabel("Stage 2 - Stage 1 absolute error (percentage points)")
    fig.tight_layout()
    fig.savefig(output / "llp_paired_bag_delta.png")
    plt.close(fig)


def plot_control_probabilities(plt, stage1_control, stage2_control, output: Path, labels: tuple[str, str]) -> None:
    stats = [combined_control_stats(stage1_control), combined_control_stats(stage2_control)]
    x = np.arange(2)
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ivt = [item["ivt_mean_prob"] * 100.0 for item in stats]
    full = [item["full_mean_prob"] * 100.0 for item in stats]
    ax.bar(x - width / 2, ivt, width, label="0% IVT mean", color="#4c72b0")
    ax.bar(x + width / 2, full, width, label="100% full-mod mean", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean predicted m6A probability (%)")
    ax.set_title("Control Retention: 0% vs 100% Mean Probabilities")
    ax.set_ylim(0.0, 105.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "control_mean_probabilities.png")
    plt.close(fig)


def plot_control_metrics(plt, stage1_control, stage2_control, output: Path, labels: tuple[str, str]) -> None:
    stats = [combined_control_stats(stage1_control), combined_control_stats(stage2_control)]
    metric_keys = ["ivt_fpr", "full_fnr", "precision", "recall", "f1"]
    metric_labels = ["IVT FPR", "Full FNR", "Precision", "Recall", "F1"]
    x = np.arange(len(metric_keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    values1 = [stats[0][key] * 100.0 for key in metric_keys]
    values2 = [stats[1][key] * 100.0 for key in metric_keys]
    ax.bar(x - width / 2, values1, width, label=labels[0], color="#3b6ea8")
    ax.bar(x + width / 2, values2, width, label=labels[1], color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Percent (%)")
    ax.set_title("Control Retention Metrics at Threshold 0.5")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "control_threshold_metrics.png")
    plt.close(fig)


def plot_summary_dashboard(plt, stage1_rows, stage2_rows, stage1_control, stage2_control, output: Path, labels: tuple[str, str]) -> None:
    bag_stats = [overall_bag_stats(stage1_rows), overall_bag_stats(stage2_rows)]
    control_stats = [combined_control_stats(stage1_control), combined_control_stats(stage2_control)]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    x = np.arange(2)
    colors = ["#3b6ea8", "#c44e52"]

    axes[0, 0].bar(x, [item["mae_percent"] for item in bag_stats], color=colors)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_title("Overall LLP Bag MAE")
    axes[0, 0].set_ylabel("Percentage points")

    axes[0, 1].bar(x, [item["bias_percent"] for item in bag_stats], color=colors)
    axes[0, 1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_title("Overall LLP Bag Bias")
    axes[0, 1].set_ylabel("Percentage points")

    axes[1, 0].bar(x, [item["ivt_fpr"] * 100.0 for item in control_stats], color=colors)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_title("0% IVT False Positive Rate")
    axes[1, 0].set_ylabel("Percent")

    axes[1, 1].bar(x, [item["f1"] * 100.0 for item in control_stats], color=colors)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_title("Combined Control F1")
    axes[1, 1].set_ylabel("Percent")
    axes[1, 1].set_ylim(85.0, 100.5)

    fig.suptitle("Promoted Training Evaluation Summary", y=0.995)
    fig.tight_layout()
    fig.savefig(output / "evaluation_summary_dashboard.png")
    plt.close(fig)


def write_summary_tables(output: Path, stage1_rows, stage2_rows, stage1_control, stage2_control, labels: tuple[str, str]) -> None:
    bag_summary_path = output / "llp_overall_summary.tsv"
    with bag_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "mae_percent", "rmse_percent", "bias_percent", "corr", "mean_score_percent", "mean_target_percent"],
            delimiter="\t",
        )
        writer.writeheader()
        for label, rows in zip(labels, (stage1_rows, stage2_rows)):
            writer.writerow({"model": label, **overall_bag_stats(rows)})

    ratio_summary_path = output / "llp_ratio_summary.tsv"
    with ratio_summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model",
            "ratio",
            "num_bags",
            "mean_score_percent",
            "median_score_percent",
            "p10_score_percent",
            "p90_score_percent",
            "mae_percent",
            "rmse_percent",
            "bias_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for label, rows in zip(labels, (stage1_rows, stage2_rows)):
            for item in ratio_stats(rows):
                writer.writerow({"model": label, **item})

    control_summary_path = output / "control_summary.tsv"
    with control_summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["model", "ivt_mean_prob", "full_mean_prob", "gap", "ivt_fpr", "full_fnr", "precision", "recall", "f1", "accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for label, metrics in zip(labels, (stage1_control, stage2_control)):
            writer.writerow({"model": label, **combined_control_stats(metrics)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot promoted Stage 1 vs Stage 2 evaluation results.")
    parser.add_argument("--val-res", type=Path, default=Path("val_res"), help="Directory containing evaluation result subdirectories.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where PNG files and summary TSVs are written.")
    parser.add_argument("--stage1-llp-dir", type=Path, default=None, help="Directory containing Stage 1 bag_scores.tsv.")
    parser.add_argument("--stage2-llp-dir", type=Path, default=None, help="Directory containing Stage 2 bag_scores.tsv.")
    parser.add_argument("--stage1-control-dir", type=Path, default=None, help="Directory containing Stage 1 dataset_metrics.tsv.")
    parser.add_argument("--stage2-control-dir", type=Path, default=None, help="Directory containing Stage 2 dataset_metrics.tsv.")
    parser.add_argument("--stage1-label", default="Stage 1 control", help="Display label for Stage 1.")
    parser.add_argument("--stage2-label", default="Stage 2 LLP", help="Display label for Stage 2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    stage1_llp_dir = args.stage1_llp_dir or args.val_res / DEFAULT_STAGE1_LLP
    stage2_llp_dir = args.stage2_llp_dir or args.val_res / DEFAULT_STAGE2_LLP
    stage1_control_dir = args.stage1_control_dir or args.val_res / DEFAULT_STAGE1_CONTROL
    stage2_control_dir = args.stage2_control_dir or args.val_res / DEFAULT_STAGE2_CONTROL

    # Force early validation of expected result files.
    require_file(stage1_llp_dir / "summary.json")
    require_file(stage2_llp_dir / "summary.json")
    require_file(stage1_control_dir / "summary.json")
    require_file(stage2_control_dir / "summary.json")
    load_json(stage1_llp_dir / "summary.json")
    load_json(stage2_llp_dir / "summary.json")
    load_json(stage1_control_dir / "summary.json")
    load_json(stage2_control_dir / "summary.json")

    stage1_rows = load_bag_scores(stage1_llp_dir)
    stage2_rows = load_bag_scores(stage2_llp_dir)
    stage1_control = load_control_metrics(stage1_control_dir)
    stage2_control = load_control_metrics(stage2_control_dir)
    labels = (args.stage1_label, args.stage2_label)

    plt = configure_matplotlib()
    plot_summary_dashboard(plt, stage1_rows, stage2_rows, stage1_control, stage2_control, output, labels)
    plot_llp_calibration(plt, stage1_rows, stage2_rows, output, labels)
    plot_llp_errors(plt, stage1_rows, stage2_rows, output, labels)
    plot_llp_distributions(plt, stage1_rows, stage2_rows, output, labels)
    plot_paired_delta(plt, stage1_rows, stage2_rows, output, labels)
    plot_control_probabilities(plt, stage1_control, stage2_control, output, labels)
    plot_control_metrics(plt, stage1_control, stage2_control, output, labels)
    write_summary_tables(output, stage1_rows, stage2_rows, stage1_control, stage2_control, labels)

    print(f"Wrote evaluation visualizations to: {output.resolve()}")


if __name__ == "__main__":
    main()
