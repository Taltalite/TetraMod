#!/usr/bin/env python3
"""
Visualize LOMO (Leave-One-Motif-Out) benchmark results.

This script creates comprehensive visualizations comparing:
1. Per-motif LOMO performance (ROC-AUC, recall, specificity, BCE)
2. Internal validation vs LOMO heldout gap
3. Training curves for all 6 LOMO models
4. Per-run breakdown for each held-out motif
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_MOTIFS = ["AGACT", "GAACT", "GGACA", "GGACC", "GGACT", "TGACT"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lomo-summary", type=Path, required=True, help="Path to lomo_summary.tsv")
    parser.add_argument("--lomo-eval-root", type=Path, required=True, help="Root of LOMO eval results")
    parser.add_argument("--model-root", type=Path, required=True, help="Root of LOMO trained models")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write PNG figures")
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    return parser.parse_args(argv)


def load_lomo_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Convert numeric columns
    numeric_cols = ["roc_auc", "pr_auc", "recall", "specificity", "bce", "balanced_accuracy", 
                    "num_sites", "num_positive", "num_negative"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_training_curves(model_root: Path, motifs: list[str]) -> pd.DataFrame:
    rows = []
    for motif in motifs:
        training_csv = model_root / f"leave_{motif}" / "training.csv"
        if not training_csv.exists():
            continue
        df = pd.read_csv(training_csv)
        df["heldout_motif"] = motif
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_lomo_performance(summary: pd.DataFrame, out: Path, motifs: list[str]) -> None:
    """Plot 1: LOMO per-motif performance bars"""
    summary["heldout_motif"] = pd.Categorical(summary["heldout_motif"], categories=motifs, ordered=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC-AUC
    ax = axes[0, 0]
    sns.barplot(data=summary, x="heldout_motif", y="roc_auc", palette="viridis", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("LOMO: ROC-AUC by Held-out Motif")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("ROC-AUC")
    ax.axhline(y=0.9, color="r", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.axhline(y=0.8, color="orange", linestyle="--", alpha=0.5, label="0.8 threshold")
    ax.legend()
    for i, row in summary.iterrows():
        ax.text(i, row["roc_auc"] + 0.02, f"{row['roc_auc']:.3f}", ha="center", fontsize=9)
    
    # Recall vs Specificity
    ax = axes[0, 1]
    x = np.arange(len(motifs))
    width = 0.35
    recall_vals = [summary[summary["heldout_motif"] == m]["recall"].values[0] for m in motifs]
    spec_vals = [summary[summary["heldout_motif"] == m]["specificity"].values[0] for m in motifs]
    ax.bar(x - width/2, recall_vals, width, label="Recall", color="#c44e52", alpha=0.8)
    ax.bar(x + width/2, spec_vals, width, label="Specificity", color="#4c72b0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(motifs)
    ax.set_ylim(0, 1.05)
    ax.set_title("LOMO: Recall vs Specificity by Held-out Motif")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    
    # BCE
    ax = axes[1, 0]
    sns.barplot(data=summary, x="heldout_motif", y="bce", palette="rocket", ax=ax)
    ax.set_title("LOMO: BCE (Binary Cross-Entropy) by Held-out Motif")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("BCE")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.legend()
    for i, row in summary.iterrows():
        ax.text(i, row["bce"] + 0.03, f"{row['bce']:.3f}", ha="center", fontsize=9)
    
    # Balanced Accuracy
    ax = axes[1, 1]
    sns.barplot(data=summary, x="heldout_motif", y="balanced_accuracy", palette="coolwarm", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("LOMO: Balanced Accuracy by Held-out Motif")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("Balanced Accuracy")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Random")
    ax.legend()
    for i, row in summary.iterrows():
        ax.text(i, row["balanced_accuracy"] + 0.02, f"{row['balanced_accuracy']:.3f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out / "01_lomo_performance_overview.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out / '01_lomo_performance_overview.png'}")


def plot_internal_vs_lomo_gap(summary: pd.DataFrame, out: Path, motifs: list[str]) -> None:
    """Plot 2: Gap between internal validation and LOMO"""
    # Internal validation values (from mAFiA-only 6motif epoch4)
    internal = {
        "AGACT": {"roc_auc": 0.9992, "recall": 0.9834, "specificity": 0.9883, "bce": 0.0350},
        "GAACT": {"roc_auc": 0.9772, "recall": 0.8824, "specificity": 0.9440, "bce": 0.1380},
        "GGACA": {"roc_auc": 0.9981, "recall": 0.9808, "specificity": 0.9820, "bce": 0.0563},
        "GGACC": {"roc_auc": 0.9983, "recall": 0.9808, "specificity": 0.9841, "bce": 0.0441},
        "GGACT": {"roc_auc": 0.9960, "recall": 0.9825, "specificity": 0.9597, "bce": 0.1030},
        "TGACT": {"roc_auc": 0.9863, "recall": 0.9488, "specificity": 0.9432, "bce": 0.1526},
    }
    
    rows = []
    for motif in motifs:
        lomo_row = summary[summary["heldout_motif"] == motif].iloc[0]
        int_row = internal[motif]
        rows.append({
            "motif": motif,
            "internal_roc": int_row["roc_auc"],
            "lomo_roc": lomo_row["roc_auc"],
            "roc_gap": int_row["roc_auc"] - lomo_row["roc_auc"],
            "internal_recall": int_row["recall"],
            "lomo_recall": lomo_row["recall"],
            "recall_gap": int_row["recall"] - lomo_row["recall"],
            "internal_spec": int_row["specificity"],
            "lomo_spec": lomo_row["specificity"],
            "spec_gap": int_row["specificity"] - lomo_row["specificity"],
        })
    
    gap_df = pd.DataFrame(rows)
    gap_df["motif"] = pd.Categorical(gap_df["motif"], categories=motifs, ordered=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # ROC-AUC gap
    ax = axes[0]
    colors = ["#d62728" if g > 0.15 else "#ff7f0e" if g > 0.05 else "#2ca02c" for g in gap_df["roc_gap"]]
    sns.barplot(data=gap_df, x="motif", y="roc_gap", palette=colors, ax=ax)
    ax.set_title("ROC-AUC Gap: Internal → LOMO")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("ROC-AUC Drop")
    ax.axhline(y=0.05, color="orange", linestyle="--", alpha=0.5)
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5)
    for i, row in gap_df.iterrows():
        ax.text(i, row["roc_gap"] + 0.005, f"{row['roc_gap']:.3f}", ha="center", fontsize=9)
    
    # Recall gap
    ax = axes[1]
    colors = ["#d62728" if g > 0.5 else "#ff7f0e" if g > 0.2 else "#2ca02c" for g in gap_df["recall_gap"]]
    sns.barplot(data=gap_df, x="motif", y="recall_gap", palette=colors, ax=ax)
    ax.set_title("Recall Gap: Internal → LOMO")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("Recall Drop")
    ax.axhline(y=0.2, color="orange", linestyle="--", alpha=0.5)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    for i, row in gap_df.iterrows():
        ax.text(i, row["recall_gap"] + 0.01, f"{row['recall_gap']:.3f}", ha="center", fontsize=9)
    
    # Spec gap
    ax = axes[2]
    colors = ["#d62728" if g > 0.3 else "#ff7f0e" if g > 0.1 else "#2ca02c" for g in gap_df["spec_gap"]]
    sns.barplot(data=gap_df, x="motif", y="spec_gap", palette=colors, ax=ax)
    ax.set_title("Specificity Gap: Internal → LOMO")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("Specificity Drop")
    ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5)
    for i, row in gap_df.iterrows():
        ax.text(i, row["spec_gap"] + 0.01, f"{row['spec_gap']:.3f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out / "02_internal_vs_lomo_gap.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out / '02_internal_vs_lomo_gap.png'}")


def plot_training_curves(training_df: pd.DataFrame, out: Path, motifs: list[str]) -> None:
    """Plot 3: Training curves for all 6 LOMO models"""
    if training_df.empty:
        print("No training curves found, skipping.")
        return
    
    training_df["heldout_motif"] = pd.Categorical(training_df["heldout_motif"], categories=motifs, ordered=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Train mod loss
    ax = axes[0, 0]
    for motif in motifs:
        sub = training_df[training_df["heldout_motif"] == motif]
        if not sub.empty:
            ax.plot(sub["epoch"], sub["train_mod_loss"], marker="o", label=motif, linewidth=2)
    ax.set_title("Training Mod Loss by LOMO Model")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Mod Loss")
    ax.legend(title="Held-out Motif")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    # Val mod loss
    ax = axes[0, 1]
    for motif in motifs:
        sub = training_df[training_df["heldout_motif"] == motif]
        if not sub.empty:
            ax.plot(sub["epoch"], sub["val_mod_loss"], marker="s", label=motif, linewidth=2)
    ax.set_title("Validation Mod Loss by LOMO Model")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Mod Loss")
    ax.legend(title="Held-out Motif")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    # Train vs Val comparison for each model
    ax = axes[1, 0]
    for motif in motifs:
        sub = training_df[training_df["heldout_motif"] == motif]
        if not sub.empty:
            ax.plot(sub["epoch"], sub["train_mod_loss"] - sub["val_mod_loss"], 
                   marker="o", label=motif, linewidth=2, alpha=0.7)
    ax.set_title("Train-Val Gap (Mod Loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss - Val Loss")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.legend(title="Held-out Motif")
    ax.grid(True, alpha=0.3)
    
    # Final epoch comparison
    ax = axes[1, 1]
    final_epoch = training_df.groupby("heldout_motif").last().reset_index()
    x = np.arange(len(motifs))
    width = 0.35
    train_vals = [final_epoch[final_epoch["heldout_motif"] == m]["train_mod_loss"].values[0] for m in motifs]
    val_vals = [final_epoch[final_epoch["heldout_motif"] == m]["val_mod_loss"].values[0] for m in motifs]
    ax.bar(x - width/2, train_vals, width, label="Train", color="#4c72b0", alpha=0.8)
    ax.bar(x + width/2, val_vals, width, label="Val", color="#c44e52", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(motifs)
    ax.set_title("Final Epoch Mod Loss")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("Mod Loss")
    ax.legend()
    ax.set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(out / "03_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out / '03_training_curves.png'}")


def plot_heatmap(summary: pd.DataFrame, out: Path, motifs: list[str]) -> None:
    """Plot 4: Heatmap of all metrics"""
    summary["heldout_motif"] = pd.Categorical(summary["heldout_motif"], categories=motifs, ordered=True)
    summary = summary.sort_values("heldout_motif")
    
    metrics = ["roc_auc", "pr_auc", "recall", "specificity", "balanced_accuracy", "bce"]
    matrix = summary.set_index("heldout_motif")[metrics].T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Normalize BCE to 0-1 range for visualization (lower is better)
    matrix_norm = matrix.copy()
    if "bce" in matrix_norm.index:
        max_bce = matrix_norm.loc["bce"].max()
        if max_bce > 0:
            matrix_norm.loc["bce"] = 1 - (matrix_norm.loc["bce"] / max_bce)  # Invert so higher is better
    
    sns.heatmap(matrix_norm, annot=matrix, fmt=".3f", cmap="RdYlGn", 
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Score (higher is better, BCE inverted)"})
    ax.set_title("LOMO Benchmark: All Metrics Heatmap")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("Metric")
    
    plt.tight_layout()
    plt.savefig(out / "04_metrics_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out / '04_metrics_heatmap.png'}")


def plot_radar_chart(summary: pd.DataFrame, out: Path, motifs: list[str]) -> None:
    """Plot 5: Radar chart comparing motifs"""
    from math import pi
    
    categories = ["ROC-AUC", "Recall", "Specificity", "Balanced Acc"]
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(motifs)))
    
    for idx, motif in enumerate(motifs):
        row = summary[summary["heldout_motif"] == motif].iloc[0]
        values = [
            row["roc_auc"],
            row["recall"],
            row["specificity"],
            row["balanced_accuracy"],
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=motif, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title("LOMO: Motif Performance Radar", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(out / "05_radar_chart.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out / '05_radar_chart.png'}")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    
    motifs = [m.strip() for m in args.motifs.split(",") if m.strip()]
    
    # Load data
    summary = load_lomo_summary(args.lomo_summary)
    training_df = load_training_curves(args.model_root, motifs)
    
    # Generate plots
    plot_lomo_performance(summary, out, motifs)
    plot_internal_vs_lomo_gap(summary, out, motifs)
    plot_training_curves(training_df, out, motifs)
    plot_heatmap(summary, out, motifs)
    plot_radar_chart(summary, out, motifs)
    
    # Save summary
    summary_dict = {
        "motifs": motifs,
        "figures": sorted([p.name for p in out.glob("*.png")]),
        "lomo_results": summary.to_dict("records"),
    }
    (out / "visual_summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    
    print(f"\nAll figures written to: {out}")


if __name__ == "__main__":
    main()
