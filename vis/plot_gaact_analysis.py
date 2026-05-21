#!/usr/bin/env python3
"""
Detailed analysis of GAACT's poor performance in LOMO benchmark.

This script creates visualizations to investigate whether GAACT's failure
is due to insufficient data or inherent unlearnability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lomo-eval-root", type=Path, required=True)
    parser.add_argument("--internal-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motifs", default="AGACT,GAACT,GGACA,GGACC,GGACT,TGACT")
    return parser.parse_args(argv)


def plot_data_size_vs_performance(output_dir: Path) -> None:
    """
    Plot: Training data size vs Internal Validation Performance
    Shows whether GAACT's poor performance correlates with data size.
    """
    # Training data sizes (from motif_balance.tsv)
    train_sizes = {
        "AGACT": 9864,
        "GAACT": 316,
        "GGACA": 11670,
        "GGACC": 8484,
        "GGACT": 1060,
        "TGACT": 4076,
    }
    
    # Internal validation performance (from full 6-motif model)
    internal_perf = {
        "AGACT":  {"recall": 0.9834, "spec": 0.9883, "roc": 0.9992},
        "GAACT":  {"recall": 0.8824, "spec": 0.9440, "roc": 0.9772},
        "GGACA":  {"recall": 0.9808, "spec": 0.9820, "roc": 0.9981},
        "GGACC":  {"recall": 0.9808, "spec": 0.9841, "roc": 0.9983},
        "GGACT":  {"recall": 0.9825, "spec": 0.9597, "roc": 0.9960},
        "TGACT":  {"recall": 0.9488, "spec": 0.9432, "roc": 0.9863},
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    motifs = list(train_sizes.keys())
    sizes = [train_sizes[m] for m in motifs]
    recalls = [internal_perf[m]["recall"] for m in motifs]
    specs = [internal_perf[m]["spec"] for m in motifs]
    rocs = [internal_perf[m]["roc"] for m in motifs]
    
    colors = ["#d62728" if m == "GAACT" else "#999999" for m in motifs]
    
    # Recall vs Data Size
    ax = axes[0]
    ax.scatter(sizes, recalls, c=colors, s=150, alpha=0.8, edgecolors="black", linewidth=1.5, zorder=5)
    for i, m in enumerate(motifs):
        ax.annotate(m, (sizes[i], recalls[i]), textcoords="offset points", 
                   xytext=(0, 10), ha="center", fontsize=9, fontweight="bold" if m == "GAACT" else "normal")
    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("Internal Validation Recall", fontsize=12)
    ax.set_title("Recall vs Training Data Size", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.3)
    
    # Specificity vs Data Size
    ax = axes[1]
    ax.scatter(sizes, specs, c=colors, s=150, alpha=0.8, edgecolors="black", linewidth=1.5, zorder=5)
    for i, m in enumerate(motifs):
        ax.annotate(m, (sizes[i], specs[i]), textcoords="offset points", 
                   xytext=(0, 10), ha="center", fontsize=9, fontweight="bold" if m == "GAACT" else "normal")
    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("Internal Validation Specificity", fontsize=12)
    ax.set_title("Specificity vs Training Data Size", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    # ROC-AUC vs Data Size
    ax = axes[2]
    ax.scatter(sizes, rocs, c=colors, s=150, alpha=0.8, edgecolors="black", linewidth=1.5, zorder=5)
    for i, m in enumerate(motifs):
        ax.annotate(m, (sizes[i], rocs[i]), textcoords="offset points", 
                   xytext=(0, 10), ha="center", fontsize=9, fontweight="bold" if m == "GAACT" else "normal")
    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("Internal Validation ROC-AUC", fontsize=12)
    ax.set_title("ROC-AUC vs Training Data Size", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("GAACT Analysis: Is Performance Correlated with Data Size?", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "20_gaact_data_size_vs_perf.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote: {output_dir / '20_gaact_data_size_vs_perf.png'}")


def plot_lomo_gap_comparison(output_dir: Path) -> None:
    """
    Plot: LOMO performance gap for each motif
    Shows which motifs degrade most when held out.
    """
    # Internal validation (full 6-motif model)
    internal = {
        "AGACT":  {"recall": 0.9834, "spec": 0.9883, "roc": 0.9992},
        "GAACT":  {"recall": 0.8824, "spec": 0.9440, "roc": 0.9772},
        "GGACA":  {"recall": 0.9808, "spec": 0.9820, "roc": 0.9981},
        "GGACC":  {"recall": 0.9808, "spec": 0.9841, "roc": 0.9983},
        "GGACT":  {"recall": 0.9825, "spec": 0.9597, "roc": 0.9960},
        "TGACT":  {"recall": 0.9488, "spec": 0.9432, "roc": 0.9863},
    }
    
    # LOMO heldout performance
    lomo = {
        "AGACT":  {"recall": 0.7783, "spec": 0.9810, "roc": 0.9794},
        "GAACT":  {"recall": 0.1913, "spec": 0.9640, "roc": 0.6944},
        "GGACA":  {"recall": 0.9253, "spec": 0.7475, "roc": 0.9389},
        "GGACC":  {"recall": 0.9887, "spec": 0.4424, "roc": 0.9374},
        "GGACT":  {"recall": 0.7145, "spec": 0.9681, "roc": 0.9515},
        "TGACT":  {"recall": 0.2662, "spec": 0.9480, "roc": 0.7784},
    }
    
    motifs = list(internal.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # ROC-AUC gap
    ax = axes[0]
    roc_gaps = [internal[m]["roc"] - lomo[m]["roc"] for m in motifs]
    colors = ["#d62728" if m == "GAACT" else "#ff7f0e" if g > 0.15 else "#2ca02c" for m, g in zip(motifs, roc_gaps)]
    bars = ax.bar(motifs, roc_gaps, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("ROC-AUC Drop (Internal → LOMO)", fontsize=12)
    ax.set_title("ROC-AUC Generalization Gap", fontsize=13, fontweight="bold")
    ax.axhline(y=0.05, color="orange", linestyle="--", alpha=0.5, label="Small gap")
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5, label="Large gap")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, gap in zip(bars, roc_gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
               f"{gap:.3f}", ha="center", fontsize=9, fontweight="bold")
    
    # Recall gap
    ax = axes[1]
    rec_gaps = [internal[m]["recall"] - lomo[m]["recall"] for m in motifs]
    colors = ["#d62728" if m == "GAACT" else "#ff7f0e" if g > 0.2 else "#2ca02c" for m, g in zip(motifs, rec_gaps)]
    bars = ax.bar(motifs, rec_gaps, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Recall Drop (Internal → LOMO)", fontsize=12)
    ax.set_title("Recall Generalization Gap", fontsize=13, fontweight="bold")
    ax.axhline(y=0.2, color="orange", linestyle="--", alpha=0.5, label="Moderate gap")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Severe gap")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, gap in zip(bars, rec_gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f"{gap:.3f}", ha="center", fontsize=9, fontweight="bold")
    
    # Specificity gap
    ax = axes[2]
    spec_gaps = [internal[m]["spec"] - lomo[m]["spec"] for m in motifs]
    colors = ["#d62728" if m == "GAACT" else "#ff7f0e" if g > 0.2 else "#2ca02c" for m, g in zip(motifs, spec_gaps)]
    bars = ax.bar(motifs, spec_gaps, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Specificity Drop (Internal → LOMO)", fontsize=12)
    ax.set_title("Specificity Generalization Gap", fontsize=13, fontweight="bold")
    ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5, label="Moderate gap")
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="Severe gap")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, gap in zip(bars, spec_gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f"{gap:.3f}", ha="center", fontsize=9, fontweight="bold")
    
    fig.suptitle("GAACT Analysis: Generalization Gap by Motif", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "21_gaact_lomo_gap.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote: {output_dir / '21_gaact_lomo_gap.png'}")


def plot_train_vs_test_per_motif(output_dir: Path) -> None:
    """
    Plot: For each motif, compare performance when IN training vs when HELD OUT
    """
    # When motif is IN training: use internal validation from full 6-motif model
    internal = {
        "AGACT":  {"recall": 0.9834, "spec": 0.9883, "roc": 0.9992},
        "GAACT":  {"recall": 0.8824, "spec": 0.9440, "roc": 0.9772},
        "GGACA":  {"recall": 0.9808, "spec": 0.9820, "roc": 0.9981},
        "GGACC":  {"recall": 0.9808, "spec": 0.9841, "roc": 0.9983},
        "GGACT":  {"recall": 0.9825, "spec": 0.9597, "roc": 0.9960},
        "TGACT":  {"recall": 0.9488, "spec": 0.9432, "roc": 0.9863},
    }
    
    # When motif is HELD OUT: use LOMO results
    lomo = {
        "AGACT":  {"recall": 0.7783, "spec": 0.9810, "roc": 0.9794},
        "GAACT":  {"recall": 0.1913, "spec": 0.9640, "roc": 0.6944},
        "GGACA":  {"recall": 0.9253, "spec": 0.7475, "roc": 0.9389},
        "GGACC":  {"recall": 0.9887, "spec": 0.4424, "roc": 0.9374},
        "GGACT":  {"recall": 0.7145, "spec": 0.9681, "roc": 0.9515},
        "TGACT":  {"recall": 0.2662, "spec": 0.9480, "roc": 0.7784},
    }
    
    motifs = list(internal.keys())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(motifs))
    width = 0.35
    
    train_rocs = [internal[m]["roc"] for m in motifs]
    test_rocs = [lomo[m]["roc"] for m in motifs]
    
    bars1 = ax.bar(x - width/2, train_rocs, width, label="In Training (Internal Val)", 
                   color="#4c72b0", alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_rocs, width, label="Held Out (LOMO)", 
                   color="#c44e52", alpha=0.8, edgecolor="black", linewidth=1.5)
    
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Performance: In-Training vs Held-Out for Each Motif", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(motifs)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f"{bar.get_height():.3f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f"{bar.get_height():.3f}", ha="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "22_gaact_train_vs_test.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote: {output_dir / '22_gaact_train_vs_test.png'}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating GAACT analysis figures...")
    plot_data_size_vs_performance(output_dir)
    plot_lomo_gap_comparison(output_dir)
    plot_train_vs_test_per_motif(output_dir)
    
    print(f"\nAll GAACT analysis figures written to: {output_dir}")


if __name__ == "__main__":
    main()
