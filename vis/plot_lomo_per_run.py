#!/usr/bin/env python3
"""
Per-run breakdown visualization for LOMO benchmark.
Shows how each held-out motif performs on each heldout run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_MOTIFS = ["AGACT", "GAACT", "GGACA", "GGACC", "GGACT", "TGACT"]
DEFAULT_RUNS = ["Mix_1_A_RTA", "Mix_2_m6A_RTA", "Mix_3_A_RTA", "Mix_4_m6A_RTA"]

RUN_DISPLAY = {
    "Mix_1_A_RTA": "Mix1 A\n(unmod)",
    "Mix_2_m6A_RTA": "Mix2 m6A\n(mod)",
    "Mix_3_A_RTA": "Mix3 A\n(unmod)",
    "Mix_4_m6A_RTA": "Mix4 m6A\n(mod)",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lomo-eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    return parser.parse_args(argv)


def load_per_run_data(eval_root: Path, motifs: list[str], runs: list[str]) -> pd.DataFrame:
    rows = []
    for motif in motifs:
        for run in runs:
            gm_path = eval_root / f"leave_{motif}" / f"heldout_{run}" / "group_metrics.tsv"
            if not gm_path.exists():
                continue
            with gm_path.open() as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    if row.get("group_by") == "motif_context" and row.get("motif_context") == motif:
                        def to_float(v):
                            try:
                                return float(v) if v and v != "NA" else np.nan
                            except:
                                return np.nan
                        
                        rows.append({
                            "heldout_motif": motif,
                            "run": run,
                            "num_sites": to_float(row.get("num_sites")),
                            "num_positive": to_float(row.get("num_positive")),
                            "num_negative": to_float(row.get("num_negative")),
                            "recall": to_float(row.get("recall")),
                            "specificity": to_float(row.get("specificity")),
                            "bce": to_float(row.get("bce")),
                            "roc_auc": to_float(row.get("roc_auc")),
                            "mean_prob": to_float(row.get("mean_prob")),
                        })
                        break
    return pd.DataFrame(rows)


def plot_per_run_recall(df: pd.DataFrame, out: Path, motifs: list[str], runs: list[str]) -> None:
    """Plot recall for modified runs only"""
    mod_runs = [r for r in runs if "m6A" in r]
    sub = df[df["run"].isin(mod_runs)].copy()
    
    if sub.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(motifs))
    width = 0.35
    
    for i, run in enumerate(mod_runs):
        run_data = [sub[(sub["heldout_motif"] == m) & (sub["run"] == run)]["recall"].values 
                    for m in motifs]
        run_vals = [v[0] if len(v) > 0 else np.nan for v in run_data]
        offset = (i - len(mod_runs)/2 + 0.5) * width
        ax.bar(x + offset, run_vals, width, label=RUN_DISPLAY.get(run, run), alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(motifs)
    ax.set_ylim(0, 1.05)
    ax.set_title("LOMO: Recall on Modified Heldout Runs by Held-out Motif")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("Recall")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.legend()
    
    for i, motif in enumerate(motifs):
        vals = [sub[(sub["heldout_motif"] == motif) & (sub["run"] == r)]["recall"].values 
                for r in mod_runs]
        for j, v in enumerate(vals):
            if len(v) > 0 and not np.isnan(v[0]):
                offset = (j - len(mod_runs)/2 + 0.5) * width
                ax.text(i + offset, v[0] + 0.02, f"{v[0]:.3f}", ha="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out / "06_per_run_recall.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out / '06_per_run_recall.png'}")


def plot_per_run_specificity(df: pd.DataFrame, out: Path, motifs: list[str], runs: list[str]) -> None:
    """Plot specificity for unmodified runs only"""
    unmod_runs = [r for r in runs if "A_RTA" in r]
    sub = df[df["run"].isin(unmod_runs)].copy()
    
    if sub.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(motifs))
    width = 0.35
    
    for i, run in enumerate(unmod_runs):
        run_data = [sub[(sub["heldout_motif"] == m) & (sub["run"] == run)]["specificity"].values 
                    for m in motifs]
        run_vals = [v[0] if len(v) > 0 else np.nan for v in run_data]
        offset = (i - len(unmod_runs)/2 + 0.5) * width
        ax.bar(x + offset, run_vals, width, label=RUN_DISPLAY.get(run, run), alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(motifs)
    ax.set_ylim(0, 1.05)
    ax.set_title("LOMO: Specificity on Unmodified Heldout Runs by Held-out Motif")
    ax.set_xlabel("Held-out Motif")
    ax.set_ylabel("Specificity")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.legend()
    
    for i, motif in enumerate(motifs):
        vals = [sub[(sub["heldout_motif"] == motif) & (sub["run"] == r)]["specificity"].values 
                for r in unmod_runs]
        for j, v in enumerate(vals):
            if len(v) > 0 and not np.isnan(v[0]):
                offset = (j - len(unmod_runs)/2 + 0.5) * width
                ax.text(i + offset, v[0] + 0.02, f"{v[0]:.3f}", ha="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out / "07_per_run_specificity.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out / '07_per_run_specificity.png'}")


def plot_mean_probability(df: pd.DataFrame, out: Path, motifs: list[str], runs: list[str]) -> None:
    """Plot mean predicted probability for each motif-run combination"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Modified runs
    mod_runs = [r for r in runs if "m6A" in r]
    ax = axes[0]
    sub = df[df["run"].isin(mod_runs)].copy()
    pivot = sub.pivot(index="heldout_motif", columns="run", values="mean_prob")
    pivot = pivot.reindex(motifs)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Mean Predicted Probability\n(Modified Runs)")
    ax.set_xlabel("Heldout Run")
    ax.set_ylabel("Held-out Motif")
    
    # Unmodified runs
    unmod_runs = [r for r in runs if "A_RTA" in r]
    ax = axes[1]
    sub = df[df["run"].isin(unmod_runs)].copy()
    pivot = sub.pivot(index="heldout_motif", columns="run", values="mean_prob")
    pivot = pivot.reindex(motifs)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Mean Predicted Probability\n(Unmodified Runs - lower is better)")
    ax.set_xlabel("Heldout Run")
    ax.set_ylabel("")
    
    plt.tight_layout()
    plt.savefig(out / "08_mean_probability_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out / '08_mean_probability_heatmap.png'}")


def plot_bce_breakdown(df: pd.DataFrame, out: Path, motifs: list[str], runs: list[str]) -> None:
    """Plot BCE breakdown by run"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot = df.pivot(index="heldout_motif", columns="run", values="bce")
    pivot = pivot.reindex(motifs)
    pivot = pivot.reindex(columns=runs)
    
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="rocket_r", ax=ax, linewidths=0.5)
    ax.set_title("BCE (Binary Cross-Entropy) by Held-out Motif and Run")
    ax.set_xlabel("Heldout Run")
    ax.set_ylabel("Held-out Motif")
    
    plt.tight_layout()
    plt.savefig(out / "09_bce_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out / '09_bce_breakdown.png'}")


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
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    
    df = load_per_run_data(args.lomo_eval_root, motifs, runs)
    
    if df.empty:
        print("No data loaded!")
        return
    
    plot_per_run_recall(df, out, motifs, runs)
    plot_per_run_specificity(df, out, motifs, runs)
    plot_mean_probability(df, out, motifs, runs)
    plot_bce_breakdown(df, out, motifs, runs)
    
    print(f"\nAll per-run figures written to: {out}")


if __name__ == "__main__":
    main()
