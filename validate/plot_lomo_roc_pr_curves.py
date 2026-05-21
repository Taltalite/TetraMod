#!/usr/bin/env python3
"""
Plot ROC and PR curves for LOMO (Leave-One-Motif-Out) benchmark results.

For each held-out motif model, generates one figure with two subplots:
- Left: ROC curves for all 6 motifs
- Right: PR curves for all 6 motifs

The held-out motif is highlighted in blue, while the 5 training motifs are in gray.

Usage:
    python validate/plot_lomo_roc_pr_curves.py \
        --lomo-eval-root val_res/lomo_stage1_mafia_6motif \
        --output-dir val_res/lomo_stage1_mafia_6motif/summary_epoch6/figures \
        --motifs AGACT,GAACT,GGACA,GGACC,GGACT,TGACT
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


# Define which runs contain which motifs
# Mix_1/2: AGACT, GAACT (oligo heldout)
# Mix_3/4: GGACA, GGACC, GGACT, TGACT (run heldout)
MOTIF_TO_RUNS = {
    "AGACT":   {"neg": "Mix_1_A_RTA",   "pos": "Mix_2_m6A_RTA"},
    "GAACT":   {"neg": "Mix_1_A_RTA",   "pos": "Mix_2_m6A_RTA"},
    "GGACA":   {"neg": "Mix_3_A_RTA",   "pos": "Mix_4_m6A_RTA"},
    "GGACC":   {"neg": "Mix_3_A_RTA",   "pos": "Mix_4_m6A_RTA"},
    "GGACT":   {"neg": "Mix_3_A_RTA",   "pos": "Mix_4_m6A_RTA"},
    "TGACT":   {"neg": "Mix_3_A_RTA",   "pos": "Mix_4_m6A_RTA"},
}

DEFAULT_MOTIFS = ["AGACT", "GAACT", "GGACA", "GGACC", "GGACT", "TGACT"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC and PR curves for LOMO benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lomo-eval-root",
        type=Path,
        required=True,
        help="Root directory of LOMO evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write output figures",
    )
    parser.add_argument(
        "--motifs",
        default=",".join(DEFAULT_MOTIFS),
        help="Comma-separated list of motifs",
    )
    parser.add_argument(
        "--heldout-label",
        default="Held-out",
        help="Label for held-out motif in legend",
    )
    parser.add_argument(
        "--train-label",
        default="Train",
        help="Label for training motifs in legend",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI",
    )
    return parser.parse_args(argv)


def load_motif_data(eval_root: Path, model_motif: str, target_motif: str) -> pd.DataFrame | None:
    """
    Load site predictions for a specific target motif from a specific LOMO model.
    
    Combines negative samples (unmodified run) and positive samples (modified run)
    for the target motif.
    
    Args:
        eval_root: Root of LOMO evaluation results
        model_motif: The held-out motif for the LOMO model
        target_motif: The motif to extract data for
        
    Returns:
        DataFrame with columns [target, prob_m6a] or None if no data
    """
    runs = MOTIF_TO_RUNS.get(target_motif)
    if runs is None:
        return None
    
    neg_run = runs["neg"]
    pos_run = runs["pos"]
    
    # Load negative samples
    neg_path = eval_root / f"leave_{model_motif}" / f"heldout_{neg_run}" / "site_predictions.tsv"
    pos_path = eval_root / f"leave_{model_motif}" / f"heldout_{pos_run}" / "site_predictions.tsv"
    
    parts = []
    
    if neg_path.exists():
        neg_df = pd.read_csv(neg_path, sep="\t")
        neg_df = neg_df[neg_df["motif_context"] == target_motif].copy()
        if len(neg_df) > 0:
            parts.append(neg_df[["target", "prob_m6a"]])
    
    if pos_path.exists():
        pos_df = pd.read_csv(pos_path, sep="\t")
        pos_df = pos_df[pos_df["motif_context"] == target_motif].copy()
        if len(pos_df) > 0:
            parts.append(pos_df[["target", "prob_m6a"]])
    
    if not parts:
        return None
    
    combined = pd.concat(parts, ignore_index=True)
    return combined


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Compute ROC curve (fpr, tpr, auc). Returns None if only one class present."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    
    pos = int(np.count_nonzero(y_true == 1))
    neg = int(np.count_nonzero(y_true == 0))
    
    if pos == 0 or neg == 0:
        return None
    
    # Sort by score descending
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    
    # Find distinct score thresholds
    distinct = np.r_[np.where(np.diff(score_sorted))[0], len(score_sorted) - 1]
    
    tp = np.cumsum(y_sorted == 1)[distinct]
    fp = np.cumsum(y_sorted == 0)[distinct]
    
    tpr = np.r_[0.0, tp / pos, 1.0]
    fpr = np.r_[0.0, fp / neg, 1.0]
    
    # Compute AUC using trapezoid rule
    auc = float(np.trapezoid(tpr, fpr))
    
    return fpr, tpr, auc


def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Compute PR curve (recall, precision, auc). Returns None if no positives."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    
    pos = int(np.count_nonzero(y_true == 1))
    neg = int(np.count_nonzero(y_true == 0))
    
    if pos == 0:
        return None
    
    # Sort by score descending
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    
    # Find distinct score thresholds
    distinct = np.r_[np.where(np.diff(score_sorted))[0], len(score_sorted) - 1]
    
    tp = np.cumsum(y_sorted == 1)[distinct]
    fp = np.cumsum(y_sorted == 0)[distinct]
    
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    
    # Compute AUC using trapezoid rule
    auc = float(np.trapezoid(precision, recall))
    
    return recall, precision, auc


def plot_lomo_roc_pr(
    eval_root: Path,
    output_dir: Path,
    heldout_motif: str,
    all_motifs: list[str],
    heldout_label: str = "Held-out",
    train_label: str = "Train",
    dpi: int = 180,
) -> Path:
    """
    Generate one figure for a held-out motif model showing ROC and PR curves
    for all 6 motifs.
    
    Args:
        eval_root: Root of LOMO evaluation results
        output_dir: Directory to write figure
        heldout_motif: The held-out motif for this model
        all_motifs: List of all motifs
        heldout_label: Label for held-out motif
        train_label: Label for training motifs
        dpi: Figure DPI
        
    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax_roc = axes[0]
    ax_pr = axes[1]
    
    # Plot each motif
    for motif in all_motifs:
        data = load_motif_data(eval_root, heldout_motif, motif)
        if data is None or len(data) == 0:
            print(f"  Warning: No data for model=leave_{heldout_motif}, motif={motif}")
            continue
        
        y_true = data["target"].values
        y_score = data["prob_m6a"].values
        
        # Check if we have both classes
        n_pos = int(np.count_nonzero(y_true == 1))
        n_neg = int(np.count_nonzero(y_true == 0))
        
        if n_pos == 0 or n_neg == 0:
            print(f"  Warning: Only one class for model=leave_{heldout_motif}, motif={motif} (pos={n_pos}, neg={n_neg})")
            continue
        
        is_heldout = (motif == heldout_motif)
        
        # Color and style
        if is_heldout:
            color = "#1f77b4"  # Blue
            alpha = 1.0
            linewidth = 2.5
            linestyle = "-"
            zorder = 10
        else:
            color = "#999999"  # Gray
            alpha = 0.6
            linewidth = 1.5
            linestyle = "--"
            zorder = 1
        
        # Compute and plot ROC
        roc_result = compute_roc_curve(y_true, y_score)
        if roc_result is not None:
            fpr, tpr, roc_auc = roc_result
            label = f"{motif}"
            if is_heldout:
                label += f" ({heldout_label}, AUC={roc_auc:.3f})"
            else:
                label += f" ({train_label}, AUC={roc_auc:.3f})"
            
            ax_roc.plot(
                fpr, tpr,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
                label=label,
                zorder=zorder,
            )
        
        # Compute and plot PR
        pr_result = compute_pr_curve(y_true, y_score)
        if pr_result is not None:
            recall, precision, pr_auc = pr_result
            label = f"{motif}"
            if is_heldout:
                label += f" ({heldout_label}, AUC={pr_auc:.3f})"
            else:
                label += f" ({train_label}, AUC={pr_auc:.3f})"
            
            ax_pr.plot(
                recall, precision,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
                label=label,
                zorder=zorder,
            )
    
    # ROC subplot formatting
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title(f"ROC Curves - Model: leave_{heldout_motif}", fontsize=13, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_aspect("equal")
    
    # PR subplot formatting
    # Compute baseline (prevalence) for reference
    ax_pr.axhline(y=0.5, color="k", linestyle="--", alpha=0.3, linewidth=1, label="Baseline (0.5)")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.set_title(f"PR Curves - Model: leave_{heldout_motif}", fontsize=13, fontweight="bold")
    ax_pr.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_aspect("equal")
    
    # Overall figure title
    train_motifs = [m for m in all_motifs if m != heldout_motif]
    fig.suptitle(
        f"LOMO Model: {heldout_motif} held out | Train: {', '.join(train_motifs)}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"10_roc_pr_leave_{heldout_motif}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return output_path


def main() -> None:
    args = parse_args()
    
    motifs = [m.strip() for m in args.motifs.split(",") if m.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating ROC/PR curves for {len(motifs)} LOMO models...")
    print(f"Motifs: {', '.join(motifs)}")
    print(f"Output directory: {output_dir}")
    print()
    
    generated = []
    for heldout_motif in motifs:
        print(f"Processing model: leave_{heldout_motif}...")
        path = plot_lomo_roc_pr(
            eval_root=args.lomo_eval_root,
            output_dir=output_dir,
            heldout_motif=heldout_motif,
            all_motifs=motifs,
            heldout_label=args.heldout_label,
            train_label=args.train_label,
            dpi=args.dpi,
        )
        generated.append(str(path))
        print(f"  Saved: {path.name}")
    
    print()
    print(f"Generated {len(generated)} figures:")
    for p in generated:
        print(f"  {p}")
    
    # Save metadata
    metadata = {
        "motifs": motifs,
        "figures": sorted([Path(p).name for p in generated]),
        "output_dir": str(output_dir),
    }
    metadata_path = output_dir / "roc_pr_curves_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nMetadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
