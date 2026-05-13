#!/usr/bin/env python3
"""
Visualize mAFiA Stage 1 motif coverage, held-out performance, and training curves.

The script consumes existing local reports:

* dataset_check_res/.../motif_balance.tsv
* val_res/mafia_stage1_epoch4/{summary.json,group_metrics.tsv}
* val_res/mafia_stage1_epoch4_heldout_*/{summary.json,group_metrics.tsv}

It writes PNG figures only.  It never calls plt.show().
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_MOTIF_ORDER = ["AGACT", "GAACT", "GGACA", "GGACC", "GGACT", "TGACT"]
METRIC_COLUMNS = [
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
]


def clean_run_name(directory: Path, prefixes: Iterable[str]) -> str:
    name = directory.name
    for prefix in prefixes:
        if prefix and name.startswith(prefix):
            return name[len(prefix):]
    return name


def resolve_glob(pattern: str) -> list[Path]:
    return [Path(path) for path in sorted(glob.glob(str(pattern)))]


def to_float(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if not value or value.upper() == "NA":
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_group_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    for col in METRIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(to_float)
    return df


def motif_order(*frames: pd.DataFrame) -> list[str]:
    motifs = set(DEFAULT_MOTIF_ORDER)
    for frame in frames:
        if "motif_context" in frame.columns:
            motifs.update(str(v) for v in frame["motif_context"].dropna().unique() if str(v))
    return [m for m in DEFAULT_MOTIF_ORDER if m in motifs] + sorted(motifs.difference(DEFAULT_MOTIF_ORDER))


def load_internal_metrics(internal_eval_dir: Path) -> tuple[dict, pd.DataFrame]:
    with (internal_eval_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    groups = read_group_metrics(internal_eval_dir / "group_metrics.tsv")
    motifs = groups[groups["group_by"] == "motif_context"].copy()
    motifs["eval_set"] = "Internal validation"
    return summary, motifs


def confusion_from_single_class(row: pd.Series) -> dict[str, float]:
    n_pos = int(row["num_positive"])
    n_neg = int(row["num_negative"])
    n = n_pos + n_neg
    pred_pos = to_float(row.get("predicted_positive_rate")) * n
    pred_pos = 0.0 if np.isnan(pred_pos) else pred_pos
    if n_pos > 0 and n_neg == 0:
        tp = pred_pos
        fn = n_pos - tp
        fp = tn = 0.0
    elif n_neg > 0 and n_pos == 0:
        fp = pred_pos
        tn = n_neg - fp
        tp = fn = 0.0
    else:
        # These heldout runs are single-class.  If that ever changes, use only
        # rates that are directly reported and leave confusion-derived metrics.
        tp = fn = fp = tn = np.nan
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "num_positive": float(n_pos),
        "num_negative": float(n_neg),
        "num_sites": float(n),
        "weighted_bce": to_float(row.get("bce")) * n,
        "weighted_mean_prob": to_float(row.get("mean_prob")) * n,
    }


def load_heldout_metrics(
    heldout_dirs: Iterable[Path],
    *,
    heldout_label: str,
    heldout_prefixes: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    motif_parts = []
    run_parts = []
    for directory in sorted(heldout_dirs):
        if not (directory / "group_metrics.tsv").exists():
            continue
        df = read_group_metrics(directory / "group_metrics.tsv")
        run_name = clean_run_name(directory, heldout_prefixes)
        motif_rows = df[df["group_by"] == "motif_context"].copy()
        motif_rows["heldout_run"] = run_name
        motif_parts.append(motif_rows)

        run_row = df[df["group_by"] == "run_id"].copy()
        if not run_row.empty:
            run_row["heldout_run"] = run_name
            run_parts.append(run_row)

    if not motif_parts:
        raise FileNotFoundError("No heldout group_metrics.tsv files found.")

    raw_motifs = pd.concat(motif_parts, ignore_index=True)
    raw_runs = pd.concat(run_parts, ignore_index=True) if run_parts else pd.DataFrame()

    agg_rows = []
    for motif, sub in raw_motifs.groupby("motif_context", sort=True):
        totals = {"tp": 0.0, "fn": 0.0, "fp": 0.0, "tn": 0.0, "num_positive": 0.0, "num_negative": 0.0, "num_sites": 0.0}
        weighted_bce = 0.0
        weighted_mean_prob = 0.0
        for _, row in sub.iterrows():
            c = confusion_from_single_class(row)
            for key in totals:
                totals[key] += c[key]
            weighted_bce += c["weighted_bce"]
            weighted_mean_prob += c["weighted_mean_prob"]
        pos = totals["num_positive"]
        neg = totals["num_negative"]
        sites = totals["num_sites"]
        recall = totals["tp"] / pos if pos else np.nan
        specificity = totals["tn"] / neg if neg else np.nan
        accuracy = (totals["tp"] + totals["tn"]) / sites if sites else np.nan
        balanced = np.nanmean([recall, specificity])
        agg_rows.append(
            {
                "motif_context": motif,
                "num_sites": sites,
                "num_positive": pos,
                "num_negative": neg,
                "recall": recall,
                "specificity": specificity,
                "balanced_accuracy": balanced,
                "accuracy": accuracy,
                "bce": weighted_bce / sites if sites else np.nan,
                "mean_prob": weighted_mean_prob / sites if sites else np.nan,
                "eval_set": heldout_label,
            }
        )
    motif_agg = pd.DataFrame(agg_rows)

    overall_totals = {"tp": 0.0, "fn": 0.0, "fp": 0.0, "tn": 0.0, "num_positive": 0.0, "num_negative": 0.0, "num_sites": 0.0}
    weighted_bce = 0.0
    weighted_mean_prob = 0.0
    for _, row in raw_runs.iterrows():
        c = confusion_from_single_class(row)
        for key in overall_totals:
            overall_totals[key] += c[key]
        weighted_bce += c["weighted_bce"]
        weighted_mean_prob += c["weighted_mean_prob"]
    pos = overall_totals["num_positive"]
    neg = overall_totals["num_negative"]
    sites = overall_totals["num_sites"]
    overall = {
        "num_sites": sites,
        "num_positive": pos,
        "num_negative": neg,
        "recall": overall_totals["tp"] / pos if pos else np.nan,
        "specificity": overall_totals["tn"] / neg if neg else np.nan,
        "balanced_accuracy": np.nanmean([
            overall_totals["tp"] / pos if pos else np.nan,
            overall_totals["tn"] / neg if neg else np.nan,
        ]),
        "accuracy": (overall_totals["tp"] + overall_totals["tn"]) / sites if sites else np.nan,
        "bce": weighted_bce / sites if sites else np.nan,
        "mean_prob": weighted_mean_prob / sites if sites else np.nan,
    }
    return motif_agg, raw_runs, overall


def load_site_predictions(eval_dir: Path, *, eval_set: str, run_name: str | None = None) -> pd.DataFrame:
    path = eval_dir / "site_predictions.tsv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    required = {"target", "prob_m6a", "motif_context"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    df["target"] = df["target"].astype(int)
    df["prob_m6a"] = df["prob_m6a"].astype(float)
    df["motif_context"] = df["motif_context"].astype(str)
    df["eval_set"] = eval_set
    if run_name is not None:
        df["heldout_run"] = run_name
    return df


def load_heldout_site_predictions(
    heldout_dirs: Iterable[Path],
    *,
    heldout_label: str,
    heldout_prefixes: Iterable[str],
) -> pd.DataFrame:
    parts = []
    for directory in sorted(heldout_dirs):
        run_name = clean_run_name(directory, heldout_prefixes)
        df = load_site_predictions(directory, eval_set=heldout_label, run_name=run_name)
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def roc_curve_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = int(np.count_nonzero(y_true == 1))
    neg = int(np.count_nonzero(y_true == 0))
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    distinct = np.r_[np.where(np.diff(score_sorted))[0], len(score_sorted) - 1]
    tp = np.cumsum(y_sorted == 1)[distinct]
    fp = np.cumsum(y_sorted == 0)[distinct]
    tpr = np.r_[0.0, tp / pos, 1.0]
    fpr = np.r_[0.0, fp / neg, 1.0]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def pr_curve_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = int(np.count_nonzero(y_true == 1))
    neg = int(np.count_nonzero(y_true == 0))
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    distinct = np.r_[np.where(np.diff(score_sorted))[0], len(score_sorted) - 1]
    tp = np.cumsum(y_sorted == 1)[distinct]
    fp = np.cumsum(y_sorted == 0)[distinct]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    auc = float(np.trapezoid(precision, recall))
    return recall, precision, auc


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def annotate_bars(ax, fmt="{:.0f}", yoffset=2, rotation=0):
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            if np.isnan(value):
                labels.append("")
            elif abs(value) <= 1.0:
                labels.append(f"{value:.2f}")
            else:
                labels.append(fmt.format(value))
        ax.bar_label(container, labels=labels, padding=yoffset, fontsize=8, rotation=rotation)


def plot_dataset_counts(balance: pd.DataFrame, out: Path, motifs: list[str]) -> None:
    long = balance.melt(
        id_vars=["split", "motif_context"],
        value_vars=["positive", "negative"],
        var_name="class",
        value_name="samples",
    )
    long["motif_context"] = pd.Categorical(long["motif_context"], categories=motifs, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=False)
    palette = {"positive": "#c44e52", "negative": "#4c72b0"}
    for ax, split in zip(axes, ["train", "validation"]):
        subset = long[long["split"] == split]
        sns.barplot(data=subset, x="motif_context", y="samples", hue="class", palette=palette, ax=ax)
        ax.set_title(f"{split} motif/class counts")
        ax.set_xlabel("DRACH motif")
        ax.set_ylabel("samples")
        ax.tick_params(axis="x", rotation=30)
        annotate_bars(ax)
        ax.legend(title="Label")
    fig.suptitle("Stage1 merged dataset motif/class counts", y=1.04)
    savefig(out / "01_dataset_motif_counts.png")


def plot_coverage_heatmap(
    balance: pd.DataFrame,
    heldout: pd.DataFrame,
    out: Path,
    motifs: list[str],
    *,
    internal_label: str,
    heldout_label: str,
) -> None:
    rows = []
    for _, row in balance.iterrows():
        split = "Train" if row["split"] == "train" else internal_label
        rows.append({"set": f"{split} positive", "motif_context": row["motif_context"], "count": row["positive"]})
        rows.append({"set": f"{split} negative", "motif_context": row["motif_context"], "count": row["negative"]})
    for _, row in heldout.iterrows():
        rows.append({"set": f"{heldout_label} positive", "motif_context": row["motif_context"], "count": row["num_positive"]})
        rows.append({"set": f"{heldout_label} negative", "motif_context": row["motif_context"], "count": row["num_negative"]})
    matrix = pd.DataFrame(rows).pivot_table(index="set", columns="motif_context", values="count", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(
        [
            "Train positive",
            "Train negative",
            f"{internal_label} positive",
            f"{internal_label} negative",
            f"{heldout_label} positive",
            f"{heldout_label} negative",
        ]
    )
    matrix = matrix.reindex(columns=motifs, fill_value=0)

    plt.figure(figsize=(12.5, 4.7))
    sns.heatmap(
        np.log10(matrix + 1),
        annot=matrix.astype(int),
        fmt="d",
        cmap="viridis",
        cbar_kws={"label": "log10(count + 1)"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Motif coverage matrix across train, validation, and held-out sets")
    plt.xlabel("DRACH motif")
    plt.ylabel("")
    savefig(out / "02_motif_coverage_heatmap.png")


def plot_internal_metrics(internal: pd.DataFrame, out: Path, motifs: list[str], *, internal_label: str) -> None:
    metrics = ["recall", "specificity", "balanced_accuracy", "pr_auc", "bce"]
    data = internal[["motif_context", *metrics, "num_sites", "num_positive", "num_negative"]].copy()
    data["motif_context"] = pd.Categorical(data["motif_context"], categories=motifs, ordered=True)
    long = data.melt(id_vars=["motif_context"], value_vars=metrics, var_name="metric", value_name="value")

    plt.figure(figsize=(12.5, 5.2))
    ax = sns.barplot(data=long, x="motif_context", y="value", hue="metric")
    ax.set_ylim(0, max(1.05, float(np.nanmax(long["value"])) * 1.1))
    ax.set_title(f"{internal_label}: per-motif threshold and ranking metrics")
    ax.set_xlabel("DRACH motif")
    ax.set_ylabel("metric value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Metric", ncols=3)
    savefig(out / "03_internal_validation_per_motif_metrics.png")

    plt.figure(figsize=(11.5, 4.8))
    prob_long = data.melt(
        id_vars=["motif_context"],
        value_vars=["num_positive", "num_negative"],
        var_name="label",
        value_name="sites",
    )
    ax = sns.barplot(data=prob_long, x="motif_context", y="sites", hue="label", palette=["#c44e52", "#4c72b0"])
    ax.set_title(f"{internal_label}: site counts by motif")
    ax.set_xlabel("DRACH motif")
    ax.set_ylabel("labeled sites")
    ax.tick_params(axis="x", rotation=30)
    annotate_bars(ax)
    savefig(out / "04_internal_validation_label_counts.png")


def plot_heldout_metrics(heldout: pd.DataFrame, out: Path, motifs: list[str], *, heldout_label: str) -> None:
    data = heldout.copy()
    data["motif_context"] = pd.Categorical(data["motif_context"], categories=motifs, ordered=True)
    metric_long = data.melt(
        id_vars=["motif_context"],
        value_vars=["recall", "specificity", "balanced_accuracy", "bce"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(12.5, 5.2))
    ax = sns.barplot(data=metric_long, x="motif_context", y="value", hue="metric")
    ax.set_title(f"{heldout_label}: per-motif performance")
    ax.set_xlabel("DRACH motif")
    ax.set_ylabel("metric value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Metric", ncols=2)
    savefig(out / "05_heldout_per_motif_metrics.png")

    count_long = data.melt(
        id_vars=["motif_context"],
        value_vars=["num_positive", "num_negative"],
        var_name="label",
        value_name="sites",
    )
    plt.figure(figsize=(11.5, 4.8))
    ax = sns.barplot(data=count_long, x="motif_context", y="sites", hue="label", palette=["#c44e52", "#4c72b0"])
    ax.set_title(f"{heldout_label}: site counts by motif")
    ax.set_xlabel("DRACH motif")
    ax.set_ylabel("labeled sites")
    ax.tick_params(axis="x", rotation=30)
    annotate_bars(ax)
    savefig(out / "06_heldout_label_counts.png")


def plot_overall_summary(
    internal_summary: dict,
    heldout_summary: dict,
    out: Path,
    *,
    internal_label: str,
    heldout_label: str,
) -> None:
    rows = [
        {
            "eval_set": internal_label,
            "num_sites": internal_summary["overall"]["num_sites"],
            "num_positive": internal_summary["overall"]["num_positive"],
            "num_negative": internal_summary["overall"]["num_negative"],
            "accuracy": internal_summary["overall"]["accuracy"],
            "balanced_accuracy": internal_summary["overall"]["balanced_accuracy"],
            "recall": internal_summary["overall"]["recall"],
            "specificity": internal_summary["overall"]["specificity"],
            "bce": internal_summary["overall"]["bce"],
        },
        {
            "eval_set": heldout_label,
            **heldout_summary,
        },
    ]
    df = pd.DataFrame(rows)
    long = df.melt(
        id_vars=["eval_set"],
        value_vars=["accuracy", "balanced_accuracy", "recall", "specificity", "bce"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10.5, 5.0))
    ax = sns.barplot(data=long, x="metric", y="value", hue="eval_set", palette=["#55a868", "#dd8452"])
    ax.set_title("Overall validation vs held-out performance")
    ax.set_xlabel("")
    ax.set_ylabel("metric value")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")
    annotate_bars(ax, yoffset=1)
    savefig(out / "07_overall_validation_vs_heldout.png")


def plot_seen_unseen_gap(balance: pd.DataFrame, internal: pd.DataFrame, heldout: pd.DataFrame, out: Path, motifs: list[str]) -> None:
    train_counts = balance[balance["split"] == "train"].copy()
    train_counts["train_total"] = train_counts["positive"] + train_counts["negative"]
    train_total_by_motif = dict(zip(train_counts["motif_context"], train_counts["train_total"]))
    rows = []
    for _, row in heldout.iterrows():
        motif = row["motif_context"]
        rows.append(
            {
                "motif_context": motif,
                "train_total": train_total_by_motif.get(motif, 0),
                "seen_in_train": "seen" if train_total_by_motif.get(motif, 0) > 0 else "unseen",
                "heldout_recall": row["recall"],
                "heldout_specificity": row["specificity"],
                "heldout_bce": row["bce"],
            }
        )
    gap = pd.DataFrame(rows)
    gap["motif_context"] = pd.Categorical(gap["motif_context"], categories=motifs, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sns.barplot(data=gap, x="motif_context", y="train_total", hue="seen_in_train", ax=axes[0], dodge=False)
    axes[0].set_title("Training support for heldout motifs")
    axes[0].set_xlabel("DRACH motif")
    axes[0].set_ylabel("train samples")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="")
    annotate_bars(axes[0])

    perf = gap.melt(
        id_vars=["motif_context", "seen_in_train"],
        value_vars=["heldout_recall", "heldout_specificity"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=perf, x="motif_context", y="value", hue="metric", ax=axes[1])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Heldout threshold metrics")
    axes[1].set_xlabel("DRACH motif")
    axes[1].set_ylabel("metric value")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(title="")
    savefig(out / "08_seen_unseen_motif_gap.png")


def plot_run_level_heldout(raw_runs: pd.DataFrame, out: Path, *, heldout_label: str) -> None:
    if raw_runs.empty:
        return
    data = raw_runs.copy()
    data["metric"] = np.where(data["num_positive"] > 0, "recall", "specificity")
    data["value"] = np.where(data["num_positive"] > 0, data["recall"], data["specificity"])
    data["label"] = np.where(data["num_positive"] > 0, "m6A modified", "unmodified")
    data = data.sort_values(["label", "heldout_run"])

    plt.figure(figsize=(12.5, 4.8))
    ax = sns.barplot(data=data, x="heldout_run", y="value", hue="label", dodge=False, palette=["#c44e52", "#4c72b0"])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{heldout_label}: run-level recall/specificity")
    ax.set_xlabel("")
    ax.set_ylabel("threshold metric at p >= 0.5")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")
    annotate_bars(ax, yoffset=1)
    savefig(out / "09_heldout_run_level_metrics.png")


def plot_training_history(training_csv: Path | None, out: Path, *, evaluated_weight: str | int | None) -> list[str]:
    if training_csv is None or not training_csv.exists():
        return []
    history = pd.read_csv(training_csv)
    if history.empty or "epoch" not in history.columns:
        return []
    numeric = history.copy()
    for column in ["epoch", "train_loss", "train_mod_loss", "train_total_loss", "val_loss", "val_mod_loss", "val_total_loss", "val_mean", "val_median"]:
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    loss_cols = [
        ("train_loss", "train base loss"),
        ("val_loss", "val base loss"),
        ("train_mod_loss", "train mod loss"),
        ("val_mod_loss", "val mod loss"),
        ("train_total_loss", "train total loss"),
        ("val_total_loss", "val total loss"),
    ]
    for column, label in loss_cols:
        if column in numeric.columns and numeric[column].notna().any():
            axes[0].plot(numeric["epoch"], numeric[column], label=label)
    axes[0].set_title("Training and validation losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    val_column = "val_mod_loss" if "val_mod_loss" in numeric.columns and numeric["val_mod_loss"].notna().any() else "val_loss"
    if val_column in numeric.columns and numeric[val_column].notna().any():
        best_idx = numeric[val_column].astype(float).idxmin()
        best_epoch = numeric.loc[best_idx, "epoch"]
        best_loss = numeric.loc[best_idx, val_column]
        axes[0].axvline(best_epoch, color="#c44e52", linestyle="--", linewidth=1.1, label=f"best {val_column}: epoch {best_epoch:g}")
        axes[0].scatter([best_epoch], [best_loss], color="#c44e52", zorder=3)
        axes[0].legend(fontsize=8)

    if evaluated_weight is not None and str(evaluated_weight).isdigit():
        axes[0].axvline(float(evaluated_weight), color="#4c72b0", linestyle=":", linewidth=1.1)

    for column, label in (("val_mean", "val mean acc"), ("val_median", "val median acc")):
        if column in numeric.columns and numeric[column].notna().any():
            axes[1].plot(numeric["epoch"], numeric[column], label=label)
    axes[1].set_title("Validation basecall accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8)
    savefig(out / "00_training_history.png")
    return ["00_training_history.png"]


def plot_curve_grid(
    sites: pd.DataFrame,
    *,
    out: Path,
    motifs: list[str],
    prefix: str,
    title_prefix: str,
) -> list[str]:
    if sites.empty:
        print(f"Skipping {title_prefix} ROC/PR curves: site_predictions.tsv not found.")
        return []

    available = [motif for motif in motifs if motif in set(sites["motif_context"].astype(str))]
    curve_rows = []
    palette = dict(zip(available, sns.color_palette("tab10", n_colors=max(len(available), 1))))

    plt.figure(figsize=(7.2, 6.0))
    ax = plt.gca()
    for motif in available:
        sub = sites[sites["motif_context"] == motif]
        curve = roc_curve_from_scores(sub["target"].to_numpy(), sub["prob_m6a"].to_numpy())
        if curve is None:
            continue
        fpr, tpr, auc = curve
        n_pos = int(np.count_nonzero(sub["target"].to_numpy() == 1))
        n_neg = int(np.count_nonzero(sub["target"].to_numpy() == 0))
        ax.plot(fpr, tpr, lw=2, color=palette[motif], label=f"{motif} AUC={auc:.3f} (+{n_pos}/-{n_neg})")
        curve_rows.append({"eval_set": title_prefix, "motif_context": motif, "curve": "roc", "auc": auc, "positive": n_pos, "negative": n_neg})
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.6")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title_prefix}: ROC by DRACH motif")
    ax.legend(fontsize=8, loc="lower right")
    roc_name = f"{prefix}_roc_by_motif.png"
    savefig(out / roc_name)

    plt.figure(figsize=(7.2, 6.0))
    ax = plt.gca()
    for motif in available:
        sub = sites[sites["motif_context"] == motif]
        curve = pr_curve_from_scores(sub["target"].to_numpy(), sub["prob_m6a"].to_numpy())
        if curve is None:
            continue
        recall, precision, auc = curve
        n_pos = int(np.count_nonzero(sub["target"].to_numpy() == 1))
        n_neg = int(np.count_nonzero(sub["target"].to_numpy() == 0))
        prevalence = n_pos / max(n_pos + n_neg, 1)
        ax.plot(recall, precision, lw=2, color=palette[motif], label=f"{motif} AUPRC={auc:.3f} (+{n_pos}/-{n_neg})")
        ax.hlines(prevalence, 0, 1, colors=[palette[motif]], linestyles=":", linewidth=0.8, alpha=0.45)
        curve_rows.append({"eval_set": title_prefix, "motif_context": motif, "curve": "pr", "auc": auc, "positive": n_pos, "negative": n_neg})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix}: Precision-recall by DRACH motif")
    ax.legend(fontsize=8, loc="lower left")
    pr_name = f"{prefix}_precision_recall_by_motif.png"
    savefig(out / pr_name)

    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(out / f"{prefix}_curve_auc_by_motif.tsv", sep="\t", index=False)
    return [roc_name, pr_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--motif-balance",
        type=Path,
        default=Path("dataset_check_res/stage1_train_mafia_wue_rl/check_reports/motif_balance.tsv"),
    )
    parser.add_argument("--internal-eval-dir", type=Path, default=Path("val_res/mafia_stage1_epoch5"))
    parser.add_argument("--heldout-glob", default="val_res/mafia_stage1_e5_heldout_WUE_splint_batch2*")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_check_res/stage1_train_mafia_wue_rl/figures"))
    parser.add_argument("--training-csv", type=Path, default=None, help="Optional model training.csv for overfit/epoch selection plot.")
    parser.add_argument("--internal-label", default="Internal validation")
    parser.add_argument("--heldout-label", default="Heldout")
    parser.add_argument(
        "--heldout-prefix",
        action="append",
        default=["mafia_stage1_e5_heldout_", "mafia_stage1_epoch4_heldout_", "mafia_stage1_heldout_"],
        help="Directory-name prefix to strip from heldout run names. Can be repeated.",
    )
    parser.add_argument(
        "--skip-curves",
        action="store_true",
        help="Skip ROC and precision-recall curves from site_predictions.tsv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    balance = pd.read_csv(args.motif_balance, sep="\t")
    internal_summary, internal_motifs = load_internal_metrics(args.internal_eval_dir)
    heldout_dirs = resolve_glob(args.heldout_glob)
    heldout_motifs, heldout_runs, heldout_summary = load_heldout_metrics(
        heldout_dirs,
        heldout_label=args.heldout_label,
        heldout_prefixes=args.heldout_prefix,
    )
    motifs = motif_order(balance, internal_motifs, heldout_motifs)

    output_dir = args.output_dir.resolve()
    training_figures = plot_training_history(
        args.training_csv,
        output_dir,
        evaluated_weight=internal_summary.get("weights"),
    )
    plot_dataset_counts(balance, output_dir, motifs)
    plot_coverage_heatmap(
        balance,
        heldout_motifs,
        output_dir,
        motifs,
        internal_label=args.internal_label,
        heldout_label=args.heldout_label,
    )
    plot_internal_metrics(internal_motifs, output_dir, motifs, internal_label=args.internal_label)
    plot_heldout_metrics(heldout_motifs, output_dir, motifs, heldout_label=args.heldout_label)
    plot_overall_summary(
        internal_summary,
        heldout_summary,
        output_dir,
        internal_label=args.internal_label,
        heldout_label=args.heldout_label,
    )
    plot_seen_unseen_gap(balance, internal_motifs, heldout_motifs, output_dir, motifs)
    plot_run_level_heldout(heldout_runs, output_dir, heldout_label=args.heldout_label)
    curve_figures = []
    if not args.skip_curves:
        internal_sites = load_site_predictions(args.internal_eval_dir, eval_set=args.internal_label)
        heldout_sites = load_heldout_site_predictions(
            heldout_dirs,
            heldout_label=args.heldout_label,
            heldout_prefixes=args.heldout_prefix,
        )
        curve_figures.extend(
            plot_curve_grid(
                internal_sites,
                out=output_dir,
                motifs=motifs,
                prefix="10_internal_validation",
                title_prefix="Internal validation",
            )
        )
        curve_figures.extend(
            plot_curve_grid(
                heldout_sites,
                out=output_dir,
                motifs=motifs,
                prefix="11_heldout",
                title_prefix=args.heldout_label,
            )
        )

    summary = {
        "motifs": motifs,
        "internal_validation": internal_summary.get("overall", {}),
        "heldout": heldout_summary,
        "training_figures": training_figures,
        "curve_figures": curve_figures,
        "figures": sorted(str(path.name) for path in output_dir.glob("*.png")),
    }
    (output_dir / "visual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote figures to: {output_dir}")


if __name__ == "__main__":
    main()
