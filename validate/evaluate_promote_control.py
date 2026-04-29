#!/usr/bin/env python3
"""
Minimal control-warmup evaluator for promoted or baseline A-head models.

This script answers two narrow questions:
1. Can the model separate full-mod from IVT on the current control datasets?
2. If mixed-ratio datasets are available, does predicted A-head modification
   probability increase monotonically with the supplied mixture ratio?
"""

from __future__ import annotations

import csv
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from bonito.data import ComputeSettings, ModelSetup
from tetramod.train_mod_data import load_train_mod_data
from tetramod.util import init, load_model

try:
    from train_mod_common import resolve_train_mod_data_settings
except ImportError:
    from validate.train_mod_common import resolve_train_mod_data_settings


CONTROL_THRESHOLD = 0.5
EXPECTED_MIX_RATIOS = (0.0, 25.0, 50.0, 75.0, 100.0)


@dataclass
class DatasetSpec:
    name: str
    directory: Path
    ratio: Optional[float] = None


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def auc_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(y, x))


def binary_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    y_true = y_true.astype(np.int64)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return None, None

    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order]

    tps = np.cumsum(sorted_true == 1)
    fps = np.cumsum(sorted_true == 0)
    threshold_idx = np.where(np.diff(sorted_scores))[0]
    threshold_idx = np.r_[threshold_idx, len(sorted_scores) - 1]

    tps = tps[threshold_idx]
    fps = fps[threshold_idx]

    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    roc_auc = auc_trapezoid(tpr, fpr)

    precision = np.r_[1.0, tps / np.maximum(tps + fps, 1)]
    recall = np.r_[0.0, tps / positives]
    pr_auc = auc_trapezoid(precision, recall)
    return roc_auc, pr_auc


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = CONTROL_THRESHOLD) -> Dict[str, Optional[float]]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    roc_auc, pr_auc = binary_auc_metrics(y_true, y_prob)

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    return {
        "threshold": float(threshold),
        "num_sites": int(y_true.size),
        "num_positive": positives,
        "num_negative": negatives,
        "accuracy": safe_div(tp + tn, y_true.size),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, (2 * tp) + fp + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mean_positive_prob": float(np.mean(y_prob[y_true == 1])) if positives else None,
        "mean_negative_prob": float(np.mean(y_prob[y_true == 0])) if negatives else None,
    }


def parse_mix_dataset(spec: str) -> DatasetSpec:
    try:
        ratio_text, directory_text = str(spec).split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid --mix-dataset value {spec!r}; expected <ratio>:<directory>.") from exc
    ratio = float(ratio_text)
    directory = Path(directory_text)
    return DatasetSpec(name=f"mix_{ratio:g}", directory=directory, ratio=ratio)


def build_dataset_specs(args) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    if args.ivt_dir:
        specs.append(DatasetSpec(name="ivt", directory=Path(args.ivt_dir), ratio=0.0))
    for item in args.mix_dataset:
        specs.append(parse_mix_dataset(item))
    if args.full_mod_dir:
        specs.append(DatasetSpec(name="full_mod", directory=Path(args.full_mod_dir), ratio=100.0))
    if not specs:
        raise ValueError("Provide at least one of --ivt-dir, --full-mod-dir, or --mix-dataset.")
    return specs


def resolve_output_dir(args) -> Path:
    weights_label = "last" if args.weights is None else str(args.weights)
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.model_directory) / f"promote_control_eval_{args.dataset}_weights_{weights_label}"


def clear_alignment_cache(model) -> None:
    unwrapped = getattr(model, "module", model)
    unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
    cache = getattr(unwrapped, "_alignment_cache", None)
    if cache is not None:
        cache.clear()
    if hasattr(unwrapped, "reset_alignment_cache_stats"):
        unwrapped.reset_alignment_cache_stats()


def resolve_loader(args, model, dataset_dir: Path, output_dir: Path):
    data_settings = resolve_train_mod_data_settings(
        directory=dataset_dir,
        output_dir=output_dir,
        chunks=args.chunks,
        valid_chunks=args.valid_chunks,
        caller="promote_control_eval",
    )
    model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=model.config.get("standardisation", {}),
    )
    compute_settings = ComputeSettings(batch_size=args.batchsize, num_workers=args.num_workers, seed=args.seed)
    train_loader, valid_loader = load_train_mod_data(data_settings, model_setup, compute_settings)
    return valid_loader if args.dataset == "valid" else train_loader


def extract_a_head_probs(model, outputs, targets, lengths, mod_targets) -> Tuple[np.ndarray, np.ndarray]:
    projection = model.align_predictions_to_targets(outputs, targets, lengths, mod_targets)
    head_projection = projection["per_head"]["A"]
    flat_logits = head_projection["flat_logits"].detach().to(torch.float32)
    flat_targets = head_projection["flat_targets"].detach().cpu().numpy().astype(np.int64)
    if flat_logits.numel() == 0:
        return np.zeros((0,), dtype=np.float32), flat_targets
    if flat_logits.ndim != 2 or flat_logits.shape[-1] != 2:
        raise ValueError(f"Expected binary A-head logits with shape [N, 2], got {tuple(flat_logits.shape)}")
    probs = torch.softmax(flat_logits, dim=-1)[:, 1].detach().cpu().numpy().astype(np.float32)
    return probs, flat_targets


def evaluate_dataset(args, model, dataset_spec: DatasetSpec, output_dir: Path) -> Dict[str, object]:
    clear_alignment_cache(model)
    loader = resolve_loader(args, model, dataset_spec.directory, output_dir)
    model_dtype = next(model.parameters()).dtype
    device = torch.device(args.device)
    prob_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    num_batches = len(loader)
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, total=num_batches, ascii=True, ncols=100, desc=f"eval:{dataset_spec.name}"):
            data, targets, lengths, mod_targets, *extra = batch
            outputs = model(
                data.to(device=device, dtype=model_dtype, non_blocking=True),
                *(item.to(device=device, non_blocking=True) for item in extra),
            )
            probs, flat_targets = extract_a_head_probs(
                model,
                outputs,
                targets.to(device=device, non_blocking=True),
                lengths.to(device=device, non_blocking=True),
                mod_targets.to(device=device, non_blocking=True),
            )
            prob_parts.append(probs)
            target_parts.append(flat_targets)
            num_samples += int(data.shape[0])

    y_prob = np.concatenate(prob_parts) if prob_parts else np.zeros((0,), dtype=np.float32)
    y_true = np.concatenate(target_parts) if target_parts else np.zeros((0,), dtype=np.int64)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=args.threshold) if y_true.size else {
        "threshold": float(args.threshold),
        "num_sites": 0,
        "num_positive": 0,
        "num_negative": 0,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "roc_auc": None,
        "pr_auc": None,
        "mean_positive_prob": None,
        "mean_negative_prob": None,
    }

    return {
        "name": dataset_spec.name,
        "directory": str(dataset_spec.directory.resolve()),
        "ratio": dataset_spec.ratio,
        "num_batches": int(num_batches),
        "num_samples": int(num_samples),
        "num_sites": int(y_true.size),
        "positive_rate": float(np.mean(y_true)) if y_true.size else None,
        "mean_pred_mod_prob": float(np.mean(y_prob)) if y_prob.size else None,
        "median_pred_mod_prob": float(np.median(y_prob)) if y_prob.size else None,
        "metrics": metrics,
    }


def monotonicity_check(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    ratio_records = [
        record for record in records
        if record.get("ratio") is not None and record.get("mean_pred_mod_prob") is not None
    ]
    ratio_records = sorted(ratio_records, key=lambda item: (float(item["ratio"]), str(item["name"])))
    comparisons = []
    non_decreasing = True
    for prev, curr in zip(ratio_records, ratio_records[1:]):
        prev_score = float(prev["mean_pred_mod_prob"])
        curr_score = float(curr["mean_pred_mod_prob"])
        ok = curr_score + 1e-8 >= prev_score
        comparisons.append({
            "prev_name": prev["name"],
            "prev_ratio": float(prev["ratio"]),
            "prev_score": prev_score,
            "curr_name": curr["name"],
            "curr_ratio": float(curr["ratio"]),
            "curr_score": curr_score,
            "non_decreasing": bool(ok),
        })
        non_decreasing = non_decreasing and ok

    ratios_present = [float(record["ratio"]) for record in ratio_records]
    expected_present = [ratio for ratio in EXPECTED_MIX_RATIOS if ratio in ratios_present]
    return {
        "ratios_present": ratios_present,
        "expected_ratios_requested": list(EXPECTED_MIX_RATIOS),
        "expected_ratios_present": expected_present,
        "has_full_expected_grid": expected_present == list(EXPECTED_MIX_RATIOS),
        "non_decreasing_by_mean_prob": bool(non_decreasing),
        "comparisons": comparisons,
    }


def combined_control_metrics(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    ivt = next((record for record in records if record["name"] == "ivt"), None)
    full_mod = next((record for record in records if record["name"] == "full_mod"), None)
    if ivt is None or full_mod is None:
        return {
            "available": False,
            "reason": "Need both --ivt-dir and --full-mod-dir to compute direct control separation metrics.",
        }

    ivt_mean = ivt.get("mean_pred_mod_prob")
    full_mean = full_mod.get("mean_pred_mod_prob")
    ivt_metrics = ivt["metrics"]
    full_metrics = full_mod["metrics"]
    return {
        "available": True,
        "ivt_mean_pred_mod_prob": ivt_mean,
        "full_mod_mean_pred_mod_prob": full_mean,
        "mean_gap_full_minus_ivt": None if ivt_mean is None or full_mean is None else float(full_mean - ivt_mean),
        "ivt_num_sites": int(ivt_metrics["num_sites"]),
        "full_mod_num_sites": int(full_metrics["num_sites"]),
        "ivt_positive_rate": ivt.get("positive_rate"),
        "full_mod_positive_rate": full_mod.get("positive_rate"),
        "full_gt_ivt_by_mean_prob": None if ivt_mean is None or full_mean is None else bool(full_mean > ivt_mean),
    }


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_text_summary(summary: Dict[str, object]) -> str:
    lines = [
        f"model_directory: {summary['model_directory']}",
        f"dataset_split: {summary['dataset_split']}",
        f"threshold: {summary['threshold']}",
        f"num_datasets: {len(summary['datasets'])}",
    ]
    control = summary["control_comparison"]
    if control.get("available"):
        lines.extend([
            f"ivt_mean_pred_mod_prob: {control['ivt_mean_pred_mod_prob']:.6f}",
            f"full_mod_mean_pred_mod_prob: {control['full_mod_mean_pred_mod_prob']:.6f}",
            f"mean_gap_full_minus_ivt: {control['mean_gap_full_minus_ivt']:.6f}",
            f"full_gt_ivt_by_mean_prob: {control['full_gt_ivt_by_mean_prob']}",
        ])
    monotonicity = summary["monotonicity"]
    lines.extend([
        f"ratios_present: {monotonicity['ratios_present']}",
        f"has_full_expected_grid: {monotonicity['has_full_expected_grid']}",
        f"non_decreasing_by_mean_prob: {monotonicity['non_decreasing_by_mean_prob']}",
    ])
    return "\n".join(lines) + "\n"


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_directory")
    parser.add_argument("--full-mod-dir", type=Path)
    parser.add_argument("--ivt-dir", type=Path)
    parser.add_argument(
        "--mix-dataset",
        action="append",
        default=[],
        help="Optional mixed-ratio dataset in the form <ratio>:<directory>, e.g. 25:/path/to/mix25",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", choices=["train", "valid"], default="valid")
    parser.add_argument("--weights", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--chunks", type=int, default=None)
    parser.add_argument("--valid-chunks", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=CONTROL_THRESHOLD)
    parser.add_argument("--no-half", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    return parser


def main(args):
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_dataset_specs(args)

    init(args.seed, args.device, deterministic=True)
    use_half = str(args.device).startswith("cuda") and not args.no_half
    model = load_model(
        args.model_directory,
        args.device,
        weights=args.weights,
        half=use_half,
        compile=not args.no_compile,
    )
    if not hasattr(model, "align_predictions_to_targets"):
        raise RuntimeError("Loaded model does not expose align_predictions_to_targets().")
    if "A" not in getattr(model, "mod_bases", []):
        raise RuntimeError("Loaded model does not expose an A-head.")

    dataset_records = [evaluate_dataset(args, model, spec, output_dir) for spec in specs]
    monotonicity = monotonicity_check(dataset_records)
    control_comparison = combined_control_metrics(dataset_records)

    rows = []
    for record in dataset_records:
        metrics = record["metrics"]
        rows.append({
            "name": record["name"],
            "ratio": record["ratio"],
            "num_samples": record["num_samples"],
            "num_sites": record["num_sites"],
            "positive_rate": record["positive_rate"],
            "mean_pred_mod_prob": record["mean_pred_mod_prob"],
            "median_pred_mod_prob": record["median_pred_mod_prob"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

    write_tsv(
        output_dir / "dataset_metrics.tsv",
        rows,
        (
            "name",
            "ratio",
            "num_samples",
            "num_sites",
            "positive_rate",
            "mean_pred_mod_prob",
            "median_pred_mod_prob",
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ),
    )

    summary = {
        "model_directory": str(Path(args.model_directory).resolve()),
        "dataset_split": args.dataset,
        "threshold": float(args.threshold),
        "datasets": dataset_records,
        "control_comparison": control_comparison,
        "monotonicity": monotonicity,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    (output_dir / "summary.txt").write_text(build_text_summary(summary), encoding="utf-8")


if __name__ == "__main__":
    main(argparser().parse_args())
