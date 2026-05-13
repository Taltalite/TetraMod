#!/usr/bin/env python3
"""
Evaluate a promoted Stage 1 mAFiA control model on labeled center A sites.

The evaluator reuses the model's Viterbi/edlib target projection, matching the
control warm-up loss path.  It reports overall and grouped A-head m6A metrics.
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tetramod.util import init, load_model


DEFAULT_THRESHOLD = 0.5
COPY_FIELD_DEFAULT = ""


class MafiaStage1Dataset(Dataset):
    def __init__(self, directory: Path, limit: int | None = None):
        self.directory = Path(directory)
        self.chunks = np.load(self.directory / "chunks.npy", mmap_mode="r")
        self.references = np.load(self.directory / "references.npy", mmap_mode="r")
        self.lengths = np.load(self.directory / "reference_lengths.npy", mmap_mode="r")
        self.mod_targets = np.load(self.directory / "mod_targets.npy", mmap_mode="r")
        metadata_file = np.load(self.directory / "metadata.npz")
        self.metadata = {name: metadata_file[name] for name in metadata_file.files}
        self.limit = int(limit) if limit and int(limit) > 0 else int(self.lengths.shape[0])
        self.limit = min(self.limit, int(self.lengths.shape[0]))
        self._validate()

    def _validate(self) -> None:
        n = int(self.lengths.shape[0])
        for name, array in (
            ("chunks", self.chunks),
            ("references", self.references),
            ("mod_targets", self.mod_targets),
        ):
            if int(array.shape[0]) != n:
                raise ValueError(f"{self.directory}: {name}.npy has {array.shape[0]} rows, expected {n}")
        for field, values in self.metadata.items():
            if int(values.shape[0]) != n:
                raise ValueError(f"{self.directory}: metadata field {field} has {values.shape[0]} rows, expected {n}")

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        idx = int(idx)
        return (
            self.chunks[idx].astype(np.float32)[None, :],
            self.references[idx].astype(np.int64),
            np.asarray(self.lengths[idx], dtype=np.int64),
            self.mod_targets[idx].astype(np.int64),
            np.asarray(idx, dtype=np.int64),
        )


def safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def auc_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x))


def binary_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float | None, float | None]:
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


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, object]:
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    roc_auc, pr_auc = binary_auc_metrics(y_true, y_prob)
    bce = float(
        F.binary_cross_entropy(
            torch.from_numpy(np.clip(y_prob, 1e-7, 1 - 1e-7)).to(torch.float64),
            torch.from_numpy(y_true).to(torch.float64),
        ).item()
    ) if y_true.size else None

    return {
        "num_sites": int(y_true.size),
        "num_positive": positives,
        "num_negative": negatives,
        "positive_rate": safe_div(positives, y_true.size),
        "predicted_positive_rate": float(np.mean(y_pred)) if y_pred.size else None,
        "threshold": float(threshold),
        "accuracy": safe_div(tp + tn, y_true.size),
        "balanced_accuracy": None if positives == 0 or negatives == 0 else 0.5 * ((tp / positives) + (tn / negatives)),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": safe_div(2 * tp, (2 * tp) + fp + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "bce": bce,
        "mean_prob": float(np.mean(y_prob)) if y_prob.size else None,
        "median_prob": float(np.median(y_prob)) if y_prob.size else None,
        "mean_positive_prob": float(np.mean(y_prob[y_true == 1])) if positives else None,
        "mean_negative_prob": float(np.mean(y_prob[y_true == 0])) if negatives else None,
        "median_positive_prob": float(np.median(y_prob[y_true == 1])) if positives else None,
        "median_negative_prob": float(np.median(y_prob[y_true == 0])) if negatives else None,
        "q05_positive_prob": float(np.quantile(y_prob[y_true == 1], 0.05)) if positives else None,
        "q95_negative_prob": float(np.quantile(y_prob[y_true == 0], 0.95)) if negatives else None,
    }


def metadata_value(metadata: dict[str, np.ndarray], field: str, idx: int) -> str:
    values = metadata.get(field)
    if values is None:
        return COPY_FIELD_DEFAULT
    return str(values[int(idx)])


def write_tsv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_records(site_rows: Sequence[dict[str, object]], fields: Sequence[str], threshold: float) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in site_rows:
        key = tuple(str(row.get(field, COPY_FIELD_DEFAULT)) for field in fields)
        groups.setdefault(key, []).append(row)

    records = []
    for key, rows in sorted(groups.items()):
        y_true = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
        y_prob = np.asarray([float(row["prob_m6a"]) for row in rows], dtype=np.float32)
        metrics = binary_metrics(y_true, y_prob, threshold)
        records.append({
            "group_by": ",".join(fields),
            **{field: value for field, value in zip(fields, key)},
            **metrics,
        })
    return records


def evaluate(args) -> dict[str, object]:
    dataset_dir = Path(args.dataset_dir)
    if args.split == "validation":
        dataset_dir = dataset_dir / "validation"
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_dir) / f"mafia_stage1_eval_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    init(args.seed, args.device, deterministic=True)
    use_half = str(args.device).startswith("cuda") and not args.no_half
    model = load_model(
        args.model_dir,
        args.device,
        weights=args.weights,
        half=use_half,
        compile=bool(args.compile),
    )
    if not hasattr(model, "align_predictions_to_targets"):
        raise RuntimeError("Loaded model does not expose align_predictions_to_targets().")
    model.eval()

    dataset = MafiaStage1Dataset(dataset_dir, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    device = torch.device(args.device)
    model_dtype = next(model.parameters()).dtype
    site_rows: list[dict[str, object]] = []
    skipped_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=f"eval:{args.split}", unit="batch", ascii=True, ncols=100):
            data, targets, lengths, mod_targets, sample_indices = batch
            outputs = model(
                data.to(device=device, dtype=model_dtype, non_blocking=True),
                sample_indices.to(device=device, non_blocking=True),
            )
            projection = model.align_predictions_to_targets(
                outputs,
                targets.to(device=device, non_blocking=True),
                lengths.to(device=device, non_blocking=True),
                mod_targets.to(device=device, non_blocking=True),
            )
            head = projection["per_head"]["A"]
            flat_logits = head["flat_logits"].detach().to(torch.float32)
            flat_targets = head["flat_targets"].detach().cpu().numpy().astype(np.int64)
            flat_sample_indices = head["flat_sample_indices"].detach().cpu().numpy().astype(np.int64)
            if flat_targets.size == 0:
                skipped_batches += 1
                continue
            if flat_logits.ndim != 2 or flat_logits.shape[-1] != 2:
                raise ValueError(f"Expected binary A-head logits with shape [N, 2], got {tuple(flat_logits.shape)}")
            probs = torch.softmax(flat_logits, dim=-1)[:, 1].detach().cpu().numpy().astype(np.float32)
            batch_sample_indices = sample_indices.detach().cpu().numpy().astype(np.int64)

            for local_target, prob, batch_pos in zip(flat_targets, probs, flat_sample_indices):
                source_idx = int(batch_sample_indices[int(batch_pos)])
                site_rows.append({
                    "sample_index": source_idx,
                    "target": int(local_target),
                    "prob_m6a": float(prob),
                    "pred": int(prob >= args.threshold),
                    "source_dataset": metadata_value(dataset.metadata, "source_dataset", source_idx),
                    "run_id": metadata_value(dataset.metadata, "run_id", source_idx),
                    "motif_context": metadata_value(dataset.metadata, "motif_context", source_idx),
                    "ligation_strategy": metadata_value(dataset.metadata, "ligation_strategy", source_idx),
                    "modification_status": metadata_value(dataset.metadata, "modification_status", source_idx),
                    "primary_site_key": metadata_value(dataset.metadata, "primary_site_key", source_idx),
                    "oligo_ids": metadata_value(dataset.metadata, "oligo_ids", source_idx),
                })

    if not site_rows:
        raise RuntimeError("No A-head labeled sites were aligned during evaluation.")

    y_true = np.asarray([int(row["target"]) for row in site_rows], dtype=np.int64)
    y_prob = np.asarray([float(row["prob_m6a"]) for row in site_rows], dtype=np.float32)
    overall = binary_metrics(y_true, y_prob, args.threshold)

    group_fields = [
        ("source_dataset",),
        ("run_id",),
        ("motif_context",),
        ("ligation_strategy",),
        ("modification_status",),
        ("source_dataset", "motif_context"),
        ("run_id", "motif_context"),
    ]
    group_rows = []
    for fields in group_fields:
        group_rows.extend(group_records(site_rows, fields, args.threshold))

    metric_fields = [
        "group_by",
        "source_dataset",
        "run_id",
        "motif_context",
        "ligation_strategy",
        "modification_status",
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
    write_tsv(output_dir / "group_metrics.tsv", group_rows, metric_fields)
    if args.write_sites:
        write_tsv(
            output_dir / "site_predictions.tsv",
            site_rows,
            (
                "sample_index",
                "target",
                "prob_m6a",
                "pred",
                "source_dataset",
                "run_id",
                "motif_context",
                "ligation_strategy",
                "modification_status",
                "primary_site_key",
                "oligo_ids",
            ),
        )

    summary = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "split": args.split,
        "weights": "last" if args.weights is None else args.weights,
        "threshold": float(args.threshold),
        "num_samples_loaded": int(len(dataset)),
        "num_batches": int(len(loader)),
        "skipped_batches_without_aligned_sites": int(skipped_batches),
        "overall": overall,
        "outputs": {
            "summary_json": str((output_dir / "summary.json").resolve()),
            "summary_txt": str((output_dir / "summary.txt").resolve()),
            "group_metrics_tsv": str((output_dir / "group_metrics.tsv").resolve()),
            "site_predictions_tsv": str((output_dir / "site_predictions.tsv").resolve()) if args.write_sites else None,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        f"model_dir: {summary['model_dir']}",
        f"dataset_dir: {summary['dataset_dir']}",
        f"split: {args.split}",
        f"weights: {summary['weights']}",
        f"num_sites: {overall['num_sites']}",
        f"num_positive: {overall['num_positive']}",
        f"num_negative: {overall['num_negative']}",
        f"bce: {overall['bce']:.6f}" if overall["bce"] is not None else "bce: NA",
        f"roc_auc: {overall['roc_auc']:.6f}" if overall["roc_auc"] is not None else "roc_auc: NA",
        f"pr_auc: {overall['pr_auc']:.6f}" if overall["pr_auc"] is not None else "pr_auc: NA",
        f"balanced_accuracy: {overall['balanced_accuracy']:.6f}" if overall["balanced_accuracy"] is not None else "balanced_accuracy: NA",
        f"accuracy: {overall['accuracy']:.6f}" if overall["accuracy"] is not None else "accuracy: NA",
        f"recall: {overall['recall']:.6f}" if overall["recall"] is not None else "recall: NA",
        f"specificity: {overall['specificity']:.6f}" if overall["specificity"] is not None else "specificity: NA",
        f"mean_positive_prob: {overall['mean_positive_prob']:.6f}" if overall["mean_positive_prob"] is not None else "mean_positive_prob: NA",
        f"mean_negative_prob: {overall['mean_negative_prob']:.6f}" if overall["mean_negative_prob"] is not None else "mean_negative_prob: NA",
        f"median_positive_prob: {overall['median_positive_prob']:.6f}" if overall["median_positive_prob"] is not None else "median_positive_prob: NA",
        f"median_negative_prob: {overall['median_negative_prob']:.6f}" if overall["median_negative_prob"] is not None else "median_negative_prob: NA",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote evaluation outputs to: {output_dir}")
    return summary


def argparser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_dir")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights", type=int, default=None, help="Checkpoint epoch; default uses latest weights_*.tar.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N samples from the selected split; 0 means all.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--no-half", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile while evaluating.")
    parser.add_argument("--write-sites", action="store_true", default=False)
    return parser


def main():
    evaluate(argparser().parse_args())


if __name__ == "__main__":
    main()
