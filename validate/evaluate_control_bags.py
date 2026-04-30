#!/usr/bin/env python3
"""
Build bag-level 0%/100% control scores from a supervised control dataset.

This is for Stage 1 control data that has per-site modification labels in
mod_targets.npy but does not have LLP bag_keys.npy/bag_targets.npy. It evaluates
the model, mean-pools A-head probabilities to read level, infers each read's
control label from its aligned A-site labels, and writes synthetic 0%/100% bags
compatible with vis/plot_bag_level_roc.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def write_tsv(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_loader(args, model, output_dir: Path):
    data_settings = resolve_train_mod_data_settings(
        directory=args.directory,
        output_dir=output_dir,
        chunks=args.chunks,
        valid_chunks=args.valid_chunks,
        caller="control_bag_eval",
    )
    model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=model.config.get("standardisation", {}),
    )
    compute_settings = ComputeSettings(batch_size=args.batchsize, num_workers=args.num_workers, seed=args.seed)
    train_loader, valid_loader = load_train_mod_data(data_settings, model_setup, compute_settings)
    return valid_loader if args.dataset == "valid" else train_loader


def read_level_scores_and_targets(model, outputs, targets, lengths, mod_targets):
    projection = model.align_predictions_to_targets(outputs, targets, lengths, mod_targets)
    head_projection = projection["per_head"]["A"]
    flat_logits = head_projection["flat_logits"].detach().to(torch.float32)
    flat_targets = head_projection["flat_targets"].detach().to(torch.long)
    flat_sample_indices = head_projection["flat_sample_indices"].detach().to(torch.long)
    num_reads = int(lengths.shape[0])
    if flat_logits.numel() == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if flat_logits.ndim != 2 or flat_logits.shape[-1] != 2:
        raise ValueError(f"Expected binary A-head logits with shape [N, 2], got {tuple(flat_logits.shape)}")

    site_probs = torch.softmax(flat_logits, dim=-1)[:, 1]
    site_targets = (flat_targets == 1).to(dtype=torch.float32)

    read_prob_sums = site_probs.new_zeros((num_reads,))
    read_target_sums = site_probs.new_zeros((num_reads,))
    read_counts = site_probs.new_zeros((num_reads,))
    read_prob_sums.scatter_add_(0, flat_sample_indices, site_probs)
    read_target_sums.scatter_add_(0, flat_sample_indices, site_targets)
    read_counts.scatter_add_(0, flat_sample_indices, torch.ones_like(site_probs))

    valid = read_counts > 0
    read_probs = read_prob_sums[valid] / read_counts[valid]
    read_targets = read_target_sums[valid] / read_counts[valid]
    return (
        read_probs.detach().cpu().numpy().astype(np.float32),
        read_targets.detach().cpu().numpy().astype(np.float32),
    )


def collect_reads(args, model, loader) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    model_dtype = next(model.parameters()).dtype
    score_parts = []
    target_parts = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), ascii=True, ncols=100, desc=f"control-bags:{args.dataset}"):
            data, targets, lengths, mod_targets, *extra = batch
            outputs = model(
                data.to(device=device, dtype=model_dtype, non_blocking=True),
                *(item.to(device=device, non_blocking=True) for item in extra),
            )
            read_scores, read_targets = read_level_scores_and_targets(
                model,
                outputs,
                targets.to(device=device, non_blocking=True),
                lengths.to(device=device, non_blocking=True),
                mod_targets.to(device=device, non_blocking=True),
            )
            if read_scores.size:
                score_parts.append(read_scores)
                target_parts.append(read_targets)

    scores = np.concatenate(score_parts) if score_parts else np.zeros((0,), dtype=np.float32)
    targets = np.concatenate(target_parts) if target_parts else np.zeros((0,), dtype=np.float32)
    return scores, targets


def build_control_bags(args, scores: np.ndarray, targets: np.ndarray) -> tuple[list[dict], dict]:
    negative_indices = np.flatnonzero(targets <= args.negative_max_fraction)
    positive_indices = np.flatnonzero(targets >= args.positive_min_fraction)
    ambiguous = int(targets.size - negative_indices.size - positive_indices.size)

    rng = np.random.default_rng(args.seed)
    if args.shuffle:
        rng.shuffle(negative_indices)
        rng.shuffle(positive_indices)

    rows = []
    bag_key = 0
    for target_fraction, indices in ((0.0, negative_indices), (1.0, positive_indices)):
        usable = int(indices.size // args.bag_size * args.bag_size)
        for start in range(0, usable, args.bag_size):
            selected = indices[start:start + args.bag_size]
            rows.append(
                {
                    "bag_key": bag_key,
                    "target_ratio": target_fraction * 100.0,
                    "target_fraction": target_fraction,
                    "bag_score": float(np.mean(scores[selected])),
                    "num_reads": int(selected.size),
                }
            )
            bag_key += 1

    summary = {
        "dataset": args.dataset,
        "num_reads_with_a_sites": int(targets.size),
        "num_negative_reads": int(negative_indices.size),
        "num_positive_reads": int(positive_indices.size),
        "num_ambiguous_reads": ambiguous,
        "negative_max_fraction": float(args.negative_max_fraction),
        "positive_min_fraction": float(args.positive_min_fraction),
        "bag_size": int(args.bag_size),
        "num_bags": len(rows),
        "num_negative_bags": int(sum(1 for row in rows if row["target_fraction"] == 0.0)),
        "num_positive_bags": int(sum(1 for row in rows if row["target_fraction"] == 1.0)),
    }
    return rows, summary


def argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_directory")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["train", "valid"], default="valid")
    parser.add_argument("--weights", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--chunks", type=int, default=None)
    parser.add_argument("--valid-chunks", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--bag-size", type=int, default=20)
    parser.add_argument("--negative-max-fraction", type=float, default=0.01)
    parser.add_argument("--positive-min-fraction", type=float, default=0.99)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--no-half", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    return parser


def main(args) -> None:
    if args.bag_size <= 0:
        raise ValueError(f"--bag-size must be positive, got {args.bag_size}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    init(args.seed, args.device, deterministic=True)
    use_half = str(args.device).startswith("cuda") and not args.no_half
    model = load_model(
        args.model_directory,
        args.device,
        weights=args.weights,
        half=use_half,
        compile=not args.no_compile,
    )
    if "A" not in getattr(model, "mod_bases", []):
        raise RuntimeError("Loaded model does not expose an A-head.")

    loader = resolve_loader(args, model, args.output_dir)
    scores, targets = collect_reads(args, model, loader)
    rows, summary = build_control_bags(args, scores, targets)
    if summary["num_negative_bags"] == 0 or summary["num_positive_bags"] == 0:
        raise ValueError(
            "Control bag ROC needs at least one 0% bag and one 100% bag. "
            f"summary={summary}"
        )

    write_tsv(
        args.output_dir / "bag_scores.tsv",
        rows,
        ("bag_key", "target_ratio", "target_fraction", "bag_score", "num_reads"),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(argparser().parse_args())
