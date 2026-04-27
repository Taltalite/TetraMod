#!/usr/bin/env python3
"""
Evaluate promoted LLP bag scores on datasets with bag_keys.npy/bag_targets.npy.

Hard check covered here:
- bag-level mean posterior should be monotonic across known ratios such as
  0/25/50/75/100.

Use this on both leave-run and leave-site LLP datasets produced by
gen_data/build_llp_mixture_dataset.py.
"""

from __future__ import annotations

import csv
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
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


EXPECTED_RATIOS = (0.0, 25.0, 50.0, 75.0, 100.0)


def resolve_loader(args, model, output_dir: Path):
    data_settings = resolve_train_mod_data_settings(
        directory=args.directory,
        output_dir=output_dir,
        chunks=args.chunks,
        valid_chunks=args.valid_chunks,
        caller="llp_bag_eval",
    )
    model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=model.config.get("standardisation", {}),
    )
    compute_settings = ComputeSettings(batch_size=args.batchsize, num_workers=args.num_workers, seed=args.seed)
    train_loader, valid_loader = load_train_mod_data(data_settings, model_setup, compute_settings)
    return valid_loader if args.dataset == "valid" else train_loader


def read_level_probs(model, outputs, targets, lengths, mod_targets):
    projection = model.align_predictions_to_targets(outputs, targets, lengths, mod_targets)
    head_projection = projection["per_head"]["A"]
    flat_logits = head_projection["flat_logits"].detach().to(torch.float32)
    flat_sample_indices = head_projection["flat_sample_indices"].detach().to(torch.long)
    num_reads = int(lengths.shape[0])
    if flat_logits.numel() == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if flat_logits.ndim != 2 or flat_logits.shape[-1] != 2:
        raise ValueError(f"Expected binary A-head logits with shape [N, 2], got {tuple(flat_logits.shape)}")

    site_probs = torch.softmax(flat_logits, dim=-1)[:, 1]
    read_sums = site_probs.new_zeros((num_reads,))
    read_counts = site_probs.new_zeros((num_reads,))
    read_sums.scatter_add_(0, flat_sample_indices, site_probs)
    read_counts.scatter_add_(0, flat_sample_indices, torch.ones_like(site_probs))
    valid = read_counts > 0
    read_indices = torch.nonzero(valid, as_tuple=False).flatten()
    read_probs = read_sums.index_select(0, read_indices) / read_counts.index_select(0, read_indices)
    return (
        read_probs.detach().cpu().numpy().astype(np.float32),
        read_indices.detach().cpu().numpy().astype(np.int64),
    )


def evaluate(args, model, loader):
    device = torch.device(args.device)
    model_dtype = next(model.parameters()).dtype
    bags = {}
    num_reads = 0

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), ascii=True, ncols=100, desc=f"llp:{args.dataset}"):
            data, targets, lengths, mod_targets, *extra = batch
            if len(extra) < 3:
                raise ValueError("LLP bag evaluation requires sample_keys, bag_keys, and bag_targets in the dataset.")
            outputs = model(
                data.to(device=device, dtype=model_dtype, non_blocking=True),
                *(item.to(device=device, non_blocking=True) for item in extra),
            )
            read_probs, read_indices = read_level_probs(
                model,
                outputs,
                targets.to(device=device, non_blocking=True),
                lengths.to(device=device, non_blocking=True),
                mod_targets.to(device=device, non_blocking=True),
            )
            if read_probs.size == 0:
                continue
            bag_keys = extra[1].detach().cpu().numpy().astype(np.int64)[read_indices]
            bag_targets = extra[2].detach().cpu().numpy().astype(np.float32)[read_indices]
            for key, target, score in zip(bag_keys, bag_targets, read_probs):
                record = bags.setdefault(int(key), {"sum": 0.0, "count": 0, "target_sum": 0.0})
                record["sum"] += float(score)
                record["target_sum"] += float(target)
                record["count"] += 1
                num_reads += 1

    bag_records = []
    for key, record in bags.items():
        count = int(record["count"])
        target = record["target_sum"] / max(count, 1)
        score = record["sum"] / max(count, 1)
        bag_records.append({
            "bag_key": int(key),
            "target_ratio": float(target * 100.0),
            "target_fraction": float(target),
            "bag_score": float(score),
            "num_reads": count,
        })
    bag_records.sort(key=lambda item: (item["target_ratio"], item["bag_key"]))
    return bag_records, num_reads


def monotonicity(bag_records):
    by_ratio = {}
    for record in bag_records:
        by_ratio.setdefault(float(record["target_ratio"]), []).append(float(record["bag_score"]))
    ratio_records = [
        {
            "ratio": ratio,
            "num_bags": len(scores),
            "mean_bag_score": float(np.mean(scores)),
            "median_bag_score": float(np.median(scores)),
        }
        for ratio, scores in sorted(by_ratio.items())
    ]
    comparisons = []
    ok = True
    for prev, curr in zip(ratio_records, ratio_records[1:]):
        passed = curr["mean_bag_score"] + 1e-8 >= prev["mean_bag_score"]
        comparisons.append({
            "prev_ratio": prev["ratio"],
            "prev_mean_bag_score": prev["mean_bag_score"],
            "curr_ratio": curr["ratio"],
            "curr_mean_bag_score": curr["mean_bag_score"],
            "non_decreasing": bool(passed),
        })
        ok = ok and passed
    ratios_present = [item["ratio"] for item in ratio_records]
    expected_present = [ratio for ratio in EXPECTED_RATIOS if ratio in ratios_present]
    return {
        "ratios_present": ratios_present,
        "expected_ratios_present": expected_present,
        "has_full_expected_grid": expected_present == list(EXPECTED_RATIOS),
        "non_decreasing_by_mean_bag_score": bool(ok),
        "by_ratio": ratio_records,
        "comparisons": comparisons,
    }


def write_tsv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def argparser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_directory")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", choices=["train", "valid"], default="valid")
    parser.add_argument("--weights", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--chunks", type=int, default=None)
    parser.add_argument("--valid-chunks", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--no-half", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    return parser


def main(args):
    output_dir = args.output_dir or (Path(args.model_directory) / f"llp_bag_eval_{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
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
    loader = resolve_loader(args, model, output_dir)
    bag_records, num_reads = evaluate(args, model, loader)
    mono = monotonicity(bag_records)

    write_tsv(
        output_dir / "bag_scores.tsv",
        bag_records,
        ("bag_key", "target_ratio", "target_fraction", "bag_score", "num_reads"),
    )
    summary = {
        "model_directory": str(Path(args.model_directory).resolve()),
        "directory": str(args.directory.resolve()),
        "dataset": args.dataset,
        "num_bags": len(bag_records),
        "num_reads": int(num_reads),
        "monotonicity": mono,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["monotonicity"], indent=2))


if __name__ == "__main__":
    main(argparser().parse_args())
