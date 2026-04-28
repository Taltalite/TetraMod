#!/usr/bin/env python3
"""
Diagnose known-ratio LLP datasets before or after promoted training.

The script reads the numpy dataset emitted by gen_data/build_llp_mixture_dataset.py
and summarizes per-ratio data balance, bag size, sequencing-quality metadata,
candidate A-site counts, and categorical distribution shifts.
"""

from __future__ import annotations

import csv
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


IGNORE_INDEX = -100
DEFAULT_CATEGORICAL_FIELDS = (
    "contig",
    "run_id",
    "primary_site_key",
    "kmer_context",
    "motif_context",
)
DEFAULT_NUMERIC_FIELDS = (
    "mean_qscore",
    "mapping_coverage",
    "reference_length",
    "candidate_a_sites",
)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def split_path(dataset_dir: Path, split: str) -> Path:
    if split == "train":
        return dataset_dir
    return dataset_dir / "validation"


def ratio_label_from_fraction(value: float) -> str:
    percent = float(value) * 100.0
    if abs(percent - round(percent)) < 1e-6:
        return str(int(round(percent)))
    return f"{percent:g}"


def as_str_array(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def load_split(dataset_dir: Path, split: str) -> dict:
    directory = split_path(dataset_dir, split)
    require_file(directory / "bag_keys.npy")
    require_file(directory / "bag_targets.npy")
    require_file(directory / "reference_lengths.npy")
    require_file(directory / "mod_targets.npy")
    require_file(directory / "metadata.npz")

    bag_keys = np.load(directory / "bag_keys.npy", mmap_mode="r")
    bag_targets = np.load(directory / "bag_targets.npy", mmap_mode="r")
    reference_lengths = np.load(directory / "reference_lengths.npy", mmap_mode="r")
    mod_targets = np.load(directory / "mod_targets.npy", mmap_mode="r")
    metadata_npz = np.load(directory / "metadata.npz", allow_pickle=False)
    metadata = {key: np.asarray(metadata_npz[key]) for key in metadata_npz.files}

    ratio_labels_path = directory / "ratio_labels.npy"
    if ratio_labels_path.exists():
        ratio_labels = as_str_array(np.load(ratio_labels_path, allow_pickle=False))
    else:
        ratio_labels = np.asarray([ratio_label_from_fraction(value) for value in np.asarray(bag_targets)])

    n = int(bag_targets.shape[0])
    if not (bag_keys.shape[0] == reference_lengths.shape[0] == mod_targets.shape[0] == ratio_labels.shape[0] == n):
        raise ValueError(
            f"{directory}: dataset arrays must share first dimension N; got "
            f"bag_keys={bag_keys.shape[0]}, bag_targets={bag_targets.shape[0]}, "
            f"reference_lengths={reference_lengths.shape[0]}, mod_targets={mod_targets.shape[0]}, "
            f"ratio_labels={ratio_labels.shape[0]}"
        )
    for field, values in metadata.items():
        if values.shape[0] != n:
            raise ValueError(f"{directory}: metadata field {field!r} has N={values.shape[0]}, expected {n}")

    candidate_a_sites = np.sum(np.asarray(mod_targets) != IGNORE_INDEX, axis=1).astype(np.int32)
    metadata = dict(metadata)
    metadata["reference_length"] = np.asarray(reference_lengths, dtype=np.float32)
    metadata["candidate_a_sites"] = candidate_a_sites
    return {
        "split": split,
        "directory": directory,
        "bag_keys": np.asarray(bag_keys),
        "bag_targets": np.asarray(bag_targets, dtype=np.float32),
        "ratio_labels": ratio_labels,
        "metadata": metadata,
    }


def numeric_stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": int(array.size),
            "finite": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p10": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "finite": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def sorted_ratio_labels(labels: np.ndarray) -> list[str]:
    unique = sorted(set(as_str_array(labels)), key=lambda item: float(item))
    return list(unique)


def ratio_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    labels = as_str_array(labels)
    return {label: labels == label for label in sorted_ratio_labels(labels)}


def summarize_bags(data: dict) -> tuple[list[dict], dict[str, np.ndarray]]:
    bag_keys = data["bag_keys"]
    ratio_labels = data["ratio_labels"]
    metadata = data["metadata"]
    records = []
    bag_sizes_by_ratio = defaultdict(list)

    for key in sorted(np.unique(bag_keys).astype(np.int64)):
        mask = bag_keys == key
        labels = ratio_labels[mask]
        label_counts = Counter(as_str_array(labels))
        ratio_label, ratio_count = label_counts.most_common(1)[0]
        if len(label_counts) > 1:
            mixed = ",".join(f"{label}:{count}" for label, count in sorted(label_counts.items()))
            raise ValueError(f"Bag key {int(key)} spans multiple ratio labels: {mixed}")

        row = {
            "bag_key": int(key),
            "ratio": str(ratio_label),
            "num_reads": int(ratio_count),
        }
        for field in ("mean_qscore", "mapping_coverage", "candidate_a_sites"):
            if field in metadata:
                row[f"mean_{field}"] = float(np.nanmean(np.asarray(metadata[field][mask], dtype=np.float64)))
        records.append(row)
        bag_sizes_by_ratio[str(ratio_label)].append(int(ratio_count))

    return records, {key: np.asarray(value, dtype=np.int32) for key, value in bag_sizes_by_ratio.items()}


def summarize_by_ratio(data: dict, numeric_fields: tuple[str, ...]) -> list[dict]:
    masks = ratio_masks(data["ratio_labels"])
    metadata = data["metadata"]
    bag_keys = data["bag_keys"]
    rows = []
    _, bag_sizes_by_ratio = summarize_bags(data)

    for ratio, mask in masks.items():
        row = {
            "ratio": ratio,
            "num_reads": int(mask.sum()),
            "num_bags": int(np.unique(bag_keys[mask]).size),
            "bag_size": numeric_stats(bag_sizes_by_ratio.get(ratio, np.asarray([], dtype=np.int32))),
        }
        for field in numeric_fields:
            if field in metadata:
                row[field] = numeric_stats(metadata[field][mask])
        rows.append(row)
    return rows


def value_counts(values, mask, top_k: int) -> tuple[list[dict], Counter]:
    selected = as_str_array(values[mask])
    counts = Counter(selected)
    total = sum(counts.values())
    rows = [
        {
            "value": value,
            "count": int(count),
            "fraction": float(count / total) if total else 0.0,
        }
        for value, count in counts.most_common(top_k)
    ]
    return rows, counts


def categorical_summary(data: dict, categorical_fields: tuple[str, ...], top_k: int) -> tuple[list[dict], dict]:
    metadata = data["metadata"]
    masks = ratio_masks(data["ratio_labels"])
    rows = []
    counters = {}
    for field in categorical_fields:
        if field not in metadata:
            continue
        counters[field] = {}
        for ratio, mask in masks.items():
            top_rows, counts = value_counts(metadata[field], mask, top_k)
            counters[field][ratio] = counts
            for item in top_rows:
                rows.append({
                    "split": data["split"],
                    "field": field,
                    "ratio": ratio,
                    **item,
                })
    return rows, counters


def distribution_distance(left: Counter, right: Counter) -> dict:
    left_total = float(sum(left.values()))
    right_total = float(sum(right.values()))
    keys = set(left) | set(right)
    if left_total == 0.0 or right_total == 0.0:
        return {"total_variation": None, "overlap": None}
    tv = 0.5 * sum(abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total) for key in keys)
    overlap = sum(min(left.get(key, 0) / left_total, right.get(key, 0) / right_total) for key in keys)
    return {"total_variation": float(tv), "overlap": float(overlap)}


def numeric_compare(data: dict, left_ratio: str, right_ratio: str, numeric_fields: tuple[str, ...]) -> list[dict]:
    masks = ratio_masks(data["ratio_labels"])
    if left_ratio not in masks or right_ratio not in masks:
        return []
    metadata = data["metadata"]
    rows = []
    for field in numeric_fields:
        if field not in metadata:
            continue
        left = np.asarray(metadata[field][masks[left_ratio]], dtype=np.float64)
        right = np.asarray(metadata[field][masks[right_ratio]], dtype=np.float64)
        left_finite = left[np.isfinite(left)]
        right_finite = right[np.isfinite(right)]
        if left_finite.size == 0 or right_finite.size == 0:
            continue
        left_mean = float(np.mean(left_finite))
        right_mean = float(np.mean(right_finite))
        pooled = float(np.sqrt((np.var(left_finite) + np.var(right_finite)) / 2.0))
        rows.append({
            "split": data["split"],
            "field": field,
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "left_mean": left_mean,
            "right_mean": right_mean,
            "delta_right_minus_left": float(right_mean - left_mean),
            "standardized_mean_diff": float((right_mean - left_mean) / pooled) if pooled > 0 else None,
            "left_median": float(np.median(left_finite)),
            "right_median": float(np.median(right_finite)),
        })
    return rows


def categorical_compare(counters: dict, split: str, left_ratio: str, right_ratio: str, top_k: int) -> tuple[list[dict], list[dict]]:
    distances = []
    enrichments = []
    for field, by_ratio in counters.items():
        if left_ratio not in by_ratio or right_ratio not in by_ratio:
            continue
        left = by_ratio[left_ratio]
        right = by_ratio[right_ratio]
        distance = distribution_distance(left, right)
        distances.append({
            "split": split,
            "field": field,
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            **distance,
        })

        left_total = float(sum(left.values()))
        right_total = float(sum(right.values()))
        scored = []
        for value in set(left) | set(right):
            left_fraction = left.get(value, 0) / left_total if left_total else 0.0
            right_fraction = right.get(value, 0) / right_total if right_total else 0.0
            scored.append((abs(right_fraction - left_fraction), value, left_fraction, right_fraction))
        for _, value, left_fraction, right_fraction in sorted(scored, reverse=True)[:top_k]:
            enrichments.append({
                "split": split,
                "field": field,
                "value": value,
                "left_ratio": left_ratio,
                "right_ratio": right_ratio,
                "left_fraction": float(left_fraction),
                "right_fraction": float(right_fraction),
                "delta_right_minus_left": float(right_fraction - left_fraction),
            })
    return distances, enrichments


def write_tsv(path: Path, rows: list[dict], fieldnames: tuple[str, ...] | list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_ratio_summary(split: str, rows: list[dict], numeric_fields: tuple[str, ...]) -> list[dict]:
    flat = []
    for row in rows:
        out = {
            "split": split,
            "ratio": row["ratio"],
            "num_reads": row["num_reads"],
            "num_bags": row["num_bags"],
            "bag_size_mean": row["bag_size"]["mean"],
            "bag_size_median": row["bag_size"]["median"],
            "bag_size_min": row["bag_size"]["min"],
            "bag_size_p10": row["bag_size"]["p10"],
            "bag_size_p90": row["bag_size"]["p90"],
            "bag_size_max": row["bag_size"]["max"],
        }
        for field in numeric_fields:
            stats = row.get(field)
            if not stats:
                continue
            for stat_name in ("mean", "median", "std", "min", "p10", "p90", "max"):
                out[f"{field}_{stat_name}"] = stats.get(stat_name)
        flat.append(out)
    return flat


def parse_fields(value: str | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return defaults
    fields = tuple(item.strip() for item in value.split(",") if item.strip())
    return fields or defaults


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("directory", type=Path, help="LLP dataset directory containing train arrays and optional validation/.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "valid", "all"], default="all")
    parser.add_argument("--compare-ratios", default="50,75", help="Pair of ratio labels to compare, e.g. 50,75.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--categorical-fields", default=None, help="Comma-separated metadata fields to compare.")
    parser.add_argument("--numeric-fields", default=None, help="Comma-separated numeric fields to summarize.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.directory / "llp_dataset_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    categorical_fields = parse_fields(args.categorical_fields, DEFAULT_CATEGORICAL_FIELDS)
    numeric_fields = parse_fields(args.numeric_fields, DEFAULT_NUMERIC_FIELDS)
    compare_ratios = tuple(item.strip() for item in args.compare_ratios.split(",") if item.strip())
    if len(compare_ratios) != 2:
        raise ValueError("--compare-ratios must contain exactly two comma-separated labels, e.g. 50,75")
    left_ratio, right_ratio = compare_ratios

    splits = ["train", "valid"] if args.split == "all" else [args.split]
    available_splits = []
    for split in splits:
        if split == "valid" and not (args.directory / "validation").exists():
            continue
        available_splits.append(split)
    if not available_splits:
        raise ValueError(f"No requested splits found under {args.directory}")

    summary = {
        "directory": str(args.directory.resolve()),
        "splits": {},
        "compare_ratios": [left_ratio, right_ratio],
    }
    ratio_rows = []
    bag_rows = []
    category_rows = []
    numeric_compare_rows = []
    category_distance_rows = []
    category_enrichment_rows = []

    for split in available_splits:
        data = load_split(args.directory, split)
        ratio_summary = summarize_by_ratio(data, numeric_fields)
        split_bag_rows, _ = summarize_bags(data)
        split_category_rows, counters = categorical_summary(data, categorical_fields, args.top_k)
        split_numeric_compare = numeric_compare(data, left_ratio, right_ratio, numeric_fields)
        split_category_distances, split_category_enrichments = categorical_compare(
            counters,
            split,
            left_ratio,
            right_ratio,
            args.top_k,
        )

        summary["splits"][split] = {
            "directory": str(data["directory"].resolve()),
            "num_reads": int(data["bag_targets"].shape[0]),
            "num_bags": int(np.unique(data["bag_keys"]).size),
            "ratios": ratio_summary,
            "numeric_compare": split_numeric_compare,
            "categorical_distance": split_category_distances,
        }
        ratio_rows.extend(flatten_ratio_summary(split, ratio_summary, numeric_fields))
        bag_rows.extend({"split": split, **row} for row in split_bag_rows)
        category_rows.extend(split_category_rows)
        numeric_compare_rows.extend(split_numeric_compare)
        category_distance_rows.extend(split_category_distances)
        category_enrichment_rows.extend(split_category_enrichments)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_tsv(
        output_dir / "ratio_summary.tsv",
        ratio_rows,
        list(ratio_rows[0].keys()) if ratio_rows else ("split", "ratio"),
    )
    write_tsv(
        output_dir / "bag_summary.tsv",
        bag_rows,
        list(bag_rows[0].keys()) if bag_rows else ("split", "bag_key", "ratio", "num_reads"),
    )
    write_tsv(
        output_dir / "category_top.tsv",
        category_rows,
        ("split", "field", "ratio", "value", "count", "fraction"),
    )
    write_tsv(
        output_dir / "numeric_compare.tsv",
        numeric_compare_rows,
        (
            "split",
            "field",
            "left_ratio",
            "right_ratio",
            "left_mean",
            "right_mean",
            "delta_right_minus_left",
            "standardized_mean_diff",
            "left_median",
            "right_median",
        ),
    )
    write_tsv(
        output_dir / "category_distance.tsv",
        category_distance_rows,
        ("split", "field", "left_ratio", "right_ratio", "total_variation", "overlap"),
    )
    write_tsv(
        output_dir / "category_enrichment.tsv",
        category_enrichment_rows,
        (
            "split",
            "field",
            "value",
            "left_ratio",
            "right_ratio",
            "left_fraction",
            "right_fraction",
            "delta_right_minus_left",
        ),
    )

    concise = {
        "output_dir": str(output_dir.resolve()),
        "splits": {
            split: {
                "num_reads": item["num_reads"],
                "num_bags": item["num_bags"],
                "numeric_compare": item["numeric_compare"],
                "categorical_distance": item["categorical_distance"],
            }
            for split, item in summary["splits"].items()
        },
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
