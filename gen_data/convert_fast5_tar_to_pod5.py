#!/usr/bin/env python3
"""
Convert tarred FAST5 archives to POD5.

This wrapper is intended for public RNA002-era releases distributed as files
such as ``RNAAB089716.fast5.tar.gz.4``.  It extracts one archive at a time,
converts all FAST5 reads with the official POD5 Python API, and writes one POD5
file per input archive.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import fnmatch
import json
import os
import re
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path

import h5py
import numpy as np
import pod5 as p5
from tqdm import tqdm

try:
    import vbz_h5py_plugin  # noqa: F401
except ImportError:
    vbz_h5py_plugin = None

from pod5.tools.pod5_convert_from_fast5 import (
    convert_fast5_end_reason,
    convert_fast5_read,
    convert_run_info,
    decode_str,
    is_multi_read_fast5,
)


ARCHIVE_PATTERNS = ("*.fast5.tar.gz*", "*.fast5.tgz*", "*.tar.gz*", "*.tgz")
NESTED_ARCHIVE_PATTERNS = ("*.fast5.tar", "*.tar", "*.fast5.tar.gz", "*.tar.gz", "*.tgz")
DEFAULT_SOFTWARE_NAME = "TetraMod FAST5 archive to POD5 converter"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract .fast5.tar.gz shards and convert their FAST5 reads to POD5.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Archive files, FAST5 files, or directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true", help="Recursively search input directories for archives.")
    parser.add_argument("--archive-pattern", action="append", default=None, help="Glob pattern for archive discovery in directories.")
    parser.add_argument(
        "--fast5-member-pattern",
        action="append",
        default=None,
        help=(
            "Tar member glob used to select FAST5 files inside archives. "
            "Defaults to *.fast5; use '*' if archived FAST5 members lack a .fast5 suffix."
        ),
    )
    parser.add_argument("--extract-dir", type=Path, default=None, help="Directory for extracted FAST5 staging data.")
    parser.add_argument(
        "--max-nested-depth",
        type=int,
        default=3,
        help="Maximum recursive tar nesting depth to unpack inside input archives.",
    )
    parser.add_argument("--keep-extracted", action="store_true", help="Keep extracted FAST5 files after conversion.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing POD5 outputs.")
    parser.add_argument("--strict", action="store_true", help="Stop on the first unreadable FAST5 instead of skipping it.")
    parser.add_argument("--limit", type=int, default=0, help="Convert at most N input archives/files; 0 disables the limit.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of input archives/files to convert in parallel. Each input writes one independent POD5. "
            "Use 1 for detailed per-FAST5 progress within one large archive."
        ),
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--software-name", default=DEFAULT_SOFTWARE_NAME)
    return parser.parse_args()


def is_archive(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in {".fast5", ".pod5"}:
        return False
    try:
        return tarfile.is_tarfile(path)
    except OSError:
        return False


def discover_inputs(paths: list[Path], recursive: bool, patterns: tuple[str, ...], limit: int) -> list[Path]:
    discovered = []
    for input_path in paths:
        input_path = input_path.expanduser()
        if input_path.is_file():
            discovered.append(input_path)
            continue
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        for pattern in patterns:
            iterator = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
            discovered.extend(path for path in iterator if path.is_file())
        fast5_iterator = input_path.rglob("*.fast5") if recursive else input_path.glob("*.fast5")
        discovered.extend(path for path in fast5_iterator if path.is_file())

    unique = sorted({path.resolve() for path in discovered})
    if limit > 0:
        unique = unique[:limit]
    if not unique:
        raise FileNotFoundError(f"No FAST5 archives or .fast5 files found in: {paths}")
    return unique


def output_name_for_input(path: Path) -> str:
    name = path.name
    match = re.match(r"(?P<base>.+?)\.fast5\.tar\.gz\.(?P<part>\d+)$", name)
    if match:
        return f"{match.group('base')}.part{match.group('part')}.pod5"
    for suffix in (".fast5.tar.gz", ".fast5.tgz", ".tar.gz", ".tgz", ".fast5"):
        if name.endswith(suffix):
            return f"{name[:-len(suffix)]}.pod5"
    return f"{name}.pod5"


def member_matches(member_name: str, patterns: tuple[str, ...]) -> bool:
    name = member_name.lower()
    basename = Path(name).name
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(basename, pattern) for pattern in patterns)


def safe_extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, extract_root: Path) -> Path:
    root_resolved = extract_root.resolve()
    target = (extract_root / member.name).resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise ValueError(f"Unsafe archive member path in {archive.name}: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.extractfile(member) as src, target.open("wb") as dst:
        if src is None:
            raise ValueError(f"Unable to read archive member: {member.name}")
        shutil.copyfileobj(src, dst)
    return target


def safe_extract_fast5_archive(
    archive_path: Path,
    extract_root: Path,
    *,
    member_patterns: tuple[str, ...],
    max_nested_depth: int,
    show_progress: bool = True,
) -> list[Path]:
    archive_path = archive_path.resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        regular_members = [member for member in archive.getmembers() if member.isfile()]
        fast5_members = [member for member in regular_members if member_matches(member.name, member_patterns)]
        nested_members = [
            member
            for member in regular_members
            if max_nested_depth > 0 and member_matches(member.name, NESTED_ARCHIVE_PATTERNS)
        ]
        members = fast5_members + [member for member in nested_members if member not in fast5_members]
        iterator = tqdm(
            members,
            desc=f"extract:{archive_path.name}",
            unit="file",
            ascii=True,
            ncols=100,
            disable=not show_progress,
        )
        for member in iterator:
            target = safe_extract_member(archive, member, extract_root)
            if member_matches(member.name, member_patterns):
                extracted.append(target)
                continue
            if max_nested_depth > 0 and is_archive(target):
                nested_root = target.parent / f"{target.name}.extract"
                extracted.extend(
                    safe_extract_fast5_archive(
                        target,
                        nested_root,
                        member_patterns=member_patterns,
                        max_nested_depth=max_nested_depth - 1,
                        show_progress=show_progress,
                    )
                )
    if not extracted:
        examples = [member.name for member in regular_members[:20]]
        raise ValueError(
            f"No FAST5 members matching {member_patterns} were found inside archive: {archive_path}. "
            f"Regular file members observed={len(regular_members)}, examples={examples}. "
            "If these files are FAST5/HDF5 but lack a .fast5 suffix, rerun with --fast5-member-pattern '*'. "
            "If this is one piece of a split gzip/tar archive, concatenate all pieces first. "
            "If members are nested tar files, increase --max-nested-depth."
        )
    return sorted(extracted)


def attrs_dict(group) -> dict:
    if group is None:
        return {}
    return dict(group.attrs)


def attr_int(attrs: dict, name: str, default: int = 0) -> int:
    value = attrs.get(name, default)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return int(value)


def attr_float(attrs: dict, name: str, default: float = 0.0) -> float:
    value = attrs.get(name, default)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return float(value)


def single_fast5_raw_read_group(h5_file: h5py.File):
    reads_group = h5_file.get("Raw/Reads")
    if reads_group is None:
        return None
    read_names = list(reads_group.keys())
    if len(read_names) != 1:
        raise ValueError(f"Expected one Raw/Reads child in single-read FAST5, found {len(read_names)}")
    return reads_group[read_names[0]]


def run_info_for_single_fast5(h5_file: h5py.File, run_info_cache: dict[str, p5.RunInfo]) -> p5.RunInfo:
    channel_attrs = attrs_dict(h5_file.get("UniqueGlobalKey/channel_id"))
    tracking_attrs = attrs_dict(h5_file.get("UniqueGlobalKey/tracking_id"))
    context_attrs = attrs_dict(h5_file.get("UniqueGlobalKey/context_tags"))
    acq_id = decode_str(tracking_attrs.get("run_id", tracking_attrs.get("protocol_run_id", b"unknown_run")))
    if acq_id not in run_info_cache:
        digitisation = attr_float(channel_attrs, "digitisation", 8192.0)
        adc_min = -4096 if digitisation == 8192 else 0
        adc_max = 4095 if digitisation == 8192 else 2047
        device_type_guess = "minion" if digitisation == 8192 else "promethion"
        run_info_cache[acq_id] = convert_run_info(
            acq_id=acq_id,
            adc_max=adc_max,
            adc_min=adc_min,
            sample_rate=attr_int(channel_attrs, "sampling_rate", 4000),
            context_tags=context_attrs,
            device_type=device_type_guess,
            tracking_id=tracking_attrs,
        )
    return run_info_cache[acq_id]


def convert_single_fast5(path: Path, h5_file: h5py.File, run_info_cache: dict[str, p5.RunInfo]) -> p5.Read:
    raw_group = single_fast5_raw_read_group(h5_file)
    if raw_group is None:
        raise ValueError(f"{path} is not a supported single-read FAST5; missing Raw/Reads")
    if "Signal" not in raw_group:
        raise ValueError(f"{path} is not a supported single-read FAST5; missing Raw/Reads/*/Signal")

    raw_attrs = attrs_dict(raw_group)
    channel_attrs = attrs_dict(h5_file.get("UniqueGlobalKey/channel_id"))
    run_info = run_info_for_single_fast5(h5_file, run_info_cache)

    read_id_value = raw_attrs.get("read_id")
    if read_id_value is None:
        raise ValueError(f"{path} has no read_id attribute")
    read_id = uuid.UUID(decode_str(read_id_value))

    calibration = p5.Calibration.from_range(
        offset=attr_float(channel_attrs, "offset", 0.0),
        adc_range=attr_float(channel_attrs, "range", 1.0),
        digitisation=attr_float(channel_attrs, "digitisation", 8192.0),
    )
    pore = p5.Pore(
        channel=attr_int(channel_attrs, "channel_number", 0),
        well=attr_int(raw_attrs, "start_mux", 0),
        pore_type=decode_str(raw_attrs.get("pore_type", b"not_set")),
    )
    signal = np.asarray(raw_group["Signal"][()], dtype=np.int16)
    return p5.Read(
        read_id=read_id,
        pore=pore,
        calibration=calibration,
        read_number=attr_int(raw_attrs, "read_number", 0),
        start_sample=attr_int(raw_attrs, "start_time", 0),
        median_before=attr_float(raw_attrs, "median_before", 0.0),
        end_reason=convert_fast5_end_reason(attr_int(raw_attrs, "end_reason", 0)),
        run_info=run_info,
        num_minknow_events=attr_int(raw_attrs, "num_minknow_events", 0),
        num_reads_since_mux_change=attr_int(raw_attrs, "num_reads_since_mux_change", 0),
        time_since_mux_change=attr_float(raw_attrs, "time_since_mux_change", 0.0),
        signal=signal,
    )


def convert_fast5_file(path: Path, writer: p5.Writer, run_info_cache: dict[str, p5.RunInfo]) -> int:
    with h5py.File(path, "r") as h5_file:
        if is_multi_read_fast5(path):
            read_count = 0
            for group_name in h5_file:
                if not group_name.startswith("read_"):
                    continue
                read = convert_fast5_read(h5_file[group_name], run_info_cache)
                writer.add_read(read)
                read_count += 1
            return read_count

        writer.add_read(convert_single_fast5(path, h5_file, run_info_cache))
        return 1


def count_pod5_reads(path: Path) -> int:
    with p5.Reader(path) as reader:
        return sum(1 for _ in reader.reads())


def convert_fast5_paths_to_pod5(
    fast5_paths: list[Path],
    output_path: Path,
    *,
    software_name: str,
    strict: bool,
    show_progress: bool = True,
) -> dict:
    if output_path.exists():
        output_path.unlink()
    run_info_cache: dict[str, p5.RunInfo] = {}
    converted_reads = 0
    failed_files = []
    with p5.Writer(output_path, software_name=software_name) as writer:
        iterator = tqdm(
            fast5_paths,
            desc=f"convert:{output_path.name}",
            unit="fast5",
            ascii=True,
            ncols=100,
            disable=not show_progress,
        )
        for fast5_path in iterator:
            try:
                converted_reads += convert_fast5_file(fast5_path, writer, run_info_cache)
            except Exception as exc:
                if strict:
                    raise
                failed_files.append({"path": str(fast5_path), "error": str(exc)})
                print(f"[warning] skipped {fast5_path}: {exc}", flush=True)

    observed_reads = count_pod5_reads(output_path) if output_path.exists() else 0
    return {
        "output": str(output_path),
        "fast5_files": int(len(fast5_paths)),
        "converted_reads": int(converted_reads),
        "observed_pod5_reads": int(observed_reads),
        "failed_files": failed_files,
    }


def stage_fast5_inputs(
    input_path: Path,
    staging_root: Path,
    *,
    member_patterns: tuple[str, ...],
    max_nested_depth: int,
    show_progress: bool = True,
) -> tuple[list[Path], Path | None]:
    if is_archive(input_path):
        archive_stage = staging_root / input_path.name.replace(os.sep, "_")
        return (
            safe_extract_fast5_archive(
                input_path,
                archive_stage,
                member_patterns=member_patterns,
                max_nested_depth=max_nested_depth,
                show_progress=show_progress,
            ),
            archive_stage,
        )
    if input_path.is_file() and input_path.name.lower().endswith(".fast5"):
        return [input_path.resolve()], None
    raise ValueError(f"Unsupported input; expected tar archive or .fast5 file: {input_path}")


def convert_one_input(
    input_path: Path,
    output_dir: Path,
    extract_dir: Path | None,
    *,
    force: bool,
    keep_extracted: bool,
    software_name: str,
    strict: bool,
    member_patterns: tuple[str, ...],
    max_nested_depth: int,
    show_progress: bool,
) -> dict:
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name_for_input(input_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_path}")

    temp_owner = None
    try:
        if extract_dir is None:
            temp_owner = tempfile.TemporaryDirectory(prefix="tetramod_fast5_extract_", dir=str(output_dir))
            staging_root = Path(temp_owner.name)
        else:
            staging_root = extract_dir / output_path.stem
            staging_root.mkdir(parents=True, exist_ok=True)

        print(f"[convert] {input_path} -> {output_path}", flush=True)
        fast5_paths, extracted_dir = stage_fast5_inputs(
            input_path,
            staging_root,
            member_patterns=member_patterns,
            max_nested_depth=max_nested_depth,
            show_progress=show_progress,
        )
        result = convert_fast5_paths_to_pod5(
            fast5_paths,
            output_path,
            software_name=software_name,
            strict=bool(strict),
            show_progress=show_progress,
        )
        result["input"] = str(input_path)
        result["extracted_dir"] = None if extracted_dir is None else str(extracted_dir)
        print(
            f"[done] {output_path} reads={result['observed_pod5_reads']} "
            f"failed_fast5={len(result['failed_files'])}",
            flush=True,
        )
        if extracted_dir is not None and not keep_extracted:
            shutil.rmtree(extracted_dir, ignore_errors=True)
        return result
    finally:
        if temp_owner is not None and not keep_extracted:
            temp_owner.cleanup()


def validate_unique_outputs(inputs: list[Path], output_dir: Path) -> None:
    outputs = [output_dir / output_name_for_input(path) for path in inputs]
    counts = Counter(str(path) for path in outputs)
    duplicates = [path for path, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Multiple inputs would write the same POD5 output: {duplicates}")


def main():
    args = parse_args()
    patterns = tuple(args.archive_pattern) if args.archive_pattern else ARCHIVE_PATTERNS
    member_patterns = tuple(pattern.lower() for pattern in (args.fast5_member_pattern or ["*.fast5"]))
    max_nested_depth = max(0, int(args.max_nested_depth))
    inputs = discover_inputs(args.inputs, args.recursive, patterns, int(args.limit))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(args.jobs))
    show_progress = not args.no_progress
    validate_unique_outputs(inputs, args.output_dir)

    summary = {"inputs": [], "output_dir": str(args.output_dir.resolve()), "jobs": jobs}
    if jobs == 1:
        for input_path in inputs:
            summary["inputs"].append(
                convert_one_input(
                    input_path,
                    args.output_dir,
                    args.extract_dir,
                    force=bool(args.force),
                    keep_extracted=bool(args.keep_extracted),
                    software_name=args.software_name,
                    strict=bool(args.strict),
                    member_patterns=member_patterns,
                    max_nested_depth=max_nested_depth,
                    show_progress=show_progress,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    convert_one_input,
                    input_path,
                    args.output_dir,
                    args.extract_dir,
                    force=bool(args.force),
                    keep_extracted=bool(args.keep_extracted),
                    software_name=args.software_name,
                    strict=bool(args.strict),
                    member_patterns=member_patterns,
                    max_nested_depth=max_nested_depth,
                    show_progress=False,
                )
                for input_path in inputs
            ]
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="archives",
                unit="file",
                ascii=True,
                ncols=100,
                disable=not show_progress,
            )
            for future in iterator:
                summary["inputs"].append(future.result())

    summary_path = args.output_dir / "fast5_to_pod5_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
