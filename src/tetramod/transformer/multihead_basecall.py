"""
Multi-head basecalling helpers that produce standard Bonito result dictionaries.
"""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
from koi.decode import beam_search, to_str

from bonito.crf.basecall import stitch_results
from bonito.multiprocessing import thread_iter
from tetramod.util import batchify, chunk, unbatchify


MOD_CODE_BY_LABEL = {
    "m6A": "a",
    "5mC": "m",
    "5hmC": "h",
}


def _profile_add(profile: Dict[str, float] | None, key: str, value: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + float(value)


def decode_scores(
    scores,
    seqdist,
    beam_width=32,
    beam_cut=100.0,
    scale=1.0,
    offset=0.0,
    blank_score=2.0,
    reverse=False,
):
    """
    Decode precomputed CRF scores with Bonito's beam search.
    """
    if reverse:
        scores = seqdist.reverse_complement(scores)
    with torch.cuda.device(scores.device):
        sequence, qstring, moves = beam_search(
            scores,
            beam_width=beam_width,
            beam_cut=beam_cut,
            scale=scale,
            offset=offset,
            blank_score=blank_score,
        )
    return {
        "moves": moves,
        "qstring": qstring,
        "sequence": sequence,
    }


def _decode_basecall_batch(model, base_scores: torch.Tensor, reverse: bool = False) -> Dict[str, object]:
    if base_scores.ndim != 3:
        raise ValueError(f"Expected base_scores with 3 dims, got shape {tuple(base_scores.shape)}")
    if not str(base_scores.device).startswith("cuda"):
        raise RuntimeError(
            "basecaller_mod currently requires a CUDA device for beam-search basecalling. "
            "Use a CUDA device or validate with CLI/import smoke checks only."
        )
    raw_score_size = model.seqdist.n_base ** (model.seqdist.state_len + 1)
    if base_scores.shape[-1] != raw_score_size:
        raise RuntimeError(
            "basecaller_mod received expanded CRF scores that are incompatible with "
            "koi beam-search decoding. Rerun with --use-koi."
        )
    base_scores = base_scores.permute(1, 0, 2).contiguous()
    return decode_scores(base_scores, model.seqdist, reverse=reverse)


def _run_model_on_batch(
    model,
    batch: torch.Tensor,
    reverse: bool = False,
    *,
    emit_mods: bool = True,
    profile: Dict[str, float] | None = None,
) -> Dict[str, object]:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    t0 = perf_counter()
    with torch.inference_mode():
        outputs = model(batch.to(device=device, dtype=model_dtype, non_blocking=True))
    _profile_add(profile, "model_forward_s", perf_counter() - t0)

    t0 = perf_counter()
    basecall_attrs = _decode_basecall_batch(model, outputs["base_scores"], reverse=reverse)
    _profile_add(profile, "beam_search_s", perf_counter() - t0)
    result = {"basecall_attrs": basecall_attrs}
    if not emit_mods:
        return result

    _profile_add(profile, "model_batches", 1.0)
    return {
        **result,
        "model_outputs": {
            "mod_logits_by_base": {
                head_name: logits.detach().contiguous()
                for head_name, logits in outputs["mod_logits_by_base"].items()
            },
        },
    }


def _format_basecall_result(stride: int, attrs: Dict[str, object], rna: bool = False) -> Dict[str, object]:
    flip = (lambda x: x[::-1]) if rna else (lambda x: x)
    sequence = to_str(attrs["sequence"])
    qstring = to_str(attrs["qstring"])
    moves = attrs["moves"]
    if isinstance(moves, torch.Tensor):
        moves = moves.detach().cpu().numpy()
    elif not isinstance(moves, np.ndarray):
        moves = np.asarray(moves)

    return {
        "stride": stride,
        "moves": moves,
        "qstring": flip(qstring),
        "sequence": flip(sequence),
    }


def _build_mod_tags(sequence: str, mapped_sites: List[Dict[str, object]]) -> List[str]:
    grouped: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)

    for site in mapped_sites:
        label = str(site.get("global_pred_label", ""))
        if label.startswith("canonical_"):
            continue
        mod_code = MOD_CODE_BY_LABEL.get(label)
        if mod_code is None:
            continue

        base_index = int(site["base_index"])
        if base_index < 0 or base_index >= len(sequence):
            continue
        base_char = str(sequence[base_index]).upper()
        if base_char not in {"A", "C", "G", "T", "U"}:
            continue

        prob = min(max(float(site["probability"]), 0.0), 1.0)
        grouped[(base_char, mod_code)].append((base_index, int(round(prob * 255.0))))

    if not grouped:
        return []

    mm_parts = []
    ml_values: List[str] = []
    for (base_char, mod_code), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item[0])
        base_positions = [idx for idx, char in enumerate(sequence) if str(char).upper() == base_char]
        if not base_positions:
            continue
        occurrence_index = {pos: idx for idx, pos in enumerate(base_positions)}

        deltas = []
        previous = -1
        for pos, prob in items:
            occurrence = occurrence_index.get(pos)
            if occurrence is None:
                continue
            deltas.append(str(occurrence if previous < 0 else occurrence - previous - 1))
            previous = occurrence
            ml_values.append(str(prob))

        if deltas:
            mm_parts.append(f"{base_char}+{mod_code}.,{','.join(deltas)};")

    if not mm_parts:
        return []

    return [
        f"MM:Z:{''.join(mm_parts)}",
        f"ML:B:C,{','.join(ml_values)}",
    ]


def _moves_array(attrs: Dict[str, object]) -> np.ndarray:
    moves = attrs["moves"]
    if isinstance(moves, torch.Tensor):
        return moves.detach().cpu().numpy()
    if isinstance(moves, np.ndarray):
        return moves
    return np.asarray(moves)


def _head_label_for_local_pred(model, head_name: str, local_idx: int) -> str | None:
    try:
        global_id = model.head_global_ids[head_name][int(local_idx)]
        return str(model.mod_global_labels[int(global_id)])
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return None


def _mod_sites_from_logits(
    model,
    attrs: Dict[str, object],
    mod_logits_by_base: Dict[str, torch.Tensor],
    *,
    rna: bool = False,
    mod_threshold: float = 0.5,
) -> List[Dict[str, object]]:
    raw_sequence = to_str(attrs["sequence"])
    sequence_length = len(raw_sequence)
    if sequence_length == 0:
        return []

    moves = _moves_array(attrs)
    emit_positions = np.flatnonzero(moves)
    usable = min(sequence_length, int(emit_positions.shape[0]))
    if usable == 0:
        return []

    sites_by_head: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for raw_idx in range(usable):
        base_label = raw_sequence[raw_idx].upper()
        try:
            head_name = model._base_slot_for_label(base_label)
        except AttributeError:
            head_name = base_label if base_label in mod_logits_by_base else None
        if head_name is None or head_name not in mod_logits_by_base:
            continue
        base_index = sequence_length - 1 - raw_idx if rna else raw_idx
        sites_by_head[head_name].append((base_index, int(emit_positions[raw_idx]), base_label))

    mapped_sites: List[Dict[str, object]] = []
    for head_name, site_specs in sites_by_head.items():
        logits = mod_logits_by_base[head_name]
        if logits.ndim != 2 or logits.shape[-1] <= 1:
            continue

        time_indices = torch.tensor(
            [time_idx for _, time_idx, _ in site_specs],
            device=logits.device,
            dtype=torch.long,
        )
        selected_logits = logits.index_select(0, time_indices).to(torch.float32)
        probs = torch.softmax(selected_logits, dim=-1)
        if probs.shape[-1] == 2:
            local_preds = torch.where(
                probs[:, 1] >= float(mod_threshold),
                torch.ones_like(time_indices),
                torch.zeros_like(time_indices),
            )
        else:
            local_preds = probs.argmax(dim=-1)
        pred_probs = probs.gather(1, local_preds.unsqueeze(1)).squeeze(1)

        for spec_idx, (base_index, _time_idx, _base_label) in enumerate(site_specs):
            local_pred = int(local_preds[spec_idx].item())
            if local_pred == 0:
                continue
            probability = float(pred_probs[spec_idx].item())
            if probability < float(mod_threshold):
                continue
            label = _head_label_for_local_pred(model, head_name, local_pred)
            if label is None or MOD_CODE_BY_LABEL.get(label) is None:
                continue
            mapped_sites.append({
                "base_index": int(base_index),
                "global_pred_label": label,
                "probability": probability,
            })

    return mapped_sites


def _result_from_stitched_outputs(
    model,
    stitched_batch_result: Dict[str, object],
    *,
    rna: bool = False,
    mod_threshold: float = 0.5,
    emit_mods: bool = True,
    profile: Dict[str, float] | None = None,
) -> Dict[str, object]:
    t0 = perf_counter()
    base_result = _format_basecall_result(model.stride, stitched_batch_result["basecall_attrs"], rna=rna)
    _profile_add(profile, "format_basecall_s", perf_counter() - t0)
    if not emit_mods:
        return base_result

    t0 = perf_counter()
    mapped_sites = _mod_sites_from_logits(
        model,
        stitched_batch_result["basecall_attrs"],
        stitched_batch_result["model_outputs"]["mod_logits_by_base"],
        rna=rna,
        mod_threshold=mod_threshold,
    )
    mods = _build_mod_tags(base_result["sequence"], mapped_sites)
    _profile_add(profile, "mod_tag_s", perf_counter() - t0)
    _profile_add(profile, "reads_postprocessed", 1.0)
    return {**base_result, "mods": mods}


def basecall(
    model,
    reads: Iterable[object],
    chunksize: int = 4000,
    overlap: int = 100,
    batchsize: int = 32,
    reverse: bool = False,
    rna: bool = False,
    mod_threshold: float = 0.5,
    emit_mods: bool = True,
    profile: Dict[str, float] | None = None,
) -> Iterator[Tuple[object, Dict[str, object]]]:
    chunks = thread_iter(
        ((read, 0, read.signal.shape[-1]), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )
    batches = thread_iter(batchify(chunks, batchsize=batchsize))
    batch_results = thread_iter(
        (
            read_info,
            _run_model_on_batch(
                model,
                batch,
                reverse=reverse,
                emit_mods=emit_mods,
                profile=profile,
            ),
        )
        for read_info, batch in batches
    )
    per_read_results = thread_iter(
        (
            read,
            {
                "basecall_attrs": stitch_results(
                    outputs["basecall_attrs"],
                    end - start,
                    chunksize,
                    overlap,
                    model.stride,
                    reverse=False,
                ),
                **(
                    {
                        "model_outputs": stitch_results(
                            outputs["model_outputs"],
                            end - start,
                            chunksize,
                            overlap,
                            model.stride,
                            reverse=False,
                        )
                    }
                    if emit_mods
                    else {}
                ),
            },
        )
        for ((read, start, end), outputs) in unbatchify(batch_results)
    )
    return thread_iter(
        (
            read,
            _result_from_stitched_outputs(
                model,
                stitched_outputs,
                rna=rna,
                mod_threshold=mod_threshold,
                emit_mods=emit_mods,
                profile=profile,
            ),
        )
        for read, stitched_outputs in per_read_results
    )
