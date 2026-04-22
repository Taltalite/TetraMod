"""
TetraMod entry point for the Bonito-derived basecaller_mod handler.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import timedelta
from itertools import islice as take
from time import perf_counter
import sys


def _imports():
    import numpy as np
    import torch
    from tqdm import tqdm

    from bonito.aligner import Aligner, align_map
    from bonito.cli.download import Downloader, __models_dir__, models
    from bonito.io import Writer, biofmt
    from bonito.multiprocessing import process_cancel
    from bonito.nn import fuse_bn_
    from bonito.reader import Reader
    from tetramod.transformer.multihead_basecall import basecall as basecall_mod
    from tetramod.util import column_to_set, init, load_model, tqdm_environ

    return {
        "Aligner": Aligner,
        "Downloader": Downloader,
        "Reader": Reader,
        "Writer": Writer,
        "__models_dir__": __models_dir__,
        "align_map": align_map,
        "basecall_mod": basecall_mod,
        "biofmt": biofmt,
        "column_to_set": column_to_set,
        "fuse_bn_": fuse_bn_,
        "init": init,
        "load_model": load_model,
        "models": models,
        "np": np,
        "process_cancel": process_cancel,
        "tqdm": tqdm,
        "tqdm_environ": tqdm_environ,
        "torch": torch,
    }


def _check_cuda_available(torch, device):
    if str(device).lower() == "cpu":
        raise SystemExit(
            "tetramod basecaller currently requires a CUDA device for modified-base "
            "beam-search decoding; --device cpu is not supported for this command."
        )
    if torch.cuda.is_available():
        return
    cuda_build = torch.version.cuda or "unknown"
    raise SystemExit(
        "tetramod basecaller requested CUDA, but PyTorch cannot initialize CUDA. "
        f"Installed torch={torch.__version__} reports CUDA build={cuda_build}. "
        "Install a PyTorch CUDA build compatible with your NVIDIA driver before running "
        "modified-base basecalling."
    )


def main(args):
    deps = _imports()
    _check_cuda_available(deps["torch"], args.device)

    if args.revcomp:
        sys.stderr.write(
            "> error: basecaller_mod stage-two minimal path does not support --revcomp; "
            "rerun without this flag\n"
        )
        raise SystemExit(1)

    try:
        reader = deps["Reader"](args.reads_directory, args.recursive)
        sys.stderr.write("> reading %s\n" % reader.fmt)
    except FileNotFoundError:
        sys.stderr.write("> error: no suitable files found in %s\n" % args.reads_directory)
        raise SystemExit(1)

    deps["init"](args.seed, args.device)

    fmt = deps["biofmt"](aligned=args.reference is not None)

    if args.reference and args.reference.endswith(".mmi") and fmt.name == "cram":
        sys.stderr.write("> error: reference cannot be a .mmi when outputting cram\n")
        raise SystemExit(1)
    elif args.reference and fmt.name == "fastq":
        sys.stderr.write(f"> warning: did you really want {fmt.aligned} {fmt.name}?\n")
    else:
        sys.stderr.write(f"> outputting {fmt.aligned} {fmt.name}\n")

    if args.model_directory in deps["models"] and not (deps["__models_dir__"] / args.model_directory).exists():
        sys.stderr.write("> downloading model\n")
        deps["Downloader"](deps["__models_dir__"]).download(args.model_directory)

    sys.stderr.write(f"> loading model {args.model_directory}\n")
    try:
        model = deps["load_model"](
            args.model_directory,
            args.device,
            weights=args.weights if args.weights > 0 else None,
            chunksize=args.chunksize,
            overlap=args.overlap,
            batchsize=args.batchsize,
            quantize=args.quantize,
            use_koi=args.use_koi,
        )
        model = model.apply(deps["fuse_bn_"])
    except FileNotFoundError:
        sys.stderr.write(f"> error: failed to load {args.model_directory}\n")
        sys.stderr.write("> available models:\n")
        for model_name in sorted(deps["models"]):
            sys.stderr.write(f" - {model_name}\n")
        raise SystemExit(1)

    if not hasattr(model, "predict_mods"):
        sys.stderr.write(
            "> error: loaded model does not provide predict_mods(); "
            "basecaller_mod expects a multi-head modified-base model\n"
        )
        raise SystemExit(1)

    if args.verbose:
        sys.stderr.write(f"> model basecaller params: {model.config['basecaller']}\n")
        sys.stderr.write(f"> koi {'enabled' if args.use_koi else 'disabled'}\n")

    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = deps["Aligner"](args.reference, preset=args.mm2_preset)
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            raise SystemExit(1)
    else:
        aligner = None

    if fmt.name != "fastq":
        groups, num_reads = reader.get_read_groups(
            args.reads_directory,
            args.model_directory,
            n_proc=8,
            recursive=args.recursive,
            read_ids=deps["column_to_set"](args.read_ids),
            skip=args.skip,
            cancel=deps["process_cancel"](),
        )
    else:
        groups = []
        num_reads = None

    reads = reader.get_reads(
        args.reads_directory,
        n_proc=8,
        recursive=args.recursive,
        read_ids=deps["column_to_set"](args.read_ids),
        skip=args.skip,
        do_trim=not args.no_trim,
        scaling_strategy=model.config.get("scaling"),
        norm_params=(
            model.config.get("standardisation")
            if (
                model.config.get("scaling")
                and model.config.get("scaling").get("strategy") == "pa"
            )
            else model.config.get("normalisation")
        ),
        cancel=deps["process_cancel"](),
    )

    if args.verbose:
        sys.stderr.write(f"> read scaling: {model.config.get('scaling')}\n")

    if args.max_reads:
        reads = take(reads, args.max_reads)
        if num_reads is not None:
            num_reads = min(num_reads, args.max_reads)

    results = deps["basecall_mod"](
        model,
        reads,
        reverse=False,
        rna=args.rna,
        batchsize=model.config["basecaller"]["batchsize"],
        chunksize=model.config["basecaller"]["chunksize"],
        overlap=model.config["basecaller"]["overlap"],
        mod_threshold=args.mod_threshold,
    )

    if aligner:
        results = deps["align_map"](aligner, results, n_thread=args.alignment_threads)

    writer = deps["Writer"](
        fmt.mode,
        deps["tqdm"](
            results,
            desc="> calling",
            unit=" reads",
            leave=False,
            total=num_reads,
            smoothing=0,
            ascii=True,
            ncols=100,
            **deps["tqdm_environ"](),
        ),
        aligner=aligner,
        group_key=args.model_directory,
        ref_fn=args.reference,
        groups=groups,
        min_qscore=args.min_qscore,
    )

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    np = deps["np"]
    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default=0, type=int)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--revcomp", action="store_true", default=False)
    parser.add_argument("--rna", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    quant_parser = parser.add_mutually_exclusive_group(required=False)
    quant_parser.add_argument("--quantize", dest="quantize", action="store_true")
    quant_parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    koi_parser = parser.add_mutually_exclusive_group(required=False)
    koi_parser.add_argument("--use-koi", dest="use_koi", action="store_true")
    koi_parser.add_argument("--no-use-koi", dest="use_koi", action="store_false")
    parser.set_defaults(use_koi=True)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--chunksize", default=None, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--max-reads", default=0, type=int)
    parser.add_argument("--min-qscore", default=0, type=int)
    parser.add_argument("--alignment-threads", default=8, type=int)
    parser.add_argument("--mm2-preset", default="lr:hq", type=str)
    parser.add_argument("--mod-threshold", default=0.5, type=float)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser
