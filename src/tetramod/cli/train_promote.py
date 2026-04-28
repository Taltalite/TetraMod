"""
Promoted training entry point that keeps the baseline train path intact.
"""

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib import import_module
from pathlib import Path

from tetramod.cli.train import (
    _check_device_available,
    extract_pretrained_encoder_config,
    load_pretrained_config,
    merge_pretrained_runtime_config,
    train_mod_default_config,
    validate_pretrained_runtime_config,
)


PROMOTE_HEAD_BASE = "A"
PROMOTE_HEAD_LABELS = ["canonical_A", "m6A"]


def _imports():
    import toml
    import torch

    from bonito.data import ComputeSettings, DataSettings, ModelSetup
    from tetramod.train_mod_data import load_train_mod_data
    from tetramod.training_promote import (
        CONTROL_WARMUP_LOSS_PATH,
        LLP_LOSS_PATH,
        PROMOTE_STAGE_CONTROL,
        PROMOTE_STAGE_LLP,
        TrainerPromote,
        resolve_llp_settings,
        resolve_promote_stage,
    )
    from tetramod.util import (
        STANDALONE_MOD_HEAD_MODE,
        init,
        load_pretrained_weights,
        load_symbol,
        resolve_model_dir,
    )

    return {
        "ComputeSettings": ComputeSettings,
        "CONTROL_WARMUP_LOSS_PATH": CONTROL_WARMUP_LOSS_PATH,
        "DataSettings": DataSettings,
        "LLP_LOSS_PATH": LLP_LOSS_PATH,
        "ModelSetup": ModelSetup,
        "PROMOTE_STAGE_CONTROL": PROMOTE_STAGE_CONTROL,
        "PROMOTE_STAGE_LLP": PROMOTE_STAGE_LLP,
        "STANDALONE_MOD_HEAD_MODE": STANDALONE_MOD_HEAD_MODE,
        "TrainerPromote": TrainerPromote,
        "init": init,
        "load_pretrained_weights": load_pretrained_weights,
        "load_symbol": load_symbol,
        "load_train_mod_data": load_train_mod_data,
        "resolve_llp_settings": resolve_llp_settings,
        "resolve_promote_stage": resolve_promote_stage,
        "resolve_model_dir": resolve_model_dir,
        "toml": toml,
        "torch": torch,
    }


def prepare_promote_config(config, *, promote_base=PROMOTE_HEAD_BASE):
    promote_base = str(promote_base).upper()
    if promote_base != PROMOTE_HEAD_BASE:
        raise ValueError(f"train_promote currently supports only {PROMOTE_HEAD_BASE}-head mode, got {promote_base!r}")

    model_cfg = dict(config.get("model", {}))
    mod_head_defs = dict(model_cfg.get("mod_head_defs", {}))
    head_labels = list(mod_head_defs.get(promote_base, PROMOTE_HEAD_LABELS))
    if not head_labels:
        raise ValueError(f"train_promote requires model.mod_head_defs.{promote_base} to be defined.")

    global_labels = list(model_cfg.get("mod_global_labels", []))
    missing = [label for label in head_labels if label not in global_labels]
    if missing:
        raise ValueError(
            "train_promote promote head labels must exist in model.mod_global_labels. "
            f"Missing: {missing}"
        )

    model_cfg["mod_bases"] = [promote_base]
    model_cfg["mod_head_defs"] = {promote_base: head_labels}
    config["model"] = model_cfg
    return config


def main(args):
    deps = _imports()
    toml = deps["toml"]
    torch = deps["torch"]
    resolve_model_dir = deps["resolve_model_dir"]

    workdir = os.path.expanduser(args.training_directory)
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        raise SystemExit(1)
    os.makedirs(workdir, exist_ok=True)

    _check_device_available(torch, args.device)
    deps["init"](args.seed, args.device, (not args.nondeterministic))
    device = torch.device(args.device)

    config = toml.load(args.config)
    config["__config_dir__"] = str(Path(args.config).resolve().parent)
    pretrained_config = load_pretrained_config(args.pretrained)
    promote_stage = deps["resolve_promote_stage"](config, args.promote_stage)
    llp_settings = {}
    if promote_stage == deps["PROMOTE_STAGE_LLP"]:
        llp_settings = deps["resolve_llp_settings"](
            config,
            cli_proportion=args.llp_proportion,
            cli_bag_size=args.llp_bag_size,
        )
    pretrained_encoder = config.get("model", {}).get("pretrained_encoder")
    if pretrained_encoder is None:
        pretrained_encoder = extract_pretrained_encoder_config(pretrained_config)
        if not pretrained_encoder:
            raise ValueError(
                "train_promote requires a pretrained basecaller with model.encoder or encoder in config.toml "
                "so the frozen encoder can be reconstructed."
            )
        config.setdefault("model", {})["pretrained_encoder"] = pretrained_encoder

    config = merge_pretrained_runtime_config(config, pretrained_config)
    config = prepare_promote_config(config, promote_base=args.promote_base)
    validate_pretrained_runtime_config(config, pretrained_config)
    loss_path = deps["LLP_LOSS_PATH"] if promote_stage == deps["PROMOTE_STAGE_LLP"] else deps["CONTROL_WARMUP_LOSS_PATH"]

    training_cfg = {
        **config.get("training", {}),
        **vars(args),
        **llp_settings,
        "pwd": os.getcwd(),
        "mode": deps["STANDALONE_MOD_HEAD_MODE"],
        "pipeline": "promote",
        "promote_stage": promote_stage,
        "loss_path": loss_path,
        "pretrained": args.pretrained,
        "pretrained_basecaller": args.pretrained,
        "pretrained_basecaller_dir": resolve_model_dir(args.pretrained),
        "mod_head_weights_pattern": "weights_{epoch}.tar",
        "promote_base": args.promote_base,
    }
    config["training"] = training_cfg

    print("[loading promoted model]")
    model = deps["load_symbol"](config, "Model")(config)
    validate_pretrained_runtime_config(config, pretrained_config, model=model)
    preload_stats = deps["load_pretrained_weights"](model, args.pretrained, device)

    try:
        model = torch.compile(model)
    except RuntimeError as exc:
        print(f"[warning] Torch model failed to compile, performance may be degraded. {exc}")

    print("[loading data]")
    data = deps["DataSettings"](
        training_data=args.directory,
        num_train_chunks=args.chunks,
        num_valid_chunks=args.valid_chunks,
        output_dir=workdir,
    )
    model_setup = deps["ModelSetup"](
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=config.get("standardisation", {}),
    )
    compute_settings = deps["ComputeSettings"](
        batch_size=args.batch,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    train_loader, valid_loader = deps["load_train_mod_data"](data, model_setup, compute_settings)

    try:
        dataset_cfg = train_loader.dataset.dataset_config
    except AttributeError:
        dataset_cfg = {}
    if preload_stats:
        config["training"]["pretrained_weights"] = preload_stats
    with open(os.path.join(workdir, "config.toml"), "w") as config_fh:
        toml.dump({**config, **dataset_cfg}, config_fh)

    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    trainer = deps["TrainerPromote"](
        model,
        device,
        train_loader,
        valid_loader,
        promote_stage=promote_stage,
        llp_proportion=training_cfg.get("llp_proportion"),
        llp_bag_size=training_cfg.get("llp_bag_size"),
        use_amp=not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split,
        quantile_grad_clip=args.quantile_grad_clip,
        chunks_per_epoch=args.chunks,
        batch_size=args.batch,
        profile_flush_chunks=args.profile_chunks,
    )

    if "," in args.lr:
        lr = [float(x) for x in args.lr.split(",")]
    else:
        lr = float(args.lr)
    optim_kwargs = config.get("optim", {})
    trainer.fit(workdir, args.epochs, lr, **optim_kwargs)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("training_directory")
    parser.add_argument("--config", default=str(train_mod_default_config))
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default="2e-3")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", type=int, help="Number of training chunks per epoch")
    parser.add_argument("--valid-chunks", type=int, help="Number of validation chunks per epoch")
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    quantile_group = parser.add_mutually_exclusive_group()
    quantile_group.add_argument("--quantile-grad-clip", dest="quantile_grad_clip", action="store_true")
    quantile_group.add_argument("--no-quantile-grad-clip", dest="quantile_grad_clip", action="store_false")
    quantile_group.set_defaults(quantile_grad_clip=True)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--profile-chunks",
        default=10000,
        type=int,
        help="Flush training profiling stats every N chunks; set 0 to disable.",
    )
    parser.add_argument(
        "--promote-base",
        default=PROMOTE_HEAD_BASE,
        choices=[PROMOTE_HEAD_BASE],
        help="Promoted training currently supports only A-head mode.",
    )
    parser.add_argument(
        "--promote-stage",
        default=None,
        choices=["control", "llp"],
        help="Promoted training stage. Defaults to config training.promote_stage or control.",
    )
    parser.add_argument(
        "--llp-proportion",
        default=None,
        type=float,
        help="Known modified proportion for LLP bags. Accepts a fraction in [0, 1] or percent in [0, 100].",
    )
    parser.add_argument(
        "--llp-bag-size",
        default=None,
        type=int,
        help=(
            "LLP bag key grouping. 0 groups valid reads in each batch into one bag; "
            "N>0 uses floor(sample_key / N)."
        ),
    )
    return parser
