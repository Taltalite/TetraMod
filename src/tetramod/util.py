"""
Small compatibility layer for Bonito helpers needed by TetraMod's migrated
modified-base commands.
"""

from importlib import import_module
from pathlib import Path
import importlib.util
import os
import sys

import toml
import torch

from bonito.util import (
    __models_dir__,
    accuracy,
    batchify,
    chunk,
    column_to_set,
    decode_ref,
    get_last_checkpoint,
    init,
    load_object,
    match_names,
    permute,
    set_config_defaults,
    tqdm_environ,
    unbatchify,
)


STANDALONE_MOD_HEAD_MODE = "standalone_mod_head"
_PACKAGE_ALIASES = {
    "bonito.transformer.multihead_model": "tetramod.transformer.multihead_model",
}


def load_symbol(config, symbol):
    """
    Dynamic load a symbol from module specified in a model config.
    """
    if not isinstance(config, dict):
        dirname = resolve_model_dir(config)
        config = toml.load(os.path.join(dirname, "config.toml"))
        config["__config_dir__"] = str(Path(dirname).resolve())

    model_config = config.get("model", {})
    model_file = model_config.get("file")
    if model_file:
        config_dir = config.get("__config_dir__")
        if config_dir and not os.path.isabs(model_file):
            model_file = os.path.join(config_dir, model_file)
        else:
            model_file = os.path.abspath(model_file)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        module_name = f"tetramod.model_file.{Path(model_file).stem}"
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(f"Unable to load model file: {model_file}")
        spec.loader.exec_module(module)
        return getattr(module, symbol)

    package = _PACKAGE_ALIASES.get(model_config["package"], model_config["package"])
    imported = import_module(package)
    return getattr(imported, symbol)


def resolve_model_dir(dirname):
    dirname = os.path.expanduser(str(dirname))
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models_dir__, dirname)):
        dirname = os.path.join(__models_dir__, dirname)
    return dirname


def strip_module_prefix(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def get_training_mode(config):
    return str(config.get("training", {}).get("mode", "")).strip()


def is_standalone_mod_head_config(config):
    return get_training_mode(config) == STANDALONE_MOD_HEAD_MODE


def get_standalone_pretrained_basecaller(config):
    training_cfg = config.get("training", {})
    for key in ("pretrained_basecaller", "pretrained_basecaller_dir", "pretrained"):
        value = training_cfg.get(key)
        if value:
            return value
    return None


def load_matching_weights(model, state_dict, *, device=None, allow_remap=True):
    state_dict = strip_module_prefix(state_dict)
    model_state = model.state_dict()
    matched = 0
    remapped = None

    if allow_remap and set(state_dict.keys()) != set(model_state.keys()):
        try:
            remapped = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()}
        except AssertionError:
            remapped = None

    to_load = remapped if remapped is not None else state_dict
    for name, value in to_load.items():
        if name in model_state and model_state[name].shape == value.shape:
            model_state[name] = value
            matched += 1

    model.load_state_dict(model_state)
    skipped = len(to_load) - matched
    stats = {"matched": matched, "skipped": skipped}
    if device is not None:
        stats["device"] = str(device)
    return stats


def load_pretrained_weights(model, pretrained, device):
    dirname = resolve_model_dir(pretrained)
    weights = get_last_checkpoint(dirname)
    sys.stderr.write(f"[loading pretrained weights] - {weights}\n")
    state_dict = torch.load(weights, map_location=device, weights_only=False)
    stats = load_matching_weights(model, state_dict, device=device)
    stats["path"] = str(weights)
    sys.stderr.write(
        f"[loading pretrained weights] - matched={stats['matched']} skipped={stats['skipped']}\n"
    )
    if stats["matched"] == 0:
        sys.stderr.write("[warning] No pretrained weights matched current model parameters.\n")
    return stats


def load_model(
    dirname,
    device,
    weights=None,
    half=True,
    chunksize=None,
    batchsize=None,
    overlap=None,
    quantize=False,
    use_koi=False,
    compile=True,
):
    """
    Load a Bonito model config and weights from disk.
    """
    dirname = resolve_model_dir(dirname)
    weights = get_last_checkpoint(dirname) if weights is None else os.path.join(dirname, "weights_%s.tar" % weights)
    config = toml.load(os.path.join(dirname, "config.toml"))
    config["__config_dir__"] = str(Path(dirname).resolve())
    config = set_config_defaults(config, chunksize, batchsize, overlap, quantize)
    return _load_model(weights, config, device, half, use_koi, compile)


def _load_model(model_file, config, device, half=True, use_koi=False, compile=True):
    device = torch.device(device)
    model = load_symbol(config, "Model")(config)
    standalone_mod_head = is_standalone_mod_head_config(config)

    if standalone_mod_head:
        pretrained = get_standalone_pretrained_basecaller(config)
        if not pretrained:
            raise ValueError(
                "Standalone mod-head configs must record training.pretrained_basecaller "
                "so the frozen basecaller can be reconstructed."
            )
        load_pretrained_weights(model, pretrained, device)

    if use_koi:
        config["basecaller"]["chunksize"] -= config["basecaller"]["chunksize"] % model.stride
        config["basecaller"]["overlap"] -= config["basecaller"]["overlap"] % (model.stride * 2)
        model.use_koi(
            batchsize=config["basecaller"]["batchsize"],
            chunksize=config["basecaller"]["chunksize"],
            quantize=config["basecaller"]["quantize"],
        )

    state_dict = torch.load(model_file, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(state_dict)
    if standalone_mod_head and hasattr(model, "load_checkpoint_state_dict"):
        model.load_checkpoint_state_dict(state_dict)
    else:
        model_state = model.state_dict()
        if set(state_dict.keys()) == set(model_state.keys()):
            model.load_state_dict(state_dict)
        else:
            try:
                remapped = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()}
            except AssertionError as exc:
                raise ValueError(
                    "Model weights do not match the current model architecture. "
                    "Ensure the config.toml and weights_*.tar are from the same model."
                ) from exc
            model.load_state_dict(remapped)

    if half:
        model = model.half()
    model.eval()
    model.to(device)

    if compile:
        try:
            model = torch.compile(model)
        except RuntimeError as exc:
            sys.stderr.write(
                f"[warning] Torch model failed to compile, performance may be degraded. {exc}\n"
            )

    return model
