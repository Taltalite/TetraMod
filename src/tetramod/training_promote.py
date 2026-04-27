"""
Promoted trainer path kept separate from the protected baseline trainer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tetramod.training_mod import TrainerMod


PROMOTE_STAGE_CONTROL = "control"
PROMOTE_STAGE_LLP = "llp"
CONTROL_WARMUP_LOSS_PATH = "a_head_control_warmup_viterbi_bce"
LLP_LOSS_PATH = "a_head_llp_mean_pool_proportion_bce"
DEFAULT_LLP_BAG_SIZE = 0


def resolve_promote_stage(config, cli_stage: str | None = None) -> str:
    stage = cli_stage or config.get("training", {}).get("promote_stage") or PROMOTE_STAGE_CONTROL
    stage = str(stage).strip().lower()
    if stage not in {PROMOTE_STAGE_CONTROL, PROMOTE_STAGE_LLP}:
        raise ValueError(
            "train_promote currently supports promote_stage="
            f"{PROMOTE_STAGE_CONTROL!r} or {PROMOTE_STAGE_LLP!r}, got {stage!r}"
        )
    return stage


def normalize_llp_proportion(proportion) -> float:
    if proportion is None:
        raise ValueError("LLP training requires --llp-proportion or training.llp_proportion.")

    value = float(proportion)
    if value > 1.0:
        value = value / 100.0
    if value < 0.0 or value > 1.0:
        raise ValueError(f"LLP proportion must be in [0, 1] or [0, 100], got {proportion!r}")
    return value


def resolve_llp_settings(config, cli_proportion=None, cli_bag_size=None) -> dict:
    training_cfg = config.get("training", {})
    proportion = cli_proportion if cli_proportion is not None else training_cfg.get("llp_proportion")
    bag_size = cli_bag_size if cli_bag_size is not None else training_cfg.get("llp_bag_size", DEFAULT_LLP_BAG_SIZE)
    bag_size = int(bag_size)
    if bag_size < 0:
        raise ValueError(f"LLP bag size must be >= 0, got {bag_size}")

    settings = {"llp_bag_size": bag_size}
    if proportion is not None:
        settings["llp_proportion"] = normalize_llp_proportion(proportion)
    return settings


def binary_cross_entropy_on_probabilities(input_probs, targets):
    """
    BCE for already-aggregated probabilities.

    PyTorch disallows F.binary_cross_entropy under autocast. LLP bag scores are
    mean probabilities rather than logits, so compute the probability-space BCE
    explicitly in fp32 while preserving gradients to the underlying logits.
    """
    probs = input_probs.to(dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    targets = targets.to(device=probs.device, dtype=torch.float32)
    return -(targets * probs.log() + (1.0 - targets) * (1.0 - probs).log()).mean()


def validate_control_warmup_model(model) -> None:
    mod_bases = list(getattr(model, "mod_bases", []))
    if mod_bases != ["A"]:
        raise ValueError(f"Control warm-up currently requires A-head-only model config, got mod_bases={mod_bases}")

    head_defs = dict(getattr(model, "mod_head_defs", {}))
    if head_defs.get("A") != ["canonical_A", "m6A"]:
        raise ValueError(
            "Control warm-up currently requires model.mod_head_defs.A == ['canonical_A', 'm6A'] "
            f"for full-mod vs IVT supervision, got {head_defs.get('A')}"
        )

    if not bool(getattr(model, "standalone_mod_head", False)):
        raise ValueError("Control warm-up requires a frozen-encoder standalone mod-head model.")


class ControlWarmupLoss:
    """
    Explicit promoted loss path for full-mod vs IVT control supervision.

    This stage reuses the model's existing Viterbi-aligned target projection, but
    applies a promoted-specific binary BCE-with-logits objective on the A-head
    for full-mod vs IVT control supervision.
    """

    def __init__(self, model):
        validate_control_warmup_model(model)
        self.model = model
        self.loss_path = CONTROL_WARMUP_LOSS_PATH
        self.promote_stage = PROMOTE_STAGE_CONTROL

    def __call__(self, outputs, targets, target_lengths, mod_targets):
        projection = self.model.align_predictions_to_targets(outputs, targets, target_lengths, mod_targets)
        head_projection = projection["per_head"]["A"]
        flat_logits = head_projection["flat_logits"]
        flat_targets = head_projection["flat_targets"]
        base_scores = outputs["base_scores"]
        base_loss = base_scores.new_zeros((), dtype=base_scores.dtype)

        if flat_targets.numel() == 0:
            mod_loss = base_scores.new_zeros(())
        else:
            if flat_logits.ndim != 2 or flat_logits.shape[-1] != 2:
                raise ValueError(
                    "Control warm-up expects binary A-head logits with shape [N, 2], "
                    f"got {tuple(flat_logits.shape)}"
                )
            positive_logits = flat_logits[:, 1] - flat_logits[:, 0]
            positive_targets = flat_targets.to(dtype=positive_logits.dtype)
            mod_loss = F.binary_cross_entropy_with_logits(positive_logits, positive_targets)

        total_loss = self.model.mod_loss_weight * mod_loss
        return {
            "loss": base_loss,
            "base_loss": base_loss,
            "mod_loss": mod_loss,
            "total_loss": total_loss,
        }


class LLPProportionLoss:
    """
    Minimal LLP objective for known mixture proportions.

    Read probabilities are computed as the mean m6A probability across aligned
    A-head sites for each read. Bags are formed from the unique per-read sample
    key using integer grouping, then optimized against a known bag proportion.
    """

    def __init__(self, model, *, llp_proportion, llp_bag_size=DEFAULT_LLP_BAG_SIZE):
        validate_control_warmup_model(model)
        self.model = model
        self.llp_proportion = None if llp_proportion is None else normalize_llp_proportion(llp_proportion)
        self.llp_bag_size = int(llp_bag_size)
        if self.llp_bag_size < 0:
            raise ValueError(f"LLP bag size must be >= 0, got {self.llp_bag_size}")
        self.loss_path = LLP_LOSS_PATH
        self.promote_stage = PROMOTE_STAGE_LLP

    def _read_probabilities(self, outputs, projection, target_lengths):
        head_projection = projection["per_head"]["A"]
        flat_logits = head_projection["flat_logits"]
        flat_sample_indices = head_projection.get("flat_sample_indices")
        base_scores = outputs["base_scores"]
        num_reads = int(target_lengths.shape[0])

        if flat_logits.numel() == 0:
            empty_probs = base_scores.new_zeros((0,), dtype=base_scores.dtype)
            empty_indices = torch.zeros((0,), device=base_scores.device, dtype=torch.long)
            return empty_probs, empty_indices
        if flat_sample_indices is None:
            raise ValueError("LLP loss requires alignment projection field per_head.A.flat_sample_indices.")
        if flat_logits.ndim != 2 or flat_logits.shape[-1] != 2:
            raise ValueError(f"LLP expects binary A-head logits with shape [N, 2], got {tuple(flat_logits.shape)}")

        positive_logits = flat_logits[:, 1] - flat_logits[:, 0]
        site_probs = torch.sigmoid(positive_logits)
        sample_indices = flat_sample_indices.to(device=site_probs.device, dtype=torch.long)

        read_sums = site_probs.new_zeros((num_reads,))
        read_counts = site_probs.new_zeros((num_reads,))
        read_sums.scatter_add_(0, sample_indices, site_probs)
        read_counts.scatter_add_(0, sample_indices, torch.ones_like(site_probs))
        valid_reads = read_counts > 0
        read_indices = torch.nonzero(valid_reads, as_tuple=False).flatten()
        read_probs = read_sums.index_select(0, read_indices) / read_counts.index_select(0, read_indices)
        return read_probs, read_indices

    def _bag_keys(self, outputs, read_indices):
        if read_indices.numel() == 0:
            return read_indices

        explicit_bag_keys = outputs.get("bag_keys")
        if explicit_bag_keys is not None:
            return explicit_bag_keys.to(device=read_indices.device, dtype=torch.long).index_select(0, read_indices)

        sample_keys = outputs.get("sample_keys")
        if sample_keys is None:
            sample_keys = torch.arange(
                int(outputs["base_scores"].shape[1]),
                device=read_indices.device,
                dtype=torch.long,
            )
        else:
            sample_keys = sample_keys.to(device=read_indices.device, dtype=torch.long)
        read_keys = sample_keys.index_select(0, read_indices)

        if self.llp_bag_size == 0:
            return torch.zeros_like(read_keys)
        return torch.div(read_keys, self.llp_bag_size, rounding_mode="floor")

    def _bag_targets(self, outputs, read_indices, inverse, num_bags):
        read_targets = outputs.get("bag_targets")
        if read_targets is None:
            if self.llp_proportion is None:
                raise ValueError(
                    "LLP loss requires per-read bag_targets from bag_targets.npy "
                    "or --llp-proportion/training.llp_proportion."
                )
            return torch.full((num_bags,), self.llp_proportion, device=read_indices.device, dtype=torch.float32)

        read_targets = read_targets.to(device=read_indices.device, dtype=torch.float32).index_select(0, read_indices)
        bag_target_sums = read_targets.new_zeros((num_bags,))
        bag_target_counts = read_targets.new_zeros((num_bags,))
        bag_target_sums.scatter_add_(0, inverse, read_targets)
        bag_target_counts.scatter_add_(0, inverse, torch.ones_like(read_targets))
        return bag_target_sums / bag_target_counts.clamp_min(1.0)

    def __call__(self, outputs, targets, target_lengths, mod_targets):
        projection = self.model.align_predictions_to_targets(outputs, targets, target_lengths, mod_targets)
        base_scores = outputs["base_scores"]
        base_loss = base_scores.new_zeros((), dtype=base_scores.dtype)

        read_probs, read_indices = self._read_probabilities(outputs, projection, target_lengths)
        if read_probs.numel() == 0:
            prop_loss = base_scores.new_zeros(())
            num_bags = base_scores.new_zeros(())
            num_reads = base_scores.new_zeros(())
        else:
            bag_keys = self._bag_keys(outputs, read_indices)
            _, inverse = torch.unique(bag_keys, sorted=True, return_inverse=True)
            num_bag_items = int(inverse.max().item()) + 1
            bag_sums = read_probs.new_zeros((num_bag_items,))
            bag_counts = read_probs.new_zeros((bag_sums.shape[0],))
            bag_sums.scatter_add_(0, inverse, read_probs)
            bag_counts.scatter_add_(0, inverse, torch.ones_like(read_probs))
            bag_probs = bag_sums / bag_counts.clamp_min(1.0)
            bag_targets = self._bag_targets(outputs, read_indices, inverse, num_bag_items).to(
                device=bag_probs.device,
                dtype=bag_probs.dtype,
            )
            prop_loss = binary_cross_entropy_on_probabilities(bag_probs, bag_targets)
            num_bags = bag_probs.new_tensor(float(bag_probs.numel()))
            num_reads = read_probs.new_tensor(float(read_probs.numel()))

        total_loss = self.model.mod_loss_weight * prop_loss
        return {
            "loss": base_loss,
            "base_loss": base_loss,
            "mod_loss": prop_loss,
            "llp_loss": prop_loss,
            "llp_num_bags": num_bags,
            "llp_num_reads": num_reads,
            "total_loss": total_loss,
        }


class TrainerPromote(TrainerMod):
    """
    Minimal promoted trainer.

    The initial promoted stage intentionally reuses the stable standalone
    mod-head optimization loop while living behind a separate import path and
    CLI entry for direct baseline comparison.
    """

    def __init__(
        self,
        model,
        device,
        train_loader,
        valid_loader,
        *,
        promote_stage=PROMOTE_STAGE_CONTROL,
        criterion=None,
        llp_proportion=None,
        llp_bag_size=DEFAULT_LLP_BAG_SIZE,
        **kwargs,
    ):
        promote_stage = str(promote_stage).strip().lower()
        if promote_stage not in {PROMOTE_STAGE_CONTROL, PROMOTE_STAGE_LLP}:
            raise ValueError(
                "TrainerPromote currently supports promote_stage="
                f"{PROMOTE_STAGE_CONTROL!r} or {PROMOTE_STAGE_LLP!r}, got {promote_stage!r}"
            )

        if criterion is not None:
            promote_loss = criterion
        elif promote_stage == PROMOTE_STAGE_CONTROL:
            promote_loss = ControlWarmupLoss(model)
        else:
            promote_loss = LLPProportionLoss(
                model,
                llp_proportion=llp_proportion,
                llp_bag_size=llp_bag_size,
            )
        self.promote_stage = promote_stage
        self.loss_path = getattr(promote_loss, "loss_path", CONTROL_WARMUP_LOSS_PATH)
        super().__init__(
            model,
            device,
            train_loader,
            valid_loader,
            criterion=promote_loss,
            **kwargs,
        )
