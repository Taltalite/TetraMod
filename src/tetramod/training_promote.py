"""
Promoted trainer path kept separate from the protected baseline trainer.
"""

from __future__ import annotations

import torch.nn.functional as F

from tetramod.training_mod import TrainerMod


PROMOTE_STAGE_CONTROL = "control"
CONTROL_WARMUP_LOSS_PATH = "a_head_control_warmup_viterbi_bce"


def resolve_promote_stage(config, cli_stage: str | None = None) -> str:
    stage = cli_stage or config.get("training", {}).get("promote_stage") or PROMOTE_STAGE_CONTROL
    stage = str(stage).strip().lower()
    if stage != PROMOTE_STAGE_CONTROL:
        raise ValueError(
            f"train_promote currently supports only promote_stage={PROMOTE_STAGE_CONTROL!r}, got {stage!r}"
        )
    return stage


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


class TrainerPromote(TrainerMod):
    """
    Minimal promoted trainer.

    The initial promoted stage intentionally reuses the stable standalone
    mod-head optimization loop while living behind a separate import path and
    CLI entry for direct baseline comparison.
    """

    def __init__(self, model, device, train_loader, valid_loader, *, promote_stage=PROMOTE_STAGE_CONTROL, criterion=None, **kwargs):
        promote_stage = str(promote_stage).strip().lower()
        if promote_stage != PROMOTE_STAGE_CONTROL:
            raise ValueError(
                f"TrainerPromote currently supports only promote_stage={PROMOTE_STAGE_CONTROL!r}, got {promote_stage!r}"
            )

        control_loss = criterion or ControlWarmupLoss(model)
        self.promote_stage = promote_stage
        self.loss_path = getattr(control_loss, "loss_path", CONTROL_WARMUP_LOSS_PATH)
        super().__init__(
            model,
            device,
            train_loader,
            valid_loader,
            criterion=control_loss,
            **kwargs,
        )
