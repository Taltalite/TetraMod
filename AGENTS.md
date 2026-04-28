# AGENTS.md

# Workflow Context

This repository is developed locally inside Windows WSL2 with Codex. The remote Linux server is used mainly for GPU-heavy training, inference, and full-scale execution. Unless explicitly instructed otherwise, Codex should edit, inspect, and run lightweight checks only in the local WSL2 workspace, and assume that code will be synchronized to the remote server through Git/GitHub.

Do not hard-code local or remote machine-specific paths, usernames, CUDA devices, proxy settings, or dataset locations. Prefer portable configuration via CLI arguments, config files, environment variables, or documented placeholders.

## Scope

This repository is the only writable target.

If reference directories exist, treat them as read-only:

- `../bonito-mixed`
- `../bonitov1.1.0`

Do not modify reference repositories unless explicitly requested.

---

# Current Goal

The current task is to extend TetraMod with a promoted training pipeline for modification modeling under limited supervision.

The promoted workflow must be exposed through a new CLI entry:

- `train_promote`

The existing baseline training workflow must remain intact:

- `train`

The original `train` path is a protected baseline and must stay runnable for direct comparison.

---

# Core Scientific Direction

The promoted training path should follow a staged strategy:

1. **Stage 1: control supervised warm-up**
   - Use matched unmodified and fully modified control data.
   - Train the mod trunk and the relevant mod head under strong 0% vs 100% supervision.
   - Keep the official ONT / Bonito basecaller encoder frozen at the beginning.
   - The purpose is to learn modification-sensitive signals from basecaller hidden features.

2. **Stage 2: ratio-IVT LLP fine-tuning**
   - Use ratio-IVT data such as 12.5%, 25%, 50%, and 75%.
   - Start from the Stage 1 checkpoint.
   - Use LLP-style bag-level proportion supervision as calibration, not as the main training anchor.
   - The purpose is to calibrate model outputs under partial modification, mixed proportions, and batch/run variation.

3. **Stage 3: wild-type / MIL adaptation**
   - Use site-level weak labels, high-confidence gold-standard sites, perturbation data, or external biological evidence when available.
   - Aggregate read-level predictions into site-level probabilities using mean pooling first, and attention pooling only after a simple baseline is working.
   - The purpose is to support final biological sample generalization.

Important principle:

- Stage 1 teaches the model what modification-like signal looks like.
- Stage 2 calibrates the output scale under controlled mixed-ratio IVT data.
- Stage 3 is required before claiming reliable wild-type generalization.

Do not claim that Stage 1 + Stage 2 alone proves wild-type generalization.

---

# Hard Rules

- Do **not** change the behavior of the existing `train` pipeline unless explicitly requested.
- Do **not** rename, replace, or shadow the existing `train` entrypoint.
- All new promoted-training logic must be implemented through `train_promote` and its supporting modules.
- Prefer adding parallel modules, configs, and trainers over invasive edits.
- If shared helpers are extracted from old code, preserve original behavior for `train`.
- Keep changes minimal, reviewable, and easy to compare against the baseline.
- Do not broad-refactor unrelated code.
- Do not mix experimental logic into baseline code paths unless required and proven safe.
- Do not fully unfreeze the official basecaller encoder at the start.
- Do not make LLP the first or only training signal.
- Do not use ratio-IVT LLP data as a substitute for wild-type validation.

---

# Target Architecture

The promoted path should follow this design unless repository constraints require a smaller viable version first:

```text
raw signal
→ frozen official Bonito / ONT basecaller encoder
→ hidden representation
→ trainable shared mod trunk
→ per-base / per-modification heads
→ read-level modification probabilities
→ optional bag-level or site-level aggregation