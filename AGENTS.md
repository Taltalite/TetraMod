
# AGENTS.md

## Workflow Context

This repository is developed locally inside Windows WSL2 with Codex. The remote Linux server is mainly used for GPU-heavy training, inference, and full-scale execution.

Unless explicitly instructed otherwise, Codex should edit, inspect, and run only lightweight checks in the local WSL2 workspace. Code will be synchronized to the remote server through Git/GitHub.

Do not hard-code machine-specific paths, usernames, CUDA devices, proxy settings, or dataset locations. Prefer CLI arguments, config files, environment variables, and documented placeholders.

---

## Scope

This repository is the only writable target.

Reference repositories, if present, are read-only:

- `../bonito-mixed`
- `../bonitov1.1.0`

Do not modify reference repositories unless explicitly requested.

---

## Current Goal

Extend TetraMod with a promoted training pipeline for modification modeling under limited supervision.

Add a new CLI entry:

- `train_promote`

Keep the existing baseline CLI entry unchanged:

- `train`

The original `train` path is a protected baseline and must remain runnable for direct comparison.

---

## Hard Rules

- Do **not** change the behavior of the existing `train` pipeline unless explicitly requested.
- Do **not** rename, replace, or shadow the existing `train` entrypoint.
- Implement promoted-training logic through `train_promote` and its supporting modules.
- Prefer parallel modules, trainers, losses, datasets, and configs over invasive edits.
- Keep changes minimal, reviewable, and easy to compare against baseline.
- Do not broad-refactor unrelated code.
- Do not mix experimental logic into baseline code paths.
- Do not fully unfreeze the official basecaller encoder at the beginning.
- Do not use LLP as the first or only training signal.
- Do not claim wild-type generalization from IVT controls or ratio-IVT data alone.

---

## Target Architecture

The promoted path should follow this structure:

```text
raw signal
→ frozen official Bonito / ONT basecaller encoder
→ hidden representation
→ trainable shared mod trunk
→ per-base / per-modification heads
→ read-level modification probabilities
→ optional bag-level or site-level aggregation
````

Initial priority:

* Focus on **A-head only** for m6A-related work.
* Keep C/G/T or C/G/U heads untouched unless real labels exist.
* Use shared trunk + separated heads for future multi-modification expansion.

Preferred long-term head design:

```text
A-head: canonical_A / m6A / other A modifications
C-head: canonical_C / m5C / hm5C / other C modifications
G-head: canonical_G / G modifications
T/U-head: canonical_T/U / relevant modifications
```

---

## Training Strategy

The promoted training path should follow this order:

```text
Stage 1: 0% vs 100% control supervised warm-up
→ Stage 2: ratio-IVT LLP fine-tuning / calibration
→ Stage 3: wild-type / MIL adaptation
```

### Stage 1: Control Supervised Warm-up

Use matched unmodified and fully modified control data.

Purpose:

* learn modification-sensitive signals from frozen basecaller hidden features
* train mod trunk + relevant mod head
* establish a strong supervised anchor before weak supervision

Default behavior:

* freeze official basecaller encoder
* train only mod trunk and A/m6A head first
* use supervised BCE / CE loss
* use balanced sampling where practical

Validation should check:

* heldout run performance
* heldout site or k-mer context performance when available
* whether 0% and 100% controls are clearly separable

This stage should produce a checkpoint for Stage 2.

---

### Stage 2: Ratio-IVT LLP Fine-tuning

Use ratio-IVT data such as:

* 12.5%
* 25%
* 50%
* 75%

This stage should start from a Stage 1 checkpoint.

Purpose:

* calibrate Stage 1 outputs under partial modification
* improve bag-level proportion consistency
* reduce mismatch between full-control training and mixed-ratio data

Loss:

* relaxed MSE, Huber, or similar bag-level proportion loss
* aggregate read-level probabilities into bag-level predictions

Important cautions:

* ratio and run may be confounded
* site/k-mer distributions may differ across ratio groups
* good bag-level LLP loss does not prove read-level correctness
* ratio-IVT performance does not prove wild-type generalization

---

### Stage 3: Wild-type / MIL Adaptation

Use this stage only when site-level weak labels or biological evidence are available.

Possible supervision:

* gold-standard wild-type sites
* miCLIP / m6A-CLIP labels
* knockout or writer perturbation data
* high-confidence positive / negative site sets

Purpose:

* adapt read-level predictions to biological site-level modification calling
* support claims about wild-type generalization

Default pooling:

* mean pooling first
* attention pooling only after a simple baseline works

Only after this stage should the model be discussed as a candidate wild-type modification caller.

---

## Implementation Strategy

### Protected Baseline

Before and after meaningful promoted changes, confirm:

* `train` is still registered
* baseline imports still work
* baseline trainer behavior is unchanged

### Promoted Path

Add new logic in parallel locations, for example:

* new CLI entry for `train_promote`
* new trainer such as `TrainerPromote`
* new promoted loss module
* new promoted dataset / bagging utilities
* new promoted config arguments

Reuse existing stable code where safe:

* model loading
* encoder loading
* signal / chunk / alignment utilities
* checkpoint loading

Avoid:

* copying the entire baseline trainer if a wrapper is enough
* adding promoted-only branches into baseline code
* speculative abstractions before the minimal path works

---

## CLI Requirements

The promoted CLI should expose explicit stages, for example:

```bash
--promote-stage control
--promote-stage llp
--promote-stage mil
```

Suggested stage meanings:

```text
control = 0% vs 100% supervised warm-up
llp     = ratio-IVT bag-level proportion fine-tuning
mil     = wild-type / site-level weak supervision
```

Suggested promoted-only arguments:

```bash
--freeze-encoder
--mod-base A
--mod-type m6A
--control-unmod-data
--control-fullmod-data
--init-promote-checkpoint
--llp-bag-key
--llp-ratio-label
--llp-loss huber
--site-label-file
--pooling mean
```

Prefer promoted-specific arguments over overloading baseline training arguments.

---

## Validation Requirements

For every meaningful promoted change, report:

1. what was implemented
2. whether `train` remained unchanged
3. how `train_promote` differs from `train`
4. which promoted stage is now supported
5. what labels are assumed
6. whether outputs are read-level, bag-level, or site-level
7. what smoke test or import test was run
8. the next smallest safe step

For Stage 1, additionally report:

* whether encoder is frozen
* which head is trained
* whether controls are balanced
* heldout run/site performance if available

For Stage 2, additionally report:

* how bags are constructed
* what bag labels mean
* how read probabilities are aggregated
* predicted vs expected bag-level ratios

For Stage 3, additionally report:

* what site-level labels are used
* how reads are pooled to site level
* whether validation supports wild-type generalization

---

## Non-goals

Unless explicitly requested, do not:

* redesign the whole TetraMod package
* rewrite the original training pipeline
* force all heads to participate in m6A training
* fully unfreeze the basecaller encoder at the start
* add RL-based logic
* make LLP the first training objective
* claim wild-type generalization from IVT data alone
* introduce large abstractions before `train_promote` is runnable

---

## Preferred Reporting Style

At the end of each task, report:

1. implemented changes
2. baseline `train` status
3. current `train_promote` behavior
4. completed promoted stage
5. validation or smoke-test results
6. next smallest safe step
