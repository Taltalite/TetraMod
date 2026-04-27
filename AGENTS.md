# AGENTS.md

## Scope

This repository is the only writable target.

If reference directories exist, treat them as read-only:
- ../bonito-mixed
- ../bonitov1.1.0

Do not modify reference repositories unless explicitly requested.

---

## Current goal

The current task is to extend TetraMod with a new promoted training pipeline for weak supervision and improved modification modeling.

The new workflow must be exposed through a **new CLI entry**:

- `train_promote`

The existing training workflow must remain intact:

- `train`

The original `train` path is a protected baseline and must stay runnable for direct comparison.

---

## Hard rules

- Do **not** change the behavior of the existing `train` pipeline unless explicitly requested.
- Do **not** rename or replace the existing `train` entrypoint.
- All new weak-supervision / promoted-training logic must be implemented through `train_promote` and its supporting modules.
- Prefer adding parallel modules, configs, and trainers over invasive edits.
- If shared helpers are extracted from old code, preserve original behavior for `train`.
- Keep changes minimal, reviewable, and easy to compare against the baseline.
- Do not broad-refactor unrelated code.
- Do not mix experimental logic into baseline code paths unless required and proven safe.

---

## Target architecture

The promoted path should follow this design unless evidence suggests a smaller viable variant first:

raw signal
→ frozen Bonito pretrained encoder
→ hidden representation
→ trainable mod trunk
→ mod heads
→ per-read / per-site modification probabilities

Initial priority:
- focus on **A-head only** for m6A-related work
- keep C/G/T heads untouched or optional unless real labels exist

Preferred promoted design:
1. frozen Bonito encoder
2. trainable mod trunk
3. optional SSL projector head for pretraining
4. optional bag/site pooling module for LLP or MIL
5. optional domain-adversarial head only if needed later
6. optional raw-signal side branch only if clearly justified later

Do not start with a large architecture rewrite.

---

## Training philosophy

The promoted path should move from strong control supervision toward weak supervision.

Important principle:
- full-modified vs IVT labels are useful anchors
- they are **not** the final target formulation for wild-type site-level performance

Therefore, `train_promote` should gradually support:

1. control supervised warm-up
2. self-supervised / consistency pretraining for mod trunk
3. LLP-style bag proportion supervision
4. MIL-style site-level supervision
5. optional pseudo-label refinement later

Do not implement every stage at once if the repository is not ready.
Build in stages.

---

## Required implementation strategy

### Protected baseline
Treat the original `train` path as frozen baseline behavior.

### Promoted path
Implement new logic in parallel locations, for example:
- new CLI entry for `train_promote`
- new trainer module(s)
- new loss module(s)
- new dataset / bagging utilities if needed
- new config arguments dedicated to promoted mode

### Reuse policy
Prefer:
- reusing existing baseline model loading
- reusing existing encoder loading
- reusing stable IO / chunking / alignment utilities

Avoid:
- copying entire baseline training code if a thin wrapper is enough
- injecting promoted-only branching into baseline code unless necessary

If code is copied or adapted, clearly state:
- source file
- reason for copying
- whether behavior diverges from baseline intentionally

---

## Preferred development order

Implement in the following order unless the repository state strongly suggests otherwise.

### Stage 0: preserve baseline comparability
- confirm original `train` still runs unchanged
- confirm new work does not break old CLI or imports

### Stage 1: add `train_promote` skeleton
- create CLI entry
- wire config parsing
- route to a separate promoted trainer path
- keep behavior minimal but runnable

### Stage 2: support frozen-encoder promoted training
- load Bonito encoder in frozen mode
- train only mod trunk and required head(s)
- start with A-head priority

### Stage 3: add SSL / consistency pretraining support
Prefer lightweight support first:
- same-chunk augmentation consistency
- hidden-feature masking/reconstruction
- contrastive or cosine-style representation alignment

Do not retrain the full Bonito encoder first.

### Stage 4: add control BCE warm-up
- support full-mod vs IVT anchor supervision
- require balanced sampling where practical
- avoid using this as the only final objective

### Stage 5: add LLP support
- bag reads by site / sample / mixture ratio when available
- aggregate read probabilities into bag prediction
- optimize bag-level proportion loss

### Stage 6: add MIL support
- aggregate read predictions to site-level prediction
- support mean pooling first unless a different pooling is clearly justified
- compare site-level outputs against gold-standard labels

### Stage 7: optional later extensions
Only after earlier stages are stable:
- pseudo-label refinement
- domain adversarial loss
- adapter/LoRA on later encoder layers
- raw-signal auxiliary branch

---

## Validation requirements

For every meaningful promoted change:

1. confirm original `train` still works
2. confirm `train_promote` is callable from CLI
3. report files added or changed
4. report whether baseline behavior changed
5. report what new promoted capability was added
6. add at least one minimal smoke test, import test, or config-path test when practical

When implementing a new promoted training stage, also report:
- what labels are assumed
- whether the output is read-level, bag-level, or site-level
- whether the stage is ready for comparison against baseline

---

## CLI requirements

The CLI must expose a new entry named:

- `train_promote`

This new entry must not replace or shadow:
- `train`

If needed, add new arguments specifically for promoted training rather than overloading baseline arguments excessively.

Prefer clear separation such as:
- baseline config / args remain baseline
- promoted config / args remain promoted

---

## Non-goals

Unless explicitly requested, do not:
- redesign the entire TetraMod package
- rewrite the original training pipeline
- force all heads to participate in m6A training
- fully unfreeze the Bonito encoder at the start
- add RL-based logic as the first solution
- introduce large speculative abstractions before the CLI path is working

---

## Preferred reporting style

At the end of each task, report:

1. what was implemented
2. whether `train` remained unchanged
3. how `train_promote` currently differs from `train`
4. what stage of the promoted roadmap is now complete
5. what the next smallest safe step should be