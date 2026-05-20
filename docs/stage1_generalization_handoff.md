# Stage 1 Generalization Handoff

This document is a compact handoff for continuing the TetraMod Stage 1 m6A
generalization work in a new Codex session.

## Project Position

TetraMod is currently extending a protected Bonito-style basecaller baseline with
a separate promoted modification-training path:

- baseline entrypoint: `tetramod train`
- promoted entrypoint: `tetramod train_promote`

The protected baseline `train` path should remain unchanged.  Current m6A work
should stay under `train_promote` and supporting promoted dataset / validation
utilities.

The current promoted Stage 1 task is:

```text
frozen RNA002 basecaller encoder
-> standalone mod trunk
-> A-head
-> canonical_A vs m6A supervised control warm-up
```

The goal is not only to fit seen IVT controls, but to improve recognition of m6A
in unseen sequence contexts / motifs.  This is the main blocker before any
credible wild-type generalization claim.

## Current Data Tracks

### mAFiA RNA002

Primary clean Stage 1 control source.

- Built by `train_mix_stage1_6motif_dataset.sh`.
- Dataset path on the remote server:
  `/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002/chunks/stage1_train_mafia_6motif_wue_batch2_rlmix1_4`
- Final heldout per-run datasets:
  `/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002/chunks/final_heldout_mix_1_4`
- mAFiA provides matched unmodified and fully modified controls.
- Six mAFiA motifs:
  `AGACT,GAACT,GGACA,GGACC,GGACT,TGACT`

Important heldout interpretation:

- `Mix_1_A_RTA` and `Mix_2_m6A_RTA` are stricter oligo/context heldouts.
- `Mix_3_A_RTA` and `Mix_4_m6A_RTA` are more run-level heldouts with more train
  oligo overlap.

### MoDiDeC RNA002 m6A

Auxiliary synthetic m6A source.

- Built by `train_modidec_m6a_stage1_dataset.sh`.
- Input should be an already-basecalled Dorado BAM with `--emit-moves` plus the
  corresponding POD5 path.
- Positive labels are the explicit Supplementary Table 1 m6A sites.
- MoDiDeC is not a matched 0% vs 100% control dataset in the current downloaded
  form.  Internal negatives must be sampled from non-matched m6A oligo regions.

MoDiDeC positive motifs from #1-#10:

```text
AAACA,AGACA,AGACC,AGACT,GAACC,GAACT,GGACC,TAACG,TAACT
```

Shared with mAFiA:

```text
AGACT,GAACT,GGACC
```

mAFiA-only:

```text
GGACA,GGACT,TGACT
```

MoDiDeC-only:

```text
AAACA,AGACA,AGACC,GAACC,TAACG,TAACT
```

The preferred MoDiDeC preprocessing mode is now:

```bash
NEGATIVE_LABEL_MODE=center
NEGATIVE_MOTIF_MODE=positive-motifs
```

This keeps one canonical A label per negative chunk and restricts MoDiDeC
internal negatives to the selected positive motif set.

## Current Result Summary

The old mixed dataset used MoDiDeC internal negatives in a way that made the
model more conservative.  On mAFiA final heldout:

```text
mAFiA-only:
  recall           0.8174
  specificity      0.9581
  balanced acc     0.8877
  BCE              0.1803

mAFiA + MoDiDeC old logic:
  recall           0.7473
  specificity      0.9659
  balanced acc     0.8566
  BCE              0.2135
```

Interpretation:

- Old-logic MoDiDeC mixing did not improve mAFiA heldout generalization.
- The mixed model improved specificity slightly but lost positive recall.
- This does not prove MoDiDeC is useless.  It shows the previous mixing /
  negative-sampling strategy was not a useful generalization strategy.

The most problematic motif remains `GAACT`.  This is not simply due to lack of
training support; the model has GAACT examples, but it fails on stricter unseen
oligo/context heldout.  The likely issue is context shortcut learning rather
than motif absence.

## Existing Command Flow

`train_modidec_mafia_mix.sh` is a command-flow checklist, not a robust executable
script.  It contains:

- old mixed dataset creation and validation steps
- new MoDiDeC positive-motif-negative dataset creation
- new mAFiA + MoDiDeC merge
- `train_promote`
- internal validation
- mAFiA heldout validation
- MoDiDeC #11 heldout validation
- visualization with the full mixed motif set

Before training any new mixed dataset, always inspect:

```bash
cat "$MIX_DIR/mafia_stage1_merge_summary.json"
```

Then set:

- `--chunks` to `train.num_samples` rounded down to a multiple of batch size
- `--valid-chunks` to `validation.num_samples`

Do not blindly reuse old `--chunks 37184 --valid-chunks 4152`.

## Main Diagnosis

Adding data does not automatically improve unseen motif generalization.  The
current Stage 1 BCE objective can be solved through shortcuts:

- motif context
- source dataset
- run identity
- ligation strategy
- oligo-specific signal patterns
- basecalling / alignment artifacts

Current `train_promote` does not explicitly force the m6A representation to be
motif-invariant.  It only asks the A-head to classify canonical_A vs m6A on the
observed training distribution.

The next phase should explicitly measure and optimize cross-context
generalization.

## Immediate Next Work

### 1. Build Leave-One-Motif-Out Evaluation

Create a systematic LOMO benchmark for mAFiA:

```text
train on 5 motifs
test on the held-out motif
repeat for each of:
AGACT,GAACT,GGACA,GGACC,GGACT,TGACT
```

Record per heldout motif:

- ROC AUC
- PR AUC
- recall at fixed specificity
- specificity at fixed recall
- BCE
- mean positive probability
- mean negative probability

This should become the main Stage 1 generalization benchmark.

### 2. Stop Physically Downsampling mAFiA When Adding MoDiDeC

The current `source-class` merge makes source/class groups equal, but this can
discard too much clean mAFiA signal.

Preferred next strategy:

- keep all mAFiA samples
- include MoDiDeC as auxiliary data
- control MoDiDeC influence by sampler weight or loss weight

Initial weights to test:

```text
mAFiA loss weight:   1.0
MoDiDeC loss weight: 0.25 or 0.5
```

This requires code support for source-aware sample weighting or weighted loss.

### 3. Evaluate New-Logic MoDiDeC Separately

Run the new MoDiDeC preprocessing track:

```bash
NEGATIVE_LABEL_MODE=center
NEGATIVE_MOTIF_MODE=positive-motifs
```

Then evaluate:

- mixed internal validation
- mAFiA final heldout
- MoDiDeC #11 heldout
- full mixed motif visualization:
  `AAACA,AGACA,AGACC,AGACT,GAACC,GAACT,GGACC,GGACA,GGACT,TAACG,TAACT,TGACT`

Only compare it against mAFiA-only after the mAFiA sample retention issue is
handled or explicitly accounted for.

## Model Improvements To Consider

### A. Supervised Contrastive Auxiliary Loss

Add a center-site embedding objective:

- pull m6A embeddings from different motifs closer
- pull canonical_A embeddings from different motifs closer
- separate canonical_A and m6A embeddings

This directly targets motif-invariant representation learning.

### B. Motif Adversarial Loss

Add a small motif classifier on the center-site embedding with gradient reversal.

Goal:

- the mod head can still classify m6A
- but the intermediate embedding should not easily reveal motif identity

Use only for experiments; keep it behind promoted-only config / CLI options.

### C. Prototype / Metric Head

Instead of only a linear classifier:

```text
center embedding
-> distance to canonical_A prototype
-> distance to m6A prototype
-> m6A score
```

This may be better aligned with few-shot / unseen motif behavior than a pure
BCE linear head.

### D. Hierarchical Base-Conditioned Mod Head

The user's proposed idea of first asking "is this base modified?" and then
"which modification type?" is directionally useful, but it should be
base-conditioned rather than one global ATCG modness head.

Preferred long-term shape:

```text
shared mod trunk
-> A_modness: canonical_A vs modified_A
-> A_type:    m6A / other_A
-> C_modness: canonical_C vs modified_C
-> C_type:    m5C / hm5C / other_C
...
```

For current Stage 1 m6A:

```text
A embedding
-> A_modness: canonical_A vs modified_A
-> A_type: m6A only when positive labels exist
```

This prepares for multi-modification training without mixing base identity
shortcuts into one global "modified ATCG" classifier.

### E. Careful Partial Unfreeze

Only after a stable frozen-encoder Stage 1:

- unfreeze the last encoder block or last small encoder section
- use very small LR for encoder: `1e-6` to `3e-6`
- keep trunk/head LR around `1e-4`
- train only a few epochs
- guard both mod heldout metrics and basecalling accuracy

Do not fully unfreeze the encoder at the start.

## Recommended Experiment Matrix

Run in this order:

1. mAFiA-only baseline with current best settings.
2. mAFiA-only LOMO benchmark.
3. mAFiA + new-logic MoDiDeC with current merge, for comparison only.
4. mAFiA full-retention + MoDiDeC weighted auxiliary.
5. Add supervised contrastive loss and repeat LOMO.
6. Add prototype head and repeat LOMO.
7. Only then consider partial unfreeze.

Decision rules:

- If MoDiDeC improves only internal validation but not mAFiA heldout, it is not
  improving cross-context generalization.
- If MoDiDeC improves shared motifs but hurts mAFiA-only motifs, the model is
  learning source/motif-specific shortcuts.
- If LOMO improves, especially for `GAACT`, the change is likely meaningful.
- Wild-type evaluation should not be used as the first proof of model quality;
  it should come after controlled LOMO improvement.

## Files To Inspect First In A New Session

```text
AGENTS.md
README.md
docs/stage1_dataset_unification.md
docs/stage1_generalization_handoff.md
train_mix_stage1_6motif_dataset.sh
train_modidec_m6a_stage1_dataset.sh
train_modidec_mafia_mix.sh
gen_data/create_mafia_synthetic_stage1_dataset.py
gen_data/create_modidec_m6a_stage1_dataset.py
gen_data/merge_mafia_stage1_datasets.py
src/tetramod/training_promote.py
src/tetramod/transformer/multihead_model.py
src/tetramod/models/configs/multihead_transformer_promote_stage1_adamw.toml
validate/evaluate_mafia_stage1.py
dataset_check/plot_mafia_stage1_visuals.py
```

## Constraints To Preserve

- Do not change baseline `train` behavior.
- Keep promoted changes behind `train_promote`.
- Keep Stage 1 focused on A/m6A first.
- Do not claim wild-type generalization from IVT controls alone.
- Avoid broad refactors; add promoted-only options and small utilities.
- Any new generalization trick should have a corresponding heldout/LOMO
  validation report.
