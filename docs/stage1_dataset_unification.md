# Stage 1 Dataset Unification Notes

This note records the current policy for building a mixed mAFiA + MoDiDeC
control dataset for `train_promote --promote-stage control`.

## Root Cause

mAFiA and MoDiDeC provide different supervision.

- mAFiA is a matched control design: each DRACH oligo is available as an
  unmodified or m6A-modified molecule, and the labeled target is the controlled
  center A/m6A site.
- MoDiDeC m6A data are an internal-label design in the downloaded RNA002 run:
  positive labels come from the explicit m6A positions in Supplementary Table 1,
  while negative labels must be sampled from A-containing windows outside the
  matched m6A oligo intervals.

Because of that difference, simply concatenating both datasets creates two
confounders:

- label-density bias: MoDiDeC internal negatives can label many A bases in one
  chunk, while mAFiA supervision is center-site oriented;
- source/class bias: mAFiA contributes many matched positive/negative controls,
  while MoDiDeC contributes many internal negatives and fewer positives.

## Current Policy

1. MoDiDeC positive chunks label only the explicit Supplementary Table 1 m6A
   site.
2. MoDiDeC internal negative chunks label one representative canonical A by
   default. This keeps the label density closer to mAFiA center-site supervision.
3. Mixed mAFiA + MoDiDeC datasets should be merged with `source-class`
   balancing, so the training split keeps equal counts for:
   `mafia:positive`, `mafia:negative`, `modidec:positive`, and
   `modidec:negative`.
4. The validation split should also be source/class balanced for control-stage
   model selection. Natural-distribution evaluation can be added later as a
   separate report, not as the primary early stopping validation set.

## Commands

Rebuild MoDiDeC if the per-run dataset was made before `NEGATIVE_LABEL_MODE` was
introduced:

```bash
SKIP_EXISTING=0 \
NEGATIVE_LABEL_MODE=center \
MERGE_BALANCE_MODE=source-class \
MERGE_BALANCE_VALIDATION=1 \
MODIDEC_BAM_SPECS="modidec_train:/path/to/modidec.bam:/path/to/pod5" \
bash train_modidec_m6a_stage1_dataset.sh
```

Then merge the already-built mAFiA and MoDiDeC datasets:

```bash
WORK_ROOT=/path/to/mixed_work_root \
MAFIA_DATASET_DIR=/path/to/stage1_train_mafia_6motif_wue_batch2_rlmix1_4 \
MODIDEC_DATASET_DIR=/path/to/stage1_train_modidec_m6a_rna002 \
bash train_mix_stage1_mafia_modidec_dataset.sh
```

The mixed script only re-merges existing numpy datasets. It does not basecall,
convert POD5, or re-parse BAM records.
