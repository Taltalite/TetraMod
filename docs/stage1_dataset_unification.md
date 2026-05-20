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
   The shell pipeline also restricts those negative A sites to the selected
   MoDiDeC positive motif set by default, reducing motif-only shortcuts.
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
NEGATIVE_MOTIF_MODE=positive-motifs \
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

## Parallel Dataset Tracks

Two tracks can be kept side by side.

Track A keeps the earlier MoDiDeC internal-negative policy. This is useful for a
direct comparison with already-started experiments:

```bash
WORK_ROOT=/path/to/modidec_old_logic_work_root \
TRAIN_DATASET_NAME=stage1_train_modidec_m6a_rna002_any_a_neg \
NEGATIVE_LABEL_MODE=center \
NEGATIVE_MOTIF_MODE=any-a \
MODIDEC_BAM_SPECS="modidec_train:/path/to/modidec.bam:/path/to/pod5" \
bash train_modidec_m6a_stage1_dataset.sh
```

Track B uses motif-restricted internal negatives:

```bash
WORK_ROOT=/path/to/modidec_motif_neg_work_root \
TRAIN_DATASET_NAME=stage1_train_modidec_m6a_rna002_positive_motif_neg \
NEGATIVE_LABEL_MODE=center \
NEGATIVE_MOTIF_MODE=positive-motifs \
MODIDEC_BAM_SPECS="modidec_train:/path/to/modidec.bam:/path/to/pod5" \
bash train_modidec_m6a_stage1_dataset.sh
```

Each track can then be merged with the same mAFiA dataset by changing
`MIX_DATASET_NAME` and `MODIDEC_DATASET_DIR`:

```bash
WORK_ROOT=/path/to/mixed_work_root \
MIX_DATASET_NAME=stage1_train_mafia_modidec_m6a_any_a_neg \
MAFIA_DATASET_DIR=/path/to/stage1_train_mafia_6motif_wue_batch2_rlmix1_4 \
MODIDEC_DATASET_DIR=/path/to/stage1_train_modidec_m6a_rna002_any_a_neg \
bash train_mix_stage1_mafia_modidec_dataset.sh

WORK_ROOT=/path/to/mixed_work_root \
MIX_DATASET_NAME=stage1_train_mafia_modidec_m6a_positive_motif_neg \
MAFIA_DATASET_DIR=/path/to/stage1_train_mafia_6motif_wue_batch2_rlmix1_4 \
MODIDEC_DATASET_DIR=/path/to/stage1_train_modidec_m6a_rna002_positive_motif_neg \
bash train_mix_stage1_mafia_modidec_dataset.sh
```
