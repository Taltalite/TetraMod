# mAFiA Synthetic Stage 1 Dataset Workflow

This workflow builds a `train_promote --promote-stage control` dataset from the
mAFiA synthetic RNA oligo runs.  It is intended for the remote GPU/server
environment where the raw FAST5 files are stored.

## 1. Prepare Manifests

Create two TSV files from the paper supplement:

```bash
python gen_data/create_mafia_synthetic_stage1_dataset.py \
  --write-template-manifest manifests/mafia
```

Fill `mafia_oligos.tsv` from Supplementary Table 1:

- `oligo_id`: e.g. `RL_M0_S0`, `SL_M0_S0`, `SL_AB`
- `sequence`: the oligo sequence, using `/m6A` at the controlled center if present
- `motif`: center 5-mer, with U normalized to T for TetraMod
- `ligation_strategy`: `random_ligation` or `splint_ligation`
- `role`: `train` or `test`

Fill `mafia_runs.tsv` from Supplementary Table 3 and local directory names:

- `modification_status`: `unmodified`, `modified`, or `mixed`
- `oligo_ids`: comma-separated oligos present in the run
- `modified_oligo_ids`: required for `mixed` runs such as TEST1

Stage 1 should use TRAIN synthetic runs only.  TEST1/TEST2 should remain
held-out benchmarks.

## 2. Convert FAST5 To POD5

For each downloaded run directory:

```bash
python gen_data/convert_fast5_tar_to_pod5.py \
  /data/biolab-backup-hdd2/public_data/mAFia_RNA002_PRJEB74106/HEK293/<RUN_DIR> \
  --output-dir /data/.../mafia_pod5/<RUN_ID> \
  --recursive \
  --jobs 1
```

If FAST5 members inside archives do not have `.fast5` suffixes, add:

```bash
--fast5-member-pattern '*'
```

## 3. Dorado RNA002 Basecalling

Basecall each POD5 directory with move tables:

```bash
dorado basecaller <RNA002_MODEL> /data/.../mafia_pod5/<RUN_ID> \
  --emit-moves \
  --device cuda:0 \
  > /data/.../mafia_bam/<RUN_ID>.bam
```

The resulting BAM must contain `mv` and `ns` tags.

## 4. Build Per-Run TetraMod Datasets

```bash
python gen_data/create_mafia_synthetic_stage1_dataset.py \
  --bam-file /data/.../mafia_bam/<RUN_ID>.bam \
  --pod5-dir /data/.../mafia_pod5/<RUN_ID> \
  --output-dir /data/.../mafia_chunks/per_run/<RUN_ID> \
  --oligo-manifest manifests/mafia/mafia_oligos.tsv \
  --run-manifest manifests/mafia/mafia_runs.tsv \
  --run-id <RUN_ID> \
  --sample-type rna \
  --rna002 \
  --chunk-len 10000 \
  --overlap 500 \
  --workers 8
```

The script writes `mod_targets.npy` directly.  Only controlled center DRACH A
positions are labeled:

- `0`: canonical_A in unmodified oligo units
- `4`: m6A in modified oligo units
- `-100`: all other positions

## 5. Merge TRAIN Runs

```bash
python gen_data/merge_mafia_stage1_datasets.py \
  --dataset run1:/data/.../mafia_chunks/per_run/run1 \
  --dataset run2:/data/.../mafia_chunks/per_run/run2 \
  --output-dir /data/.../mafia_chunks/stage1_train \
  --valid-fraction 0.25 \
  --seed 114514
```

The merger creates:

- `/data/.../mafia_chunks/stage1_train`
- `/data/.../mafia_chunks/stage1_train/validation`

Training samples are balanced per motif between positive and negative labels.

## 6. Train Stage 1

```bash
tetramod train_promote -f /data/.../tetramod_model/stage1_mafia_run1 \
  --directory /data/.../mafia_chunks/stage1_train \
  --config src/tetramod/models/configs/multihead_transformer.toml \
  --pretrained <RNA002_TETRAMOD_PRETRAINED_DIR> \
  --promote-stage control \
  --promote-base A \
  --epochs 20 \
  --batch 64 \
  --lr 1e-4 \
  --device cuda:0
```

Do not use HEK293 WT/IVT ratio data in this Stage 1 supervised run.
