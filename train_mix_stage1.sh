# data source:

# /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks

MAFIA_ROOT=/data/biolab-backup-hdd2/public_data/mAFia_RNA002_PRJEB74106/HEK293
WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
REPO=/home/lijy/workspace/TetraMod
DORADO_MODEL=/path/to/dorado/rna002_model

mkdir -p \
    "$WORK_ROOT"/manifests \
    "$WORK_ROOT"/pod5 \
    "$WORK_ROOT"/bam \
    "$WORK_ROOT"/chunks/per_run \
    "$WORK_ROOT"/chunks/stage1_train \
    "$WORK_ROOT"/tmp_extract \
    "$WORK_ROOT"/models

cd "$REPO"

python gen_data/create_mafia_synthetic_stage1_dataset.py \
  --write-template-manifest "$WORK_ROOT/manifests"


# 2. FAST5 转 POD5
MAFIA_ROOT=/data/biolab-backup-hdd2/public_data/mAFia_RNA002_PRJEB74106/HEK293
WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
RUN_ID=WUE_splint_lig_A_RTA
RUN_DIR=$(awk -F'\t' -v id="$RUN_ID" 'NR > 1 && $1 == id {print $3}' gen_data/mafia_runs.tsv)
python gen_data/convert_fast5_tar_to_pod5.py \
    "$MAFIA_ROOT/$RUN_DIR" \
    --output-dir "$WORK_ROOT/pod5/$RUN_ID" \
    --recursive \
    --jobs 4


RUN_ID=WUE_splint_lig_m6A_RTA
RUN_DIR=$(awk -F'\t' -v id="$RUN_ID" 'NR > 1 && $1 == id {print $3}' gen_data/mafia_runs.tsv)
python gen_data/convert_fast5_tar_to_pod5.py \
    "$MAFIA_ROOT/$RUN_DIR" \
    --output-dir "$WORK_ROOT/pod5/$RUN_ID" \
    --recursive \
    --jobs 4


RUN_ID=RL_Mix1_A_RTA
RUN_DIR=$(awk -F'\t' -v id="$RUN_ID" 'NR > 1 && $1 == id {print $3}' gen_data/mafia_runs.tsv)
python gen_data/convert_fast5_tar_to_pod5.py \
    "$MAFIA_ROOT/$RUN_DIR" \
    --output-dir "$WORK_ROOT/pod5/$RUN_ID" \
    --recursive \
    --jobs 4


RUN_ID=RL_Mix3_m6A_RTA
RUN_DIR=$(awk -F'\t' -v id="$RUN_ID" 'NR > 1 && $1 == id {print $3}' gen_data/mafia_runs.tsv)
python gen_data/convert_fast5_tar_to_pod5.py \
    "$MAFIA_ROOT/$RUN_DIR" \
    --output-dir "$WORK_ROOT/pod5/$RUN_ID" \
    --recursive \
    --jobs 4


# 3. Dorado RNA002 basecalling

DORADO_MODEL=/home/lijy/workspace/TetraMod/src/tetramod/models/rna002_70bps_sup@v3/rna002_70bps_sup@v3
WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
RUN_ID=WUE_splint_lig_A_RTA
/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller "$DORADO_MODEL" "$WORK_ROOT/pod5/$RUN_ID" \
  --emit-moves \
  --device cuda:0 \
  > "$WORK_ROOT/bam/$RUN_ID.bam"

RUN_ID=WUE_splint_lig_m6A_RTA
/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller "$DORADO_MODEL" "$WORK_ROOT/pod5/$RUN_ID" \
  --emit-moves \
  --device cuda:0 \
  > "$WORK_ROOT/bam/$RUN_ID.bam"

RUN_ID=RL_Mix1_A_RTA
/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller "$DORADO_MODEL" "$WORK_ROOT/pod5/$RUN_ID" \
  --emit-moves \
  --device cuda:0 \
  > "$WORK_ROOT/bam/$RUN_ID.bam"

RUN_ID=RL_Mix3_m6A_RTA
/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller "$DORADO_MODEL" "$WORK_ROOT/pod5/$RUN_ID" \
  --emit-moves \
  --device cuda:0 \
  > "$WORK_ROOT/bam/$RUN_ID.bam"


# 4. 生成 per-run TetraMod 数据集

WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
RUN_ID=WUE_splint_lig_A_RTA
python gen_data/create_mafia_synthetic_stage1_dataset.py \
  --bam-file "$WORK_ROOT/bam/$RUN_ID.sorted.bam" \
  --pod5-dir "$WORK_ROOT/pod5/$RUN_ID" \
  --output-dir "$WORK_ROOT/chunks/per_run/$RUN_ID" \
  --oligo-manifest /home/lijy/workspace/TetraMod/gen_data/mafia_oligos.tsv \
  --run-manifest /home/lijy/workspace/TetraMod/gen_data/mafia_runs.tsv \
  --run-id "$RUN_ID" \
  --sample-type rna \
  --rna002 \
  --chunk-len 5000 \
  --overlap 500 \
  --workers 8

RUN_ID=WUE_splint_lig_m6A_RTA
python gen_data/create_mafia_synthetic_stage1_dataset.py \
  --bam-file "$WORK_ROOT/bam/$RUN_ID.sorted.bam" \
  --pod5-dir "$WORK_ROOT/pod5/$RUN_ID" \
  --output-dir "$WORK_ROOT/chunks/per_run/$RUN_ID" \
  --oligo-manifest /home/lijy/workspace/TetraMod/gen_data/mafia_oligos.tsv \
  --run-manifest /home/lijy/workspace/TetraMod/gen_data/mafia_runs.tsv \
  --run-id "$RUN_ID" \
  --sample-type rna \
  --rna002 \
  --chunk-len 5000 \
  --overlap 500 \
  --workers 8

RUN_ID=RL_Mix1_A_RTA
python gen_data/create_mafia_synthetic_stage1_dataset.py \
  --bam-file "$WORK_ROOT/bam/$RUN_ID.sorted.bam" \
  --pod5-dir "$WORK_ROOT/pod5/$RUN_ID" \
  --output-dir "$WORK_ROOT/chunks/per_run/$RUN_ID" \
  --oligo-manifest /home/lijy/workspace/TetraMod/gen_data/mafia_oligos.tsv \
  --run-manifest /home/lijy/workspace/TetraMod/gen_data/mafia_runs.tsv \
  --run-id "$RUN_ID" \
  --sample-type rna \
  --rna002 \
  --chunk-len 5000 \
  --overlap 500 \
  --workers 8

RUN_ID=RL_Mix3_m6A_RTA
python gen_data/create_mafia_synthetic_stage1_dataset.py \
  --bam-file "$WORK_ROOT/bam/$RUN_ID.sorted.bam" \
  --pod5-dir "$WORK_ROOT/pod5/$RUN_ID" \
  --output-dir "$WORK_ROOT/chunks/per_run/$RUN_ID" \
  --oligo-manifest /home/lijy/workspace/TetraMod/gen_data/mafia_oligos.tsv \
  --run-manifest /home/lijy/workspace/TetraMod/gen_data/mafia_runs.tsv \
  --run-id "$RUN_ID" \
  --sample-type rna \
  --rna002 \
  --chunk-len 5000 \
  --overlap 500 \
  --workers 8

# 5. 合并 Stage1 train dataset

WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002

python gen_data/merge_mafia_stage1_datasets.py \
    --dataset WUE_splint_lig_A_RTA:"$WORK_ROOT/chunks/per_run/WUE_splint_lig_A_RTA" \
    --dataset WUE_splint_lig_m6A_RTA:"$WORK_ROOT/chunks/per_run/WUE_splint_lig_m6A_RTA" \
    --dataset RL_Mix1_A_RTA:"$WORK_ROOT/chunks/per_run/RL_Mix1_A_RTA" \
    --dataset RL_Mix3_m6A_RTA:"$WORK_ROOT/chunks/per_run/RL_Mix3_m6A_RTA" \
    --output-dir "$WORK_ROOT/chunks/stage1_train_mafia_wue_rl" \
    --valid-fraction 0.1 \
    --seed 114514


# 6. stage 1 training
WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
BONITO_MODEL_DIR=/data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3
tetramod train_promote "$WORK_ROOT/models/stage1_mafia_wue_rl" \
    --config src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained "$BONITO_MODEL_DIR" \
    --directory "$WORK_ROOT/chunks/stage1_train_mafia_wue_rl" \
    --promote-stage control \
    --chunks 33280 \
    --valid-chunks 11913 \
    --batch 64 \
    --epochs 20 \
    --device cuda:0
