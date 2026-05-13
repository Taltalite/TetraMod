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


# 运行 validation 评估

WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
MODEL_DIR="$WORK_ROOT/models/stage1_mafia_wue_rl"
DATA_DIR="$WORK_ROOT/chunks/stage1_train_mafia_wue_rl"

python validate/evaluate_mafia_stage1.py "$MODEL_DIR" \
  --dataset-dir "$DATA_DIR" \
  --split validation \
  --weights 5 \
  --device cuda:0 \
  --batchsize 64 \
  --num-workers 4 \
  --output-dir "val_res/mafia_stage1_epoch5" \
  --write-sites

# 制作用于 validat 的数据集

WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
REPO=/home/lijy/workspace/TetraMod
cd "$REPO"

MAFIA_ROOT=/data/biolab-backup-hdd2/public_data/mAFia_RNA002_PRJEB74106/HEK293

for RUN_ID in \
  WUE_splint_batch2_A_RTA \
  WUE_splint_batch2_m6A_RTA \
  WUE_splint_batch2_m6A_RTA_1 \
  WUE_splint_batch2_m6A_RTA_2
do
  RUN_DIR=$(awk -F'\t' -v id="$RUN_ID" 'NR > 1 && $1 == id {print $3}' gen_data/mafia_runs.tsv)
  python gen_data/convert_fast5_tar_to_pod5.py \
      "$MAFIA_ROOT/$RUN_DIR" \
      --output-dir "$WORK_ROOT/pod5/$RUN_ID" \
      --recursive \
      --jobs 4
done

DORADO_MODEL=/home/lijy/workspace/TetraMod/src/tetramod/models/rna002_70bps_sup@v3/rna002_70bps_sup@v3
for RUN_ID in \
  WUE_splint_batch2_A_RTA \
  WUE_splint_batch2_m6A_RTA \
  WUE_splint_batch2_m6A_RTA_1 \
  WUE_splint_batch2_m6A_RTA_2
do
  /home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller "$DORADO_MODEL" "$WORK_ROOT/pod5/$RUN_ID" \
    --emit-moves \
    --device cuda:0 \
    > "$WORK_ROOT/bam/$RUN_ID.bam"
done

WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
for RUN_ID in \
  WUE_splint_batch2_A_RTA \
  WUE_splint_batch2_m6A_RTA \
  WUE_splint_batch2_m6A_RTA_1 \
  WUE_splint_batch2_m6A_RTA_2
do
  samtools sort -o "$WORK_ROOT/bam/$RUN_ID.sorted.bam" "$WORK_ROOT/bam/$RUN_ID.bam"
  samtools index "$WORK_ROOT/bam/$RUN_ID.sorted.bam"
done


for RUN_ID in \
  WUE_splint_batch2_A_RTA \
  WUE_splint_batch2_m6A_RTA \
  WUE_splint_batch2_m6A_RTA_1 \
  WUE_splint_batch2_m6A_RTA_2
do
  python gen_data/create_mafia_synthetic_stage1_dataset.py \
    --bam-file "$WORK_ROOT/bam/$RUN_ID.sorted.bam" \
    --pod5-dir "$WORK_ROOT/pod5/$RUN_ID" \
    --output-dir "$WORK_ROOT/chunks/per_run/$RUN_ID" \
    --oligo-manifest "$REPO/gen_data/mafia_oligos.tsv" \
    --run-manifest "$REPO/gen_data/mafia_runs.tsv" \
    --run-id "$RUN_ID" \
    --sample-type rna \
    --rna002 \
    --chunk-len 5000 \
    --overlap 500 \
    --workers 8
done


for RUN_ID in \
  WUE_splint_batch2_A_RTA \
  WUE_splint_batch2_m6A_RTA \
  WUE_splint_batch2_m6A_RTA_1 \
  WUE_splint_batch2_m6A_RTA_2
do
  python validate/evaluate_mafia_stage1.py "$MODEL_DIR" \
    --dataset-dir "$WORK_ROOT/chunks/per_run/$RUN_ID" \
    --split train \
    --weights 5 \
    --device cuda:0 \
    --batchsize 64 \
    --num-workers 4 \
    --write-sites \
    --output-dir "$MODEL_DIR/mafia_stage1_e5_heldout_$RUN_ID"
done


# check training set

python dataset_check/check_mafia_stage1_dataset.py \
  "$WORK_ROOT/chunks/stage1_train_mafia_wue_rl" \
  --output-dir "/home/lijy/workspace/TetraMod/dataset_check_res/stage1_train_mafia_wue_rl/check_reports"

WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
python dataset_check/plot_mafia_stage1_visuals.py \
  --motif-balance dataset_check_res/stage1_train_mafia_wue_rl/check_reports/motif_balance.tsv \
  --internal-eval-dir val_res/mafia_stage1_epoch5 \
  --heldout-glob 'val_res/mafia_stage1_e5_heldout_WUE_splint_batch2*' \
  --output-dir dataset_check_res/stage1_train_mafia_wue_rl/figures


# train a large train set with more motifs

WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
DATA_DIR="$WORK_ROOT/chunks/stage1_train_mafia_6motif_wue_batch2_rlmix1_4"
MODEL_DIR="$WORK_ROOT/models/stage1_mafia_6motif_wue_batch2_rlmix1_4"
BONITO_MODEL_DIR=/data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3
REPO=/home/lijy/workspace/TetraMod

BATCH=64
TRAIN_CHUNKS=$(python - <<PY
import numpy as np
n = np.load("$DATA_DIR/reference_lengths.npy", mmap_mode="r").shape[0]
print((n // $BATCH) * $BATCH)
PY
)
VALID_CHUNKS=$(python - <<PY
import numpy as np
print(np.load("$DATA_DIR/validation/reference_lengths.npy", mmap_mode="r").shape[0])
PY
)

cd "$REPO"

tetramod train_promote "$MODEL_DIR" \
  --config src/tetramod/models/configs/multihead_transformer.toml \
  --pretrained "$BONITO_MODEL_DIR" \
  --directory "$DATA_DIR" \
  --promote-stage control \
  --chunks "$TRAIN_CHUNKS" \
  --valid-chunks "$VALID_CHUNKS" \
  --batch 32 \
  --epochs 30 \
  --lr 1e-4 \
  --device cuda:0 \
  --num-workers 4 \
  --no-compile \
  > /home/lijy/workspace/TetraMod/log/train_log/tetramod_trainpromote_mafia_stage1_260513_6motif_dataset.log 2>&1