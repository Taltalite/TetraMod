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


# 3. Dorado RNA002 basecalling

DORADO_MODEL=/home/lijy/workspace/TetraMod/src/tetramod/models/rna002_70bps_sup@v3/rna002_70bps_sup@v3

/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller "$DORADO_MODEL" "$WORK_ROOT/pod5/$RUN_ID" \
  --emit-moves \
  --device cuda:0 \
  > "$WORK_ROOT/bam/$RUN_ID.bam"