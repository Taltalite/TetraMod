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