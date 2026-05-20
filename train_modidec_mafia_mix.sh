# 1. mix dataset

MAFIA_DIR="/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002/chunks/stage1_train_mafia_6motif_wue_batch2_rlmix1_4"
MODIDEC_DIR="/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/chunks/stage1_train_modidec_m6a_rna002"
MIX_DIR="/data/biolab-nvme-pcie2/lijy/tetramod_mix_rna002/chunks/stage1_train_mafia_modidec_m6a"

python gen_data/merge_mafia_stage1_datasets.py \
    --dataset "mafia:$MAFIA_DIR" \
    --dataset "modidec:$MODIDEC_DIR" \
    --output-dir "$MIX_DIR" \
    --valid-fraction 0.1 \
    --no-balance-train \
    --seed 114514


# 2. Mixed dataset QC

python dataset_check/check_mafia_stage1_dataset.py \
    "$MIX_DIR" \
    --output-dir /home/lijy/workspace/TetraMod/dataset_check_res/stage1_train_mafia_modidec_m6a