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


# 3. training

PRETRAINED=/data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3
DATASET=/data/biolab-nvme-pcie2/lijy/tetramod_mix_rna002/chunks/stage1_train_mafia_modidec_m6a
OUT=/data/biolab-nvme-pcie2/lijy/tetramod_models/stage1_mafia_modidec_any_a_neg_lr1e4_bs64_wd1e2

tetramod train_promote -f "$OUT" \
	--directory "$DATASET" \
	--config "/home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer_promote_stage1_adamw.toml" \
	--pretrained "$PRETRAINED" \
	--device cuda:0 \
	--promote-stage control \
	--promote-base A \
	--lr 1e-4 \
	--epochs 20 \
	--batch 64 \
	--chunks 37184 \
	--valid-chunks 4152 \
	--num-workers 8 \
	--seed 114514 \
	--grad-accum-split 1 \
	--save-optim-every 5 \
	--profile-chunks 20000 \
	--no-compile


# 4. Mixed Dataset Internal Eval

REPO=/home/lijy/workspace/TetraMod
MODEL=/data/biolab-nvme-pcie2/lijy/tetramod_models/stage1_mafia_modidec_any_a_neg_lr1e4_bs64_wd1e2
DATASET=/data/biolab-nvme-pcie2/lijy/tetramod_mix_rna002/chunks/stage1_train_mafia_modidec_m6a

cd "$REPO"

for E in 5 6 7; do
python validate/evaluate_mafia_stage1.py "$MODEL" \
	--dataset-dir "$DATASET" \
	--split validation \
	--weights "$E" \
	--device cuda:0 \
	--batchsize 64 \
	--num-workers 8 \
	--output-dir "$REPO/val_res/stage1_mix_any_a_epoch${E}_internal_valid" \
	--write-sites
done


# 5. mAFiA Heldout Eval

# 注意 per-run dataset 没有 validation/ 子目录，所以这里用 --split train。

# 如果你的 final heldout 在这个目录：

MAFIA_HELDOUT_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002/chunks/final_heldout_mix_1_4

for D in "$MAFIA_HELDOUT_ROOT"/*; do
NAME=$(basename "$D")
python validate/evaluate_mafia_stage1.py "$MODEL" \
	--dataset-dir "$D" \
	--split train \
	--weights 6 \
	--device cuda:0 \
	--batchsize 64 \
	--num-workers 8 \
	--output-dir "val_res/stage1_mix_any_a_epoch6_mafia_heldout_${NAME}" \
	--write-sites
done


# 6. MoDiDeC #11 Heldout Eval

#   如果构建了 #11 heldout：

MODIDEC_H11=/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/chunks/heldout_modidec_m6a_rna002/modidec_h11

python validate/evaluate_mafia_stage1.py "$MODEL" \
	--dataset-dir "$MODIDEC_H11" \
	--split train \
	--weights 6 \
	--device cuda:0 \
	--batchsize 64 \
	--num-workers 8 \
	--output-dir "val_res/stage1_mix_any_a_epoch6_modidec_h11" \
	--write-sites

