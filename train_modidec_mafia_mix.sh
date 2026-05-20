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


python dataset_check/plot_mafia_stage1_visuals.py \
	--motif-balance dataset_check_res/stage1_train_mafia_modidec_m6a/check_reports/motif_balance.tsv \
	--internal-eval-dir val_res/stage1_mix_any_a_epoch6_internal_valid \
	--heldout-glob 'val_res/stage1_mix_any_a_epoch6_mafia_heldout_*' \
	--output-dir val_res/stage1_mix_any_a_epoch6_figures_all_motifs \
	--training-csv /data/biolab-nvme-pcie2/lijy/tetramod_models/stage1_mafia_modidec_any_a_neg_lr1e4_bs64_wd1e2/training.csv \
	--internal-label 'Mixed internal validation' \
	--heldout-label 'mAFiA final heldout' \
	--heldout-prefix stage1_mix_any_a_epoch6_mafia_heldout_ \
	--motifs AAACA,AGACA,AGACC,AGACT,GAACC,GAACT,GGACC,GGACA,GGACT,TAACG,TAACT,TGACT



# 7. New-logic MoDiDeC dataset

# 新逻辑只改变 MoDiDeC internal negative 的截取方式：
#   NEGATIVE_LABEL_MODE=center
#   NEGATIVE_MOTIF_MODE=positive-motifs
# negative 仍来自非 matched m6A oligo 区域，但只保留 centered 5-mer 属于 MoDiDeC m6A positive motif set 的 A 位点。
# 注意这里使用新的 WORK_ROOT 和 TRAIN_DATASET_NAME，避免覆盖旧逻辑数据集。

REPO=/home/lijy/workspace/TetraMod
MODIDEC_BAM=/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/bam/modidec_train.bam
MODIDEC_POD5=/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/m6A_pod5

cd "$REPO"

REPO="$REPO" \
WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002_motifneg \
TRAIN_DATASET_NAME=stage1_train_modidec_m6a_rna002_positive_motif_neg \
HELDOUT_ROOT_NAME=heldout_modidec_m6a_rna002_positive_motif_neg \
BUILD_HELDOUT=1 \
SKIP_EXISTING=0 \
RUN_DATASET_CHECK=1 \
NEGATIVE_LABEL_MODE=center \
NEGATIVE_MOTIF_MODE=positive-motifs \
MERGE_BALANCE_MODE=source-class \
MERGE_BALANCE_VALIDATION=1 \
MODIDEC_BAM_SPECS="modidec_train:$MODIDEC_BAM:$MODIDEC_POD5" \
MODIDEC_HELDOUT_BAM_SPECS="modidec_h11:$MODIDEC_BAM:$MODIDEC_POD5" \
bash ./train_modidec_m6a_stage1_dataset.sh


# 8. New-logic mixed dataset

MAFIA_DIR="/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002/chunks/stage1_train_mafia_6motif_wue_batch2_rlmix1_4"
MODIDEC_DIR="/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002_motifneg/chunks/stage1_train_modidec_m6a_rna002_positive_motif_neg"
MIX_DIR="/data/biolab-nvme-pcie2/lijy/tetramod_mix_rna002_motifneg/chunks/stage1_train_mafia_modidec_m6a_positive_motif_neg"

python gen_data/merge_mafia_stage1_datasets.py \
	--dataset "mafia:$MAFIA_DIR" \
	--dataset "modidec:$MODIDEC_DIR" \
	--output-dir "$MIX_DIR" \
	--valid-fraction 0.1 \
	--balance-mode source-class \
	--balance-validation \
	--seed 114514


# 9. New-logic mixed dataset QC

python dataset_check/check_mafia_stage1_dataset.py \
	"$MIX_DIR" \
	--output-dir /home/lijy/workspace/TetraMod/dataset_check_res/stage1_train_mafia_modidec_m6a_positive_motif_neg


# 10. Check new-logic dataset size before training

# 训练前先查看 summary，并把 --chunks 设置为 train.num_samples 向下取 batch=64 的整数倍；
# --valid-chunks 可以直接使用 validation.num_samples。

cat "$MIX_DIR/mafia_stage1_merge_summary.json"

# 例如如果 summary 显示：
#   train.num_samples = 37212
#   validation.num_samples = 4152
# 则使用：
#   --chunks 37184
#   --valid-chunks 4152


# 11. New-logic train_promote

PRETRAINED=/data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3
DATASET=/data/biolab-nvme-pcie2/lijy/tetramod_mix_rna002_motifneg/chunks/stage1_train_mafia_modidec_m6a_positive_motif_neg
OUT=/data/biolab-nvme-pcie2/lijy/tetramod_models/stage1_mafia_modidec_positive_motif_neg_lr1e4_bs64_wd1e2

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


# 12. New-logic internal validation

REPO=/home/lijy/workspace/TetraMod
MODEL=/data/biolab-nvme-pcie2/lijy/tetramod_models/stage1_mafia_modidec_positive_motif_neg_lr1e4_bs64_wd1e2
DATASET=/data/biolab-nvme-pcie2/lijy/tetramod_mix_rna002_motifneg/chunks/stage1_train_mafia_modidec_m6a_positive_motif_neg

cd "$REPO"

for E in 5 6 7; do
python validate/evaluate_mafia_stage1.py "$MODEL" \
	--dataset-dir "$DATASET" \
	--split validation \
	--weights "$E" \
	--device cuda:0 \
	--batchsize 64 \
	--num-workers 8 \
	--output-dir "$REPO/val_res/stage1_mix_positive_motif_neg_epoch${E}_internal_valid" \
	--write-sites
done


# 13. New-logic mAFiA heldout validation

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
	--output-dir "val_res/stage1_mix_positive_motif_neg_epoch6_mafia_heldout_${NAME}" \
	--write-sites
done


# 14. New-logic MoDiDeC #11 heldout validation

MODIDEC_H11=/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002_motifneg/chunks/heldout_modidec_m6a_rna002_positive_motif_neg/modidec_h11

python validate/evaluate_mafia_stage1.py "$MODEL" \
	--dataset-dir "$MODIDEC_H11" \
	--split train \
	--weights 6 \
	--device cuda:0 \
	--batchsize 64 \
	--num-workers 8 \
	--output-dir "val_res/stage1_mix_positive_motif_neg_epoch6_modidec_h11" \
	--write-sites


# 15. New-logic visualization

python dataset_check/plot_mafia_stage1_visuals.py \
	--motif-balance dataset_check_res/stage1_train_mafia_modidec_m6a_positive_motif_neg/check_reports/motif_balance.tsv \
	--internal-eval-dir val_res/stage1_mix_positive_motif_neg_epoch6_internal_valid \
	--heldout-glob 'val_res/stage1_mix_positive_motif_neg_epoch6_mafia_heldout_*' \
	--output-dir val_res/stage1_mix_positive_motif_neg_epoch6_figures_all_motifs \
	--training-csv /data/biolab-nvme-pcie2/lijy/tetramod_models/stage1_mafia_modidec_positive_motif_neg_lr1e4_bs64_wd1e2/training.csv \
	--internal-label 'Mixed internal validation' \
	--heldout-label 'mAFiA final heldout' \
	--heldout-prefix stage1_mix_positive_motif_neg_epoch6_mafia_heldout_ \
	--motifs AAACA,AGACA,AGACC,AGACT,GAACC,GAACT,GGACC,GGACA,GGACT,TAACG,TAACT,TGACT
