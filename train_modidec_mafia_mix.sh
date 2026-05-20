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



# 7. Finish new-logic MoDiDeC train dataset after heldout #11 produced no chunks

# 实际运行情况：
#   modidec_train 已成功写出 per-run dataset，例如 22060 chunks。
#   modidec_h11 在同一个 BAM/POD5 中没有匹配到 modidec_m6A_11，heldout 构建失败：
#       No valid mAFiA synthetic samples remained after filtering.
# 这不影响 mAFiA LOMO benchmark。先跳过 MoDiDeC #11 heldout，只完成 MoDiDeC train/validation merge。
#
# 注意：这个文件是命令清单，不是直接执行的 pipeline。按你的实际路径逐段运行。

REPO=/home/lijy/workspace/TetraMod
MODIDEC_BAM=/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/bam/modidec_train.bam
MODIDEC_POD5=/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/m6A_pod5

cd "$REPO"

REPO="$REPO" \
WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002_motifneg \
TRAIN_DATASET_NAME=stage1_train_modidec_m6a_rna002_positive_motif_neg \
HELDOUT_ROOT_NAME=heldout_modidec_m6a_rna002_positive_motif_neg \
BUILD_HELDOUT=0 \
SKIP_EXISTING=1 \
RUN_DATASET_CHECK=1 \
NEGATIVE_LABEL_MODE=center \
NEGATIVE_MOTIF_MODE=positive-motifs \
MERGE_BALANCE_MODE=source-class \
MERGE_BALANCE_VALIDATION=1 \
MODIDEC_BAM_SPECS="modidec_train:$MODIDEC_BAM:$MODIDEC_POD5" \
bash ./train_modidec_m6a_stage1_dataset.sh


# 8. Optional new-logic mAFiA + MoDiDeC mixed dataset, for comparison only

# 这一步不是 LOMO 的前置条件。它用于比较 new-logic MoDiDeC mixing 是否改善普通 heldout。
# 当前 source-class merge 会把各 source/class 组下采样到同一数量；不要把它当作最终泛化策略。

MAFIA_DIR=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002/chunks/stage1_train_mafia_6motif_wue_batch2_rlmix1_4
MODIDEC_DIR=/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002_motifneg/chunks/stage1_train_modidec_m6a_rna002_positive_motif_neg
MIX_DIR=/data/biolab-nvme-pcie2/lijy/tetramod_mix_rna002_motifneg/chunks/stage1_train_mafia_modidec_m6a_positive_motif_neg

python gen_data/merge_mafia_stage1_datasets.py \
	--dataset "mafia:$MAFIA_DIR" \
	--dataset "modidec:$MODIDEC_DIR" \
	--output-dir "$MIX_DIR" \
	--valid-fraction 0.1 \
	--balance-mode source-class \
	--balance-validation \
	--seed 114514

python dataset_check/check_mafia_stage1_dataset.py \
	"$MIX_DIR" \
	--output-dir "$REPO/dataset_check_res/stage1_train_mafia_modidec_m6a_positive_motif_neg/check_reports"

cat "$MIX_DIR/mafia_stage1_merge_summary.json"


# 9. Build mAFiA-only LOMO datasets

# LOMO 是当前 Stage 1 泛化主线：
#   train on 5 mAFiA motifs
#   test on the held-out motif using final heldout runs
# 默认 validation 也排除 held-out motif，避免模型选择阶段看到 held-out motif。

MAFIA_DIR=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002/chunks/stage1_train_mafia_6motif_wue_batch2_rlmix1_4
LOMO_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002/chunks/lomo_stage1_6motif
LOMO_MOTIFS=AGACT,GAACT,GGACA,GGACC,GGACT,TGACT

python gen_data/create_mafia_lomo_datasets.py "$MAFIA_DIR" \
	--output-root "$LOMO_ROOT" \
	--motifs "$LOMO_MOTIFS" \
	--validation-mode train-motifs \
	--force

cat "$LOMO_ROOT/lomo_datasets_summary.json"


# 10. LOMO dataset QC

for MOTIF in AGACT GAACT GGACA GGACC GGACT TGACT; do
python dataset_check/check_mafia_stage1_dataset.py \
	"$LOMO_ROOT/leave_${MOTIF}" \
	--output-dir "$REPO/dataset_check_res/lomo_stage1_6motif/leave_${MOTIF}/check_reports"
done


# 11. Train six mAFiA-only LOMO Stage 1 models

# 这一段可以一次性训练 6 个 leave-one-motif-out 模型。
# --chunks 按 lomo_datasets_summary.json 中 train.num_samples 向下取 batch=64 的整数倍；
# --valid-chunks 固定为 4096，减少验证开销。最终结论以后面的 final heldout eval 为准。
# 如果某个 motif 已有 weights_10.tar，会自动跳过，便于 CUDA 异常或中断后续跑。
# 如果某个 motif 训练失败，会立刻停止；不要在 CUDA launch failure 后继续提交后续训练。

PRETRAINED=/data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3
CONFIG="$REPO/src/tetramod/models/configs/multihead_transformer_promote_stage1_adamw.toml"
MODEL_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_models/lomo_stage1_mafia_6motif_lr1e4_bs64_wd1e2
VALID_CHUNKS=4096
FINAL_EPOCH=10
DEVICE=cuda:0

for MOTIF in AGACT GAACT GGACA GGACC GGACT TGACT; do
case "$MOTIF" in
	AGACT) CHUNKS=25600 ;;
	GAACT) CHUNKS=35136 ;;
	GGACA) CHUNKS=23744 ;;
	GGACC) CHUNKS=26944 ;;
	GGACT) CHUNKS=34368 ;;
	TGACT) CHUNKS=31360 ;;
	*) echo "[error] Unknown LOMO motif: $MOTIF" >&2; exit 1 ;;
esac

DATASET="$LOMO_ROOT/leave_${MOTIF}"
OUT="$MODEL_ROOT/leave_${MOTIF}"

if [[ -s "$OUT/weights_${FINAL_EPOCH}.tar" ]]; then
	echo "[skip LOMO] MOTIF=$MOTIF already has $OUT/weights_${FINAL_EPOCH}.tar"
	continue
fi

echo "[train LOMO] MOTIF=$MOTIF DATASET=$DATASET OUT=$OUT CHUNKS=$CHUNKS VALID_CHUNKS=$VALID_CHUNKS"

tetramod train_promote -f "$OUT" \
	--directory "$DATASET" \
	--config "$CONFIG" \
	--pretrained "$PRETRAINED" \
	--device "$DEVICE" \
	--promote-stage control \
	--promote-base A \
	--lr 1e-4 \
	--epochs "$FINAL_EPOCH" \
	--batch 64 \
	--chunks "$CHUNKS" \
	--valid-chunks "$VALID_CHUNKS" \
	--num-workers 8 \
	--seed 114514 \
	--grad-accum-split 1 \
	--save-optim-every 5 \
	--profile-chunks 20000 \
	--no-compile || {
		echo "[error] LOMO training failed for MOTIF=$MOTIF. Stop here, recover CUDA, then rerun this block." >&2
		exit 1
	}
done


# 12. Evaluate each LOMO model on mAFiA final heldout runs

# per-run heldout dataset 没有 validation/ 子目录，所以使用 --split train。
# 必须加 --write-sites；后面的 LOMO 汇总器用 site_predictions.tsv 计算跨 run ROC/PR AUC。

MAFIA_HELDOUT_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002/chunks/final_heldout_mix_1_4
LOMO_EVAL_ROOT="$REPO/val_res/lomo_stage1_mafia_6motif"
WEIGHTS=6

for MOTIF in AGACT GAACT GGACA GGACC GGACT TGACT; do
MODEL="$MODEL_ROOT/leave_${MOTIF}"
for D in "$MAFIA_HELDOUT_ROOT"/*; do
NAME=$(basename "$D")
python validate/evaluate_mafia_stage1.py "$MODEL" \
	--dataset-dir "$D" \
	--split train \
	--weights "$WEIGHTS" \
	--device cuda:0 \
	--batchsize 64 \
	--num-workers 8 \
	--output-dir "$LOMO_EVAL_ROOT/leave_${MOTIF}/heldout_${NAME}" \
	--write-sites
done
done


# 13. Aggregate LOMO benchmark

python validate/run_mafia_lomo_benchmark.py \
	--eval-root "$LOMO_EVAL_ROOT" \
	--output-dir "$LOMO_EVAL_ROOT/summary_epoch${WEIGHTS}" \
	--motifs AGACT,GAACT,GGACA,GGACC,GGACT,TGACT \
	--eval-glob 'heldout_*' \
	--threshold 0.5

cat "$LOMO_EVAL_ROOT/summary_epoch${WEIGHTS}/lomo_summary.tsv"


# 14. Optional: compare new-logic mixed model on normal mAFiA heldout

# 这一步只回答 "MoDiDeC positive-motif-neg mixing 是否改善普通 heldout"。
# 它不能替代 LOMO，也不能证明 wild-type generalization。

PRETRAINED=/data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3
DATASET=/data/biolab-nvme-pcie2/lijy/tetramod_mix_rna002_motifneg/chunks/stage1_train_mafia_modidec_m6a_positive_motif_neg
OUT=/data/biolab-nvme-pcie2/lijy/tetramod_models/stage1_mafia_modidec_positive_motif_neg_lr1e4_bs64_wd1e2
MIX_CHUNKS=SET_FROM_MAFIA_STAGE1_MERGE_SUMMARY_TRAIN_NUM_SAMPLES
MIX_VALID_CHUNKS=SET_FROM_MAFIA_STAGE1_MERGE_SUMMARY_VALIDATION_NUM_SAMPLES

tetramod train_promote -f "$OUT" \
	--directory "$DATASET" \
	--config "$CONFIG" \
	--pretrained "$PRETRAINED" \
	--device cuda:0 \
	--promote-stage control \
	--promote-base A \
	--lr 1e-4 \
	--epochs 20 \
	--batch 64 \
	--chunks "$MIX_CHUNKS" \
	--valid-chunks "$MIX_VALID_CHUNKS" \
	--num-workers 8 \
	--seed 114514 \
	--grad-accum-split 1 \
	--save-optim-every 5 \
	--profile-chunks 20000 \
	--no-compile

for D in "$MAFIA_HELDOUT_ROOT"/*; do
NAME=$(basename "$D")
python validate/evaluate_mafia_stage1.py "$OUT" \
	--dataset-dir "$D" \
	--split train \
	--weights 6 \
	--device cuda:0 \
	--batchsize 64 \
	--num-workers 8 \
	--output-dir "$REPO/val_res/stage1_mix_positive_motif_neg_epoch6_mafia_heldout_${NAME}" \
	--write-sites
done
