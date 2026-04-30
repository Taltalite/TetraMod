# convert fast5 to pod5

python gen_data/convert_fast5_tar_to_pod5.py \
    /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/fast5/unmod/RNAAB089716.fast5.tar.gz.4 \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc0/ \
    --jobs 1 \
    --force


python gen_data/convert_fast5_tar_to_pod5.py \
    /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/fast5/allmod/RNAAB090763.fast5.tar.gz.3 \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc100/ \
    --jobs 1 \
    --force



/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /home/lijy/workspace/TetraMod/src/tetramod/models/rna002_70bps_sup@v3/rna002_70bps_sup@v3 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE124309_FASTA_sequences_of_Curlcakes_4pole.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc0/ \
 > /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc0/rna002_sup_cc0_4pole.bam



/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /home/lijy/workspace/TetraMod/src/tetramod/models/rna002_70bps_sup@v3/rna002_70bps_sup@v3 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE124309_FASTA_sequences_of_Curlcakes_4pole.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc100/ \
 > /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/rna002_sup_cc100_4pole.bam


python gen_data/create_dataset_dorado_ctc_like.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc0/rna002_sup_cc0_4pole.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc0/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE124309_FASTA_sequences_of_Curlcakes_4pole.fasta \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/0mod_sup/ \
    --run-id mod_0_suprun1 \
    --sample-type rna \
    --rna002 \
    --chunk-len 10000 \
    --overlap 500 \
    --filter-preset relaxed \
    --metadata-kmer 5 \
    --workers 8

python gen_data/create_dataset_dorado_ctc_like.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/rna002_sup_cc100_4pole.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc100/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE124309_FASTA_sequences_of_Curlcakes_4pole.fasta \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod100_sup/ \
    --run-id mod_100_suprun1 \
    --sample-type rna \
    --rna002 \
    --chunk-len 10000 \
    --overlap 500 \
    --filter-preset relaxed \
    --metadata-kmer 5 \
    --workers 8

samtools fastq -f 4 /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/rna002_hac_cc100.sorted.bam | head -4000 > unmapped_1000.fastq
minimap2 -ax map-ont /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta unmapped_1000.fastq | samtools flagstat -

# 只用 cc100 mapped reads 做救援测试
samtools view -b -F 4 rna002_hac_cc100.sorted.bam > cc100.mapped_only.bam
samtools index cc100.mapped_only.bam

python gen_data/create_dataset_dorado_ctc_like.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/cc100.mapped_only.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc100/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod100_dna/ \
    --run-id mod_100_run1 \
    --sample-type dna \
    --rna002 \
    --chunk-len 10000 \
    --overlap 500 \
    --filter-preset relaxed \
    --mm2-preset map-ont \
    --metadata-kmer 5 \
    --workers 8


python gen_data/create_dataset_bam_aligned_highmod.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/cc100.mapped_only.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc100/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod100_bam_aligned_debug \
    --run-id mod_100_run1 \
    --sample-type rna \
    --rna002 \
    --chunk-len 10000 \
    --overlap 500 \
    --filter-preset relaxed \
    --workers 8


# 260428 training pipeline

CC0=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod0
CC100=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod100_bam_aligned_debug
MIX=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/stage1_control_mix
MODEL=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage1_control_run1
PRETRAINED=/data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3
CONFIG=/home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml

#   1. 给 cc0 / cc100 生成标签

python gen_data/make_mod_targets_m6a.py \
    --dataset-dir "$CC0" \
    --mode canonical \
    --non-a-policy ignore

python gen_data/make_mod_targets_m6a.py \
    --dataset-dir "$CC100" \
    --mode full-mod \
    --non-a-policy ignore

#   2. 合并 0% 和 100% control dataset

python gen_data/merge_mod_datasets.py \
    --canonical-dir "$CC0" \
    --full-mod-dir "$CC100" \
    --output-dir "$MIX" \
    --seed 114514

#   3. 训练 Stage 1 promoted control warm-up

tetramod train_promote -f "$MODEL" \
    --directory "$MIX" \
    --config "$CONFIG" \
    --pretrained "$PRETRAINED" \
    --promote-stage control \
    --promote-base A \
    --epochs 20 \
    --batch 64 \
    --lr 1e-4 \
    --chunks 253400 \
    --valid-chunks 20000 \
    --device cuda:0 \
    --profile-chunks 100000 \
    > /home/lijy/workspace/TetraMod/log/train_log/rna002_m6a_stage1_control_run1.log 2>&1

#   如果显存不够，把 --batch 48 降到 32 或 24。

#   4. 验证 0% / 100% 分离度

python validate/evaluate_promote_control.py \
    "$MODEL" \
    --output-dir /home/lijy/workspace/TetraMod/val_res/rna002_m6a_stage1_control_run1 \
    --ivt-dir "$CC0" \
    --full-mod-dir "$CC100" \
    --dataset valid \
    --chunks 253400 \
    --valid-chunks 20000 \
    --batchsize 32 \
    --device cuda:0

# 检查数据集生成bags情况

python validate/diagnose_llp_dataset.py \
    /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/llp_real_ratio \
    --output-dir /home/lijy/workspace/TetraMod/val_res/rna002_llp_dataset_diagnosis \
    --split all \
    --compare-ratios 50,75 \
    --top-k 30

# 用highmod去做IVT ratio input数据

REF=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion_4ratio.fasta


python gen_data/create_dataset_bam_aligned_highmod.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc12p5/rna002_hac_cc12p5.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc12p5 \
    --reference-fasta "$REF" \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod12p5 \
    --run-id rna002_cc12p5 \
    --sample-type rna \
    --filter-preset relaxed \
    --chunk-len 10000 \
    --overlap 500 \
    --rna002 \
    --workers 8


python gen_data/create_dataset_bam_aligned_highmod.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc25/rna002_hac_cc25.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc25 \
    --reference-fasta "$REF" \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod25 \
    --run-id rna002_cc25 \
    --sample-type rna \
    --filter-preset relaxed \
    --chunk-len 10000 \
    --overlap 500 \
    --rna002 \
    --workers 8


python gen_data/create_dataset_bam_aligned_highmod.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc50/rna002_hac_cc50.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc50 \
    --reference-fasta "$REF" \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod50 \
    --run-id rna002_cc50 \
    --sample-type rna \
    --filter-preset relaxed \
    --chunk-len 10000 \
    --overlap 500 \
    --rna002 \
    --workers 8


python gen_data/create_dataset_bam_aligned_highmod.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc75/rna002_hac_cc75.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc75 \
    --reference-fasta "$REF" \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod75 \
    --run-id rna002_cc75 \
    --sample-type rna \
    --filter-preset relaxed \
    --chunk-len 10000 \
    --overlap 500 \
    --rna002 \
    --workers 8

# 这些不是强监督标签。这里的 mod_targets.npy 只是告诉后续训练“哪些 A 位点是可参与 LLP 聚合的 candidate site”。

CHUNKS_SRC=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/

for d in "$CHUNKS_SRC"/mod12p5 "$CHUNKS_SRC"/mod25 "$CHUNKS_SRC"/mod50 "$CHUNKS_SRC"/mod75; do
python gen_data/make_mod_targets_m6a.py \
    --dataset-dir "$d" \
    --mode llp-candidate \
    --non-a-policy ignore
done

#   - --bagging-mode ratio-stratified：适合真实 ratio-IVT。它在每个 ratio 内部分层组 bag，不强迫每个 ratio 必须拥有完全相同的 site。真实实验里 ratio/run 已经混杂，强行 common-strata 往往会丢掉大量数据。
#   - --match-fields contig,kmer_context,motif_context：控制主要序列上下文，避免 bag 内 context 太杂；但不加入 primary_site_key，否则 bag 会被切得太碎。
#   - --heldout-mode leave-site：验证集按 site 留出，比随机 split 更严格。随机 split 容易让同一 reference site 同时出现在 train/valid，验证会偏乐观。
#   - 默认 ratio balance 是开启的，不要加 --no-balance-ratios。这会让每个 ratio 选中的 bag 数一致，避免高产量比例主导训练。

LLP=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/stage2_llp_ratio_aligned_highmod_bag20

python gen_data/build_llp_mixture_dataset.py \
    --ratio-dataset 12.5:"$CHUNKS_SRC/mod12p5" \
    --ratio-dataset 25:"$CHUNKS_SRC/mod25" \
    --ratio-dataset 50:"$CHUNKS_SRC/mod50" \
    --ratio-dataset 75:"$CHUNKS_SRC/mod75" \
    --output-dir "$LLP" \
    --bagging-mode ratio-stratified \
    --match-fields contig,kmer_context,motif_context \
    --bag-size 20 \
    --min-bag-size 20 \
    --heldout-mode leave-site \
    --leave-site-fraction 0.1 \
    --qscore-bins 8,10,12,14,16 \
    --coverage-bins 0.70,0.80,0.90,0.95,0.98 \
    --seed 114514


# QC dataset

python validate/diagnose_llp_dataset.py \
    "$LLP" \
    --output-dir /home/lijy/workspace/TetraMod/val_res/rna002_llp_aligned_highmod_bag20

# train llp

LLP=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/stage2_llp_ratio_aligned_highmod_bag20
MODEL_CONFIG=/home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml
tetramod train_promote -f /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage2_llp_run1 \
    --directory "$LLP" \
    --config "$MODEL_CONFIG" \
    --pretrained /data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3 \
    --init-promote-checkpoint /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage1_control_run1/ \
    --promote-stage llp \
    --promote-base A \
    --llp-loss huber \
    --llp-tolerance 0.025 \
    --llp-huber-delta 0.05 \
    --epochs 10 \
    --batch 64 \
    --no-amp \
    --lr 1e-5 \
    --chunks 283120 \
    --valid-chunks 28080 \
    --device cuda:0 \
    --profile-chunks 100000 \
    > /home/lijy/workspace/TetraMod/log/train_log/rna002_m6a_stage2_llp_run1.log 2>&1


python validate/evaluate_llp_bags.py /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage2_llp_run1 \
    --directory "$LLP" \
    --dataset valid \
    --device cuda:0 \
    --batchsize 64 \
    --chunks 283120 \
    --valid-chunks 28080 \
    --no-compile \
    --output-dir /home/lijy/workspace/TetraMod/val_res/stage2_llp_run1_evaluate_llp_bags


python validate/evaluate_llp_bags.py /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage1_control_run1 \
    --directory "$LLP" \
    --dataset valid \
    --device cuda:0 \
    --batchsize 64 \
    --chunks 283120 \
    --valid-chunks 28080 \
    --no-compile \
    --output-dir /home/lijy/workspace/TetraMod/val_res/stage1_control_run1_evaluate_baseline_bags


python validate/evaluate_promote_control.py /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage1_control_run1 \
    --ivt-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod0 \
    --full-mod-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod100_bam_aligned_debug \
    --dataset valid \
    --device cuda:0 \
    --batchsize 64 \
    --chunks 253400 \
    --valid-chunks 20000 \
    --no-compile \
    --output-dir /home/lijy/workspace/TetraMod/val_res/stage1_control_run1_control_eval


# visualization

python validate/evaluate_promote_control.py /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage2_llp_run1 \
    --ivt-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod0 \
    --full-mod-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod100_bam_aligned_debug \
    --dataset valid \
    --device cuda:0 \
    --batchsize 64 \
    --chunks 253400 \
    --valid-chunks 20000 \
    --no-compile \
    --output-dir /home/lijy/workspace/TetraMod/val_res/stage2_llp_run1_control_eval


python vis/plot_eval_results.py \
  --stage1-llp-dir /home/lijy/workspace/TetraMod/val_res/stage1_control_run1_evaluate_baseline_bags \
  --stage2-llp-dir /home/lijy/workspace/TetraMod/val_res/stage2_llp_run1_evaluate_llp_bags \
  --stage1-control-dir /home/lijy/workspace/TetraMod/val_res/stage1_control_run1_control_eval \
  --stage2-control-dir /home/lijy/workspace/TetraMod/val_res/stage2_llp_run1_control_eval \
  --output-dir /home/lijy/workspace/TetraMod/val_res/vis/


python vis/plot_control_labeled_eval.py \
    --stage1-control-dir /home/lijy/workspace/TetraMod/val_res/stage1_control_run1_control_eval \
    --stage2-control-dir /home/lijy/workspace/TetraMod/val_res/stage2_llp_run1_control_eval \
    --output-dir /home/lijy/workspace/TetraMod/val_res/vis_control_out


CONTROL_MIX=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/stage1_control_mix
STAGE1=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage1_control_run1
STAGE2=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage2_llp_run1

python validate/evaluate_train_mod.py \
    --model-directory "$STAGE1" \
    --directory "$CONTROL_MIX" \
    --dataset valid \
    --chunks 253400 \
    --valid-chunks 20000 \
    --batchsize 64 \
    --device cuda:0 \
    --mod-threshold 0.5 \
    --site-report-limit 200000 \
    --signal-example-limit 0 \
    --no-compile \
    --output-dir val_res/stage1_control_run1_train_mod_eval

python validate/evaluate_train_mod.py \
    --model-directory "$STAGE2" \
    --directory "$CONTROL_MIX" \
    --dataset valid \
    --chunks 253400 \
    --valid-chunks 20000 \
    --batchsize 64 \
    --device cuda:0 \
    --mod-threshold 0.5 \
    --site-report-limit 200000 \
    --signal-example-limit 0 \
    --output-dir val_res/stage2_llp_run1_train_mod_eval


python vis/plot_control_labeled_eval.py \
    --stage1-control-dir val_res/stage1_control_run1_control_eval \
    --stage2-control-dir val_res/stage2_llp_run1_control_eval \
    --stage1-sites-tsv val_res/stage1_control_run1_train_mod_eval/mod_site_examples.tsv \
    --stage2-sites-tsv val_res/stage2_llp_run1_train_mod_eval/mod_site_examples.tsv \
    --output-dir val_res/vis_control_out_with_sites


LLP=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/stage2_llp_ratio_aligned_highmod_bag20
MODEL_CONFIG=/home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml
tetramod train_promote -f /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage2_llp_run2 \
    --directory "$LLP" \
    --config "$MODEL_CONFIG" \
    --pretrained /data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3 \
    --init-promote-checkpoint /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage1_control_run1/ \
    --promote-stage llp \
    --promote-base A \
    --llp-loss huber \
    --llp-tolerance 0.025 \
    --llp-huber-delta 0.05 \
    --epochs 10 \
    --batch 64 \
    --no-amp \
    --lr 1e-5 \
    --chunks 283120 \
    --valid-chunks 28080 \
    --device cuda:0 \
    --profile-chunks 100000 \
    > /home/lijy/workspace/TetraMod/log/train_log/rna002_m6a_stage2_llp_run2.log 2>&1

CONTROL_MIX=/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/stage1_control_mix

python validate/evaluate_control_bags.py \
    /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage1_control_run1 \
    --directory "$CONTROL_MIX" \
    --dataset valid \
    --chunks 253400 \
    --valid-chunks 20000 \
    --output-dir /home/lijy/workspace/TetraMod/val_res/stage1_control_0_100_bags \
    --batchsize 64 \
    --bag-size 20 \
    --device cuda:0 \
    --no-compile

python validate/evaluate_control_bags.py \
    /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage1_control_run1 \
    --directory "$CONTROL_MIX" \
    --dataset valid \
    --chunks 253400 \
    --valid-chunks 20000 \
    --output-dir val_res/stage1_control_readlevel_bags \
    --batchsize 64 \
    --bag-size 1 \
    --device cuda:0 \
    --no-compile

python vis/plot_bag_level_roc.py \
    --stage1-bags /home/lijy/workspace/TetraMod/val_res/stage1_control_0_100_bags \
    --output-dir /home/lijy/workspace/TetraMod/val_res/vis_bag_roc

python vis/plot_bag_level_roc.py \
    --stage1-bags /home/lijy/workspace/TetraMod/val_res/stage1_control_readlevel_bags \
    --output-dir /home/lijy/workspace/TetraMod/val_res/vis_bag_roc_read



python vis/plot_bag_level_roc.py \
    --stage2-bags /home/lijy/workspace/TetraMod/val_res/stage2_llp_run1_evaluate_llp_bags \
    --output-dir val_res/vis_bag_roc