/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/rna002_70bps_hac@v3 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc125/ > /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc125/rna002_hac_cc125.bam


/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/rna002_70bps_hac@v3 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc25/ > /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc25/rna002_hac_cc25.bam


/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/rna002_70bps_hac@v3 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc50/ > /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc50/rna002_hac_cc50.bam


/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/rna002_70bps_hac@v3 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc75/ > /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc75/rna002_hac_cc75.bam



# 下面是从 RNA002 POD5 + Dorado BAM 到 train_promote 的最小命令模板。假设 BAM 已由 RNA002 对应 Dorado 模型生成，并带有 --emit-moves --reference。

# 1. 生成真实比例 LLP 数据集

# python gen_data/build_real_llp_from_ratio_ivt.py \
#     --ratio-run 12.5:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc125/rna002_hac_cc125.sorted.bam:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc125/:rna002_12p5 \
#     --ratio-run 25:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc25/rna002_hac_cc25.sorted.bam:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc25/:rna002_25 \
#     --ratio-run 50:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc50/rna002_hac_cc50.sorted.bam:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc50/:rna002_50 \
#     --ratio-run 75:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc75/rna002_hac_cc75.sorted.bam:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc75/:rna002_75 \
#     --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
#     --work-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/ratio_sources \
#     --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/llp_real_ratio \
#     --rna002 \
#     --sample-type rna \
#     --filter-preset relaxed \
#     --heldout-mode leave-site \
#     --leave-site-fraction 0.1 \
#     --workers 8 \
#     --seed 114514

python gen_data/build_real_llp_from_ratio_ivt.py \
    --ratio-run 12.5:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc125/rna002_hac_cc125.sorted.bam:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc125/:rna002_12p5 \
    --ratio-run 25:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc25/rna002_hac_cc25.sorted.bam:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc25/:rna002_25 \
    --ratio-run 50:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc50/rna002_hac_cc50.sorted.bam:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc50/:rna002_50 \
    --ratio-run 75:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc75/rna002_hac_cc75.sorted.bam:/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc75/:rna002_75 \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
    --work-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/ratio_sources \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/llp_real_ratio \
    --rna002 \
    --sample-type rna \
    --bagging-mode ratio-stratified \
    --match-fields contig,kmer_context,motif_context \
    --bag-size 20 \
    --min-bag-size 4 \
    --heldout-mode leave-site \
    --leave-site-fraction 0.1 \
    --filter-preset relaxed \
    --workers 8 \
    --seed 114514

# 完整逻辑流程

#   1. 每个比例的 BAM/POD5 先独立转成 chunk dataset。
#   2. --rna002 使用 RNA002 config 的 signal normalisation。
#   3. make_mod_targets_m6a.py --mode llp-candidate 只标记 A-head 候选位点，不制造 read-level 正负标签。
#   4. build_llp_mixture_dataset.py 在每个比例内部按 contig,kmer_context,motif_context 分组。
#   5. 每个组内 shuffle 后切成 bag，默认每 bag 20 reads，少于 4 reads 的尾 bag 丢弃。
#   6. 每条 read 写入：
#       - bag_keys.npy: 它属于哪个 LLP bag
#       - bag_targets.npy: 该 bag 的投料比例，如 0.125/0.25/0.5/0.75
#   7. 各比例默认平衡到相同 bag 数，避免某个比例样本量更大主导训练。
#   8. leave-site 从全体 site 中统一抽 heldout sites，所有比例都排除这些 sites 作为 validation，避免 site 泄漏。

#   2. 训练 LLP 阶段

tetramod train_promote -f /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/llp_rna002_run1 \
    --directory /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/llp_real_ratio \
    --config /home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained /data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3 \
    --promote-stage llp \
    --promote-base A \
    --llp-loss huber \
    --llp-tolerance 0.05 \
    --llp-huber-delta 0.05 \
    --epochs 20 \
    --batch 20 \
    --grad-accum-split 1 \
    --lr 5e-5 \
    --chunks 15606 \
    --valid-chunks 1470 \
    --profile-chunks 5000 \
    --device cuda:0 \
    > /home/lijy/workspace/TetraMod/log/train_log/rna002_m6a_llp_run1.log 2>&1

#   3. 验证 LLP 单调性

python validate/evaluate_llp_bags.py \
    /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/llp_rna002_run1 \
    --directory /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/llp_real_ratio \
    --dataset valid \
    --valid-chunks 1470 \
    --output-dir /home/lijy/TetraMod/val_res/rna002_llp_run1_all \
    --batchsize 20 \
    --device cuda:0 \
    --weights 20 \
    --no-compile


python validate/evaluate_llp_bags.py \
    /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/llp_rna002_run1 \
    --directory /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/llp_real_ratio \
    --dataset train \
    --chunks 15606 \
    --output-dir /home/lijy/workspace/TetraMod/val_res/rna002_llp_run1_train \
    --batchsize 20 \
    --device cuda:0 \
    --no-compile

# =============数据质量检查===================
 python validate/diagnose_llp_dataset.py \
    /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/llp_real_ratio \
    --output-dir /home/lijy/workspace/TetraMod/val_res/rna002_llp_dataset_diagnosis \
    --split all \
    --compare-ratios 50,75 \
    --top-k 30
# ===========================================

tetramod train_promote -f /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/llp_rna002_run2 \
    --directory /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/llp_real_ratio \
    --config /home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained /data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3 \
    --promote-stage llp \
    --promote-base A \
    --llp-loss huber \
    --llp-tolerance 0.05 \
    --llp-huber-delta 0.05 \
    --epochs 20 \
    --batch 20 \
    --grad-accum-split 1 \
    --lr 5e-5 \
    --chunks 15606 \
    --valid-chunks 1470 \
    --profile-chunks 5000 \
    --device cuda:0 \
    > /home/lijy/workspace/TetraMod/log/train_log/rna002_m6a_llp_run1.log 2>&1