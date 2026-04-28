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



下面是从 RNA002 POD5 + Dorado BAM 到 train_promote 的最小命令模板。假设 BAM 已由 RNA002 对应 Dorado 模型生成，并带有 --emit-moves --reference。

  1. 生成真实比例 LLP 数据集

  python gen_data/build_real_llp_from_ratio_ivt.py \
    --ratio-run 12.5:/data/rna002/ivt_12p5/dorado.bam:/data/rna002/ivt_12p5/pod5:rna002_12p5 \
    --ratio-run 25:/data/rna002/ivt_25/dorado.bam:/data/rna002/ivt_25/pod5:rna002_25 \
    --ratio-run 50:/data/rna002/ivt_50/dorado.bam:/data/rna002/ivt_50/pod5:rna002_50 \
    --ratio-run 75:/data/rna002/ivt_75/dorado.bam:/data/rna002/ivt_75/pod5:rna002_75 \
    --reference-fasta /data/ref/transcripts_or_genome.fa \
    --work-dir /data/rna002/tetramod_work/ratio_sources \
    --output-dir /data/rna002/tetramod_data/llp_real_ratio \
    --rna002 \
    --sample-type rna \
    --filter-preset relaxed \
    --heldout-mode leave-site \
    --leave-site-fraction 0.1 \
    --workers 8 \
    --seed 114514

  --rna002 会读取：

  src/tetramod/models/rna002_70bps_sup@v3/config.toml

  并使用其中的 RNA002 signal normalisation 参数。

  2. 训练 LLP 阶段

  tetramod train_promote -f /data/rna002/tetramod_model/llp_run1 \
    --directory /data/rna002/tetramod_data/llp_real_ratio \
    --config /home/lijy/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained /home/lijy/TetraMod/src/tetramod/models/rna002_70bps_sup@v3 \
    --promote-stage llp \
    --promote-base A \
    --epochs 20 \
    --batch 20 \
    --grad-accum-split 1 \
    --lr 5e-5 \
    --chunks 160000 \
    --valid-chunks 20000 \
    --profile-chunks 50000 \
    --device cuda:0 \
    > /home/lijy/TetraMod/log/train_log/rna002_m6a_llp_run1.log 2>&1

  3. 验证 LLP 单调性

  python validate/evaluate_llp_bags.py \
    /data/rna002/tetramod_model/llp_run1 \
    --directory /data/rna002/tetramod_data/llp_real_ratio \
    --dataset valid \
    --valid-chunks 20000 \
    --output-dir /home/lijy/TetraMod/val_res/rna002_llp_run1_all \
    --batchsize 20 \
    --device cuda:0

  如果你已有每个比例的 chunk dataset，也可以跳过 POD5/BAM 转换，直接用：

  python gen_data/build_real_llp_from_ratio_ivt.py \
    --ratio-dataset 12.5:/data/rna002/chunks_12p5 \
    --ratio-dataset 25:/data/rna002/chunks_25 \
    --ratio-dataset 50:/data/rna002/chunks_50 \
    --ratio-dataset 75:/data/rna002/chunks_75 \
    --work-dir /data/rna002/tetramod_work/ratio_sources \
    --output-dir /data/rna002/tetramod_data/llp_real_ratio \
    --heldout-mode leave-site \
    --leave-site-fraction 0.1