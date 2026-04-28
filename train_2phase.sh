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



/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/rna002_70bps_hac@v3 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc0/ \
 > /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc0/rna002_hac_cc0.bam



/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/rna002_70bps_hac@v3 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc100/ \
 > /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/rna002_hac_cc100.bam


python gen_data/create_dataset_dorado_ctc_like.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc0/rna002_hac_cc0.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc0/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/0mod/ \
    --run-id mod_0_run1 \
    --sample-type rna \
    --rna002 \
    --chunk-len 10000 \
    --overlap 500 \
    --filter-preset relaxed \
    --metadata-kmer 5 \
    --workers 8

python gen_data/create_dataset_dorado_ctc_like.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/rna002_hac_cc100.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc100/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE246151_curlcake_constructs_EcoRV_BamHI_digestion.fasta \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/chunks/mod100/ \
    --run-id mod_100_run1 \
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

  4. 验证 0% / 100% 分离度

  python validate/evaluate_promote_control.py \
    "$MODEL" \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/val_res/stage1_control_run1 \
    --ivt-dir "$CC0" \
    --full-mod-dir "$CC100" \
    --dataset valid \
    --chunks 30000 \
    --valid-chunks 2000 \
    --batchsize 32 \
    --device cuda:0