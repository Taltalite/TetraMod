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