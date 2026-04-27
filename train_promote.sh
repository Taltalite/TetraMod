# curlcakes m6A dataset generation

bonito basecaller \
 /data/biolab-nvme-pcie2/lijy/bonito_models/rna004_130bps_sup@v5.2.0 \
 --device cuda:0 \
 --rna \
 --save-ctc \
 --reference /data/biolab-nvme-pcie2/lijy/curlcakes/curlcake_constructs_EcoRV_BamHI_digestion.fasta \
 --alignment-threads 8 \
 /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/pod5 \
 > /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/bonito/bonito_rnasup520.bam

/home/lijy/dorado-1.4.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/rna004_130bps_sup@v5.2.0 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/curlcake_constructs_EcoRV_BamHI_digestion.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/pod5 > /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/dorado_bam/dorado_rnasup520.bam


python gen_data/create_dataset_dorado_ctc_like.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/dorado_bam/dorado_rnasup520.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/pod5 \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/curlcake_constructs_EcoRV_BamHI_digestion.fasta \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/chunks/ \
    --sample-type rna \
    --chunk-len 12000 \
    --overlap 600 \
    --filter-preset relaxed \
    --norm-strategy from-bam \
    --workers 8 

python gen_data/make_mod_targets_m6a.py \
    --dataset-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/chunks/ \
    --mode full-mod \
    --non-a-policy ignore


/home/lijy/dorado-1.4.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/rna004_130bps_sup@v5.2.0 \
 --reference  /data/biolab-nvme-pcie2/lijy/curlcakes/curlcake_constructs_EcoRV_BamHI_digestion.fasta \
 --emit-moves \
 /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/pod5 > /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/dorado_bam/dorado_rnasup520.bam


python gen_data/create_dataset_dorado_ctc_like.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/dorado_bam/dorado_rnasup520.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/pod5 \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/curlcake_constructs_EcoRV_BamHI_digestion.fasta \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/chunks/ \
    --sample-type rna \
    --chunk-len 12000 \
    --overlap 600 \
    --filter-preset relaxed \
    --norm-strategy from-bam \
    --workers 8

python gen_data/make_mod_targets_m6a.py \
    --dataset-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/chunks/ \
    --mode canonical \
    --non-a-policy ignore 


python gen_data/merge_mod_datasets.py \
    --full-mod-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/chunks/ \
    --canonical-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/chunks/ \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mix/chunks/

tetramod train_promote -f /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/tetramod_model/mini_run/ \
    --directory /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mix/chunks/ \
    --config /home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained /data/biolab-nvme-pcie2/lijy/bonito_models/rna004_130bps_sup@v5.2.0 \
    --promote-stage control \
    --promote-base A \
    --epochs 10 \
    --batch 48 \
    --chunks 30000 \
    --valid-chunks 2000 \
    --device cuda:0

# control

tetramod train -f /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/tetramod_model/mini_run_control/ \
    --directory /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mix/chunks/ \
    --config /home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained /data/biolab-nvme-pcie2/lijy/bonito_models/rna004_130bps_sup@v5.2.0 \
    --epochs 10 \
    --batch 64 \
    --chunks 30000 \
    --valid-chunks 2000 \
    --device cuda:0


python validate/evaluate_promote_control.py \
  /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/tetramod_model/mini_run \
  --output-dir /home/lijy/workspace/TetraMod/val_res/curlcakes_m6a_mini_run/promote/ \
  --ivt-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/chunks \
  --full-mod-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/chunks \
  --dataset valid \
  --chunks 30000 \
  --valid-chunks 2000 \
  --batchsize 32 \
  --device cuda:0