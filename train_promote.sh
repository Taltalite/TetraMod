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


# stage LLP

python gen_data/build_synthetic_llp_from_controls.py \
    --full-mod-bam /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/dorado_bam/dorado_rnasup520.bam \
    --full-mod-pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mod_100/pod5 \
    --canonical-bam /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/dorado_bam/dorado_rnasup520.bam \
    --canonical-pod5-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/canonical/pod5 \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/curlcakes/curlcake_constructs_EcoRV_BamHI_digestion.fasta \
    --work-dir /data/biolab-nvme-pcie2/lijy/curlcakes/llp_work \
    --output-dir /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mix/llp \
    --full-mod-run-id fullmod_run1 \
    --canonical-run-id canonical_run1 \
    --ratios 0,25,50,75,100 \
    --bag-size 20 \
    --bags-per-stratum 1 \
    --allow-replacement \
    --heldout-mode leave-site \
    --leave-site-fraction 0.1 \
    --sample-type rna \
    --chunk-len 12000 \
    --overlap 600 \
    --filter-preset relaxed \
    --norm-strategy from-bam \
    --metadata-kmer 5 \
    --workers 8 \
    --seed 114514


tetramod train_promote -f /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/tetramod_model/llp_run1 \
    --directory /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mix/llp \
    --config /home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained /data/biolab-nvme-pcie2/lijy/bonito_models/rna004_130bps_sup@v5.2.0 \
    --promote-stage llp \
    --promote-base A \
    --epochs 20 \
    --batch 20 \
    --grad-accum-split 1 \
    --lr 5e-5 \
    --chunks 167300 \
    --valid-chunks 20500 \
    --profile-chunks 50000 \
    --device cuda:0 \
    > /home/lijy/workspace/TetraMod/log/train_log/rna004_curlcakes_m6a_llp_run1.log 2>&1

#  然后验证：

python validate/evaluate_llp_bags.py /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/tetramod_model/llp_run1 \
    --directory /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mix/llp \
    --dataset valid \
    --output-dir /home/lijy/workspace/TetraMod/val_res/llp_run1 \
    --batchsize 20 \
    --device cuda:0

python validate/evaluate_llp_bags.py /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/tetramod_model/llp_run1 \
    --directory /data/biolab-nvme-pcie2/lijy/curlcakes/m6A/mix/llp \
    --dataset valid \
    --valid-chunks 20500 \
    --output-dir /home/lijy/workspace/TetraMod/val_res/llp_run1_all \
    --batchsize 20 \
    --device cuda:0