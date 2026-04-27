#!/usr/bin/env bash
set -euo pipefail

# Example command template for Dorado-BAM-driven Bonito train_mod datasets.
# Replace paths before use.

# mod dataset
python gen_data/create_dataset_dorado_ctc_like.py \
  --bam-file /data/path/to/mod/basecaller_with_moves.bam \
  --pod5-dir /data/path/to/mod/pod5 \
  --reference-fasta /data/path/to/reference.fa \
  --output-dir /data/path/to/mod_dataset \
  --run-id mod_run_001 \
  --sample-type rna \
  --chunk-len 12000 \
  --overlap 600 \
  --filter-preset strict \
  --norm-strategy from-bam \
  --metadata-kmer 5 \
  --workers 8

python gen_data/make_mod_targets_m6a.py \
  --dataset-dir /data/path/to/mod_dataset \
  --mode full-mod \
  --non-a-policy ignore


# canonical dataset
python gen_data/create_dataset_dorado_ctc_like.py \
  --bam-file /data/path/to/canonical/basecaller_with_moves.bam \
  --pod5-dir /data/path/to/canonical/pod5 \
  --reference-fasta /data/path/to/reference.fa \
  --output-dir /data/path/to/canonical_dataset \
  --run-id canonical_run_001 \
  --sample-type rna \
  --chunk-len 12000 \
  --overlap 600 \
  --filter-preset strict \
  --norm-strategy from-bam \
  --metadata-kmer 5 \
  --workers 8

python gen_data/make_mod_targets_m6a.py \
  --dataset-dir /data/path/to/canonical_dataset \
  --mode canonical \
  --non-a-policy ignore


# merge into one train_mod dataset
python gen_data/merge_mod_datasets.py \
  --full-mod-dir /data/path/to/mod_dataset \
  --canonical-dir /data/path/to/canonical_dataset \
  --output-dir /data/path/to/mix_dataset


# Synthetic LLP mixtures from only 100% full-mod and 0% canonical controls.
# This wrapper can generate source chunk datasets from Dorado BAM/POD5, create
# source mod_targets.npy, then synthesize 0/25/50/75/100 bag-level mixtures.
python gen_data/build_synthetic_llp_from_controls.py \
  --full-mod-bam /data/path/to/full_mod/basecaller_with_moves.bam \
  --full-mod-pod5-dir /data/path/to/full_mod/pod5 \
  --canonical-bam /data/path/to/canonical/basecaller_with_moves.bam \
  --canonical-pod5-dir /data/path/to/canonical/pod5 \
  --reference-fasta /data/path/to/reference.fa \
  --work-dir /data/path/to/synthetic_llp_work \
  --output-dir /data/path/to/synthetic_llp_leave_site \
  --full-mod-run-id full_mod_run_001 \
  --canonical-run-id canonical_run_001 \
  --ratios 0,25,50,75,100 \
  --bag-size 20 \
  --bags-per-stratum 1 \
  --heldout-mode leave-site \
  --leave-site-fraction 0.1 \
  --match-fields primary_site_key,kmer_context,motif_context,qscore_bin,coverage_bin \
  --qscore-bins 8,10,12,14,16 \
  --coverage-bins 0.85,0.9,0.95,0.98 \
  --sample-type rna \
  --chunk-len 12000 \
  --overlap 600 \
  --filter-preset strict \
  --norm-strategy from-bam \
  --metadata-kmer 5 \
  --workers 8 \
  --seed 1

# If source chunk datasets already exist, skip BAM/POD5 chunking and synthesize
# directly from them.
python gen_data/build_synthetic_llp_from_controls.py \
  --full-mod-dataset /data/path/to/mod_dataset \
  --canonical-dataset /data/path/to/canonical_dataset \
  --work-dir /data/path/to/synthetic_llp_work \
  --output-dir /data/path/to/synthetic_llp_leave_run \
  --ratios 0,25,50,75,100 \
  --bag-size 20 \
  --bags-per-stratum 1 \
  --heldout-mode leave-run \
  --heldout-run run_to_hold_out \
  --seed 2


# LLP mixture dataset from known-ratio datasets generated with the command above.
# Each ratio directory must contain metadata.npz and mod_targets.npy.
python gen_data/build_llp_mixture_dataset.py \
  --ratio-dataset 0:/data/path/to/ratio_0_dataset \
  --ratio-dataset 25:/data/path/to/ratio_25_dataset \
  --ratio-dataset 50:/data/path/to/ratio_50_dataset \
  --ratio-dataset 75:/data/path/to/ratio_75_dataset \
  --ratio-dataset 100:/data/path/to/ratio_100_dataset \
  --output-dir /data/path/to/llp_leave_run_dataset \
  --heldout-mode leave-run \
  --heldout-run run_to_hold_out \
  --qscore-bins 8,10,12,14,16 \
  --coverage-bins 0.85,0.9,0.95,0.98 \
  --seed 1

python gen_data/build_llp_mixture_dataset.py \
  --ratio-dataset 0:/data/path/to/ratio_0_dataset \
  --ratio-dataset 25:/data/path/to/ratio_25_dataset \
  --ratio-dataset 50:/data/path/to/ratio_50_dataset \
  --ratio-dataset 75:/data/path/to/ratio_75_dataset \
  --ratio-dataset 100:/data/path/to/ratio_100_dataset \
  --output-dir /data/path/to/llp_leave_site_dataset \
  --heldout-mode leave-site \
  --leave-site-fraction 0.1 \
  --qscore-bins 8,10,12,14,16 \
  --coverage-bins 0.85,0.9,0.95,0.98 \
  --seed 2


# Train promoted LLP on a prepared LLP dataset. bag_targets.npy supplies each
# bag's known ratio, so --llp-proportion is not needed for multi-ratio training.
tetramod train_promote -f /data/path/to/training_model/rna004_m6a_llp \
  --directory /data/path/to/llp_leave_run_dataset \
  --config /home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
  --pretrained /home/lijy/workspace/bonito/bonito/models/rna004_130bps_sup@v5.2.0 \
  --promote-stage llp \
  --promote-base A \
  --epochs 10 \
  --batch 48 \
  --lr 5e-5 \
  --chunks 300000 \
  --valid-chunks 20000 \
  --device cuda:0

# Hard check 1: bag score monotonicity across 0/25/50/75/100.
# Hard check 2: run the same command on the leave-run and leave-site datasets.
python validate/evaluate_llp_bags.py /data/path/to/training_model/rna004_m6a_llp \
  --directory /data/path/to/llp_leave_run_dataset \
  --dataset valid \
  --output-dir /data/path/to/llp_eval_leave_run \
  --batchsize 32 \
  --device cuda:0

python validate/evaluate_llp_bags.py /data/path/to/training_model/rna004_m6a_llp \
  --directory /data/path/to/llp_leave_site_dataset \
  --dataset valid \
  --output-dir /data/path/to/llp_eval_leave_site \
  --batchsize 32 \
  --device cuda:0


# train
bonito train_mod -f /data/path/to/training_model/rna004_m6a_mix_ft \
  --directory /data/path/to/mix_dataset \
  --config /home/lijy/workspace/bonito/bonito/models/configs/multihead_transformer.toml \
  --pretrained /home/lijy/workspace/bonito/bonito/models/rna004_130bps_sup@v5.2.0 \
  --epochs 30 \
  --batch 48 \
  --lr 5e-5 \
  --chunks 300000 \
  --valid-chunks 20000 \
  --device cuda:0


# validate
python validate/evaluate_train_mod.py \
  --model_directory /data/path/to/training_model/rna004_m6a_mix_ft \
  --directory /data/path/to/mix_dataset \
  --dataset valid \
  --chunks 300000 \
  --valid-chunks 20000 \
  --batchsize 32 \
  --device cuda:0
