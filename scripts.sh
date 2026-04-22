tetramod train -f /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_mix_tetra \
    --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/mix_PAW43156_92158b33_73a20312_0+10_ctclike/ \
    --config /home/lijy/workspace/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained /data/biolab-nvme-pcie2/lijy/bonito_models/rna004_130bps_sup@v5.2.0 \
    --epochs 30 \
    --batch 48 \
    --lr 1e-4 \
    --chunks 50000 \
    --valid-chunks 5000 \
    --device cuda:0 \
    > /home/lijy/workspace/TetraMod/log/train_log/rna004_m6a_mix_tetra.log 2>&1


python validate/evaluate_train_mod.py \
    --model-directory /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_mix_tetra \
    --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/mix_PAW43156_92158b33_73a20312_0+10_ctclike/ \
    --output-dir /home/lijy/workspace/TetraMod/val_res/rna004_m6a_mix_tetra \
    --dataset valid \
    --chunks 50000 \
    --valid-chunks 5000 \
    --batchsize 48 \
    --device cuda:0 \
    --mod-threshold 0.5

tetramod basecaller \
    /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_mix_tetra \
    /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/wt_PRJEB80229_open_data/pod5/ \
    --device cuda:0 \
    --recursive \
    --rna \
    --max-reads 200 \
    --reference /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    > /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/wt_PRJEB80229_open_data/tetramod_out/tetramod_basecaller_test.bam


python validate/compare_basecaller_bams.py \
    --tetramod-bam /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/wt_PRJEB80229_open_data/tetramod_out/tetramod_basecaller_test.bam \
    --bonito-bam /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/wt_PRJEB80229_open_data/bonito_bam/PAU05273_pass_fd81c83d_c90ac4b0_10.bam \
    --output-dir /home/lijy/workspace/TetraMod/val_res/rna004_m6a_mix_tetra/bam_compare/