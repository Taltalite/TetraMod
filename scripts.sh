tetramod train -f /data/biolab-nvme-pcie2/lijy/m6A/training_model/rna004_m6a_mix_tetra \
    --directory /data/biolab-nvme-pcie2/lijy/m6A/dorado_rna004_sup/mix/dataset/mix_PAW43156_92158b33_73a20312_0+10_ctclike/ \
    --config /home/lijy/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained /home/lijy/workspace/bonito/bonito/models/rna004_130bps_sup@v5.2.0 \
    --epochs 30 \
    --batch 64 \
    --lr 1e-4 \
    --chunks 50000 \
    --valid-chunks 5000 \
    --device cuda:0 \
    > /home/lijy/workspace/TetraMod/log/train_log/rna004_m6a_mix_tetra.log 2>&1