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