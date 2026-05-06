FAST5_INPUT="${FAST5_INPUT:?set FAST5_INPUT to a FAST5 directory, FAST5 file, or fast5.tar.gz shard}"
POD5_DIR="${POD5_DIR:?set POD5_DIR for converted POD5 output}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the trained Stage2 LLP TetraMod model directory}"
REF_FASTA="${REF_FASTA:?set REF_FASTA to the HeLa/human genome FASTA used for alignment}"
OUT_ROOT="${OUT_ROOT:-hela_rna002_tetramod_out}"

DEVICE="${DEVICE:-cuda:0}"
CONVERT_JOBS="${CONVERT_JOBS:-1}"
ALIGN_THREADS="${ALIGN_THREADS:-8}"
SORT_THREADS="${SORT_THREADS:-8}"
WEIGHTS="${WEIGHTS:-0}"              # 0 means latest weights_*.tar in MODEL_DIR.
MAX_READS="${MAX_READS:-0}"          # 0 means all reads; set e.g. 1000 for a smoke run.
MOD_THRESHOLD="${MOD_THRESHOLD:-0.5}"
MM2_PRESET="${MM2_PRESET:-splice}"   # Use lr:hq/map-ont instead if aligning to transcriptome/constructs.
MIN_COVERAGE="${MIN_COVERAGE:-5}"

RAW_BAM="$OUT_ROOT/hela_rna002_tetramod.raw.bam"
SORTED_BAM="$OUT_ROOT/hela_rna002_tetramod.sorted.bam"

mkdir -p "$POD5_DIR" "$OUT_ROOT" "$OUT_ROOT/logs"

# 1. Convert RNA002 FAST5 data to POD5.
FAST5_INPUT="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/fast5_pass/"
POD5_DIR="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/converted_pod5/"
python gen_data/convert_fast5_tar_to_pod5.py \
    "$FAST5_INPUT" \
    --output-dir "$POD5_DIR" \
    --recursive \
    --jobs 8 \
    > "/home/lijy/workspace/TetraMod/log/gen_data/convert_fast5_to_pod5_20260506.log" 2>&1

# 2. Run the trained promoted model on HeLa POD5 and emit aligned modBAM.
MODEL_DIR="/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage2_llp_run2/"
REF_FASTA="/data/biolab-nvme-pcie2/lijy/HG002/hg38.fa"
POD5_SINGLE="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/pod5_test_single/"
RAW_BAM="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/tetramod_bam/pod5_test_single.bam"
tetramod basecaller \
    "$MODEL_DIR" \
    "$POD5_SINGLE" \
    --device cuda:1 \
    --weights 10 \
    --recursive \
    --rna \
    --reference "$REF_FASTA" \
    --alignment-threads 4 \
    --mod-threshold 0.0 \
    > "$RAW_BAM"

bonito basecaller \
    /data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3 \
    --rna \
    --reference "$REF_FASTA" \
    "$POD5_SINGLE" \
    > "/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/bonito_bam/pod5_test_single.bam"


# 3. compare tetramod vs bonito mod calls.
VAL_OUT_DIR="/home/lijy/workspace/TetraMod/val_res/hela_rna002_tetramod_single/"
TET_BAM="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/tetramod_bam/pod5_test_single.sorted.bam"
BON_BAM="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/bonito_bam/pod5_test_single.sorted.bam"
python validate/compare_basecaller_bams.py \
    --tetramod-bam "$TET_BAM" \
    --bonito-bam "$BON_BAM" \
    --output-dir "$VAL_OUT_DIR/basecall_compare"


# 4. check mod tag is being emitted in the tetramod BAM.
samtools view "$TET_BAM" | \
    awk 'BEGIN{n=0;m=0} {n++; if ($0 ~ /\tMM:Z:/ && $0 ~ /\tML:B:C/) m++} END{print "records",n,"with_MM_ML",m}'


# 5.1 Gold-site test 先跑坐标约定检查
GOLD_TSV="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/m6A_HeLa.tsv"

python validate/check_gold_coordinate_conventions.py \
    --bam "$TET_BAM" \
    --gold-bed "$GOLD_TSV" \
    --gold-format m6aatlas \
    --reference "$REF_FASTA" \
    --output-dir "$VAL_OUT_DIR/m6a_gold_convention_check" \
    --mod-code a \
    --canonical-base A \
    --min-coverage 5 \
    --prob-threshold 0.5 \
    --score-column mean_prob_zero_filled \
    --motif ""


# 5.2 Gold-site test 评估模型性能
python validate/evaluate_modbam_gold_sites.py \
    --bam "$TET_BAM" \
    --gold-bed "$GOLD_TSV" \
    --gold-format m6aatlas \
    --reference "$REF_FASTA" \
    --output-dir "$VAL_OUT_DIR/m6a_gold_evaluation" \
    --mod-code a \
    --canonical-base A \
    --min-coverage 1 \
    --prob-threshold 0.5 \
    --score-column mean_prob_zero_filled \
    --motif ""

# 6.1 ReRun the trained promoted model on HeLa POD5 and emit aligned modBAM.
MODEL_DIR="/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage2_llp_run2/"
REF_FASTA="/data/biolab-nvme-pcie2/lijy/HG002/hg38.fa"
POD5_SINGLE="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/pod5_test_15/"
RAW_BAM="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/tetramod_bam/pod5_test_15.bam"
tetramod basecaller \
    "$MODEL_DIR" \
    "$POD5_SINGLE" \
    --device cuda:1 \
    --weights 10 \
    --recursive \
    --rna \
    --reference "$REF_FASTA" \
    --alignment-threads 4 \
    --mod-threshold 0.0 \
    > "$RAW_BAM"


bonito basecaller \
    /data/biolab-nvme-pcie2/lijy/bonito_models/rna002_70bps_sup@v3 \
    --rna \
    --reference "$REF_FASTA" \
    "$POD5_SINGLE" \
    > "/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/bonito_bam/pod5_test_15.bam"


python validate/compare_basecaller_bams.py \
    --tetramod-bam /data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/tetramod_bam/pod5_test_15.sorted.bam \
    --bonito-bam /data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/bonito_bam/pod5_test_15.sorted.bam \
    --output-dir val_res/hela_rna002_tetramod_15/basecall_compare


GOLD_TSV="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/m6A_HeLa.tsv"
TETRA_BAM="/data/biolab-nvme-pcie2/lijy/PRJNA1108269/HeLa_mRNA_Direct_rep.1/tetramod_bam/pod5_test_15.sorted.bam"
REF_FASTA="/data/biolab-nvme-pcie2/lijy/HG002/hg38.fa"

python validate/evaluate_modbam_gold_sites.py \
    --bam "$TETRA_BAM" \
    --gold-bed "$GOLD_TSV" \
    --gold-format auto \
    --reference "$REF_FASTA" \
    --output-dir val_res/hela_rna002_tetramod_15/m6a_gold_eval_mincov5 \
    --canonical-base A \
    --mod-code a \
    --min-coverage 5 \
    --score-column mean_prob_zero_filled \
    --prob-threshold 0.5



# IVT baseline
MODEL_DIR="/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/tetramod_model/stage2_llp_run2/"
IVT_POD5="/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/converted_pod5/cc0"
IVT_REF="/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/GSE124309_FASTA_sequences_of_Curlcakes_4pole.fasta"
tetramod basecaller "$MODEL_DIR" "$IVT_POD5" \
    --device cuda:0 \
    --weights 10 \
    --recursive \
    --rna \
    --max-reads 4000 \
    --reference "$IVT_REF" \
    --alignment-threads 4 \
    --mod-threshold 0.0 \
    > ivt_unmod_tetramod.bam
