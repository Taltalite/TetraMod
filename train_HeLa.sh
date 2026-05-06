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
    > "$OUT_ROOT/logs/convert_fast5_to_pod5.log" 2>&1

# 2. Run the trained promoted model on HeLa POD5 and emit aligned modBAM.
tetramod basecaller \
    "$MODEL_DIR" \
    "$POD5_DIR" \
    --device "$DEVICE" \
    --weights "$WEIGHTS" \
    --recursive \
    --rna \
    --reference "$REF_FASTA" \
    --mm2-preset "$MM2_PRESET" \
    --alignment-threads "$ALIGN_THREADS" \
    --mod-threshold "$MOD_THRESHOLD" \
    --max-reads "$MAX_READS" \
    > "$RAW_BAM" \
    2> "$OUT_ROOT/logs/tetramod_basecaller.log"

# 3. Sort and index the modBAM.
samtools sort -@ "$SORT_THREADS" -o "$SORTED_BAM" "$RAW_BAM"
samtools index "$SORTED_BAM"

# 4. Optional gold-site evaluation. Set GOLD_BED only if you have reliable
# HeLa-compatible m6A site labels in the same reference coordinate system.
if [[ -n "${GOLD_BED:-}" ]]; then
    python validate/check_gold_coordinate_conventions.py \
        --bam "$SORTED_BAM" \
        --gold-bed "$GOLD_BED" \
        --reference "$REF_FASTA" \
        --output-dir "$OUT_ROOT/gold_convention_check" \
        --mod-code a \
        --canonical-base A \
        --min-coverage "$MIN_COVERAGE"

    python validate/evaluate_modbam_gold_sites.py \
        --bam "$SORTED_BAM" \
        --gold-bed "$GOLD_BED" \
        --reference "$REF_FASTA" \
        --output-dir "$OUT_ROOT/gold_eval" \
        --mod-code a \
        --canonical-base A \
        --min-coverage "$MIN_COVERAGE" \
        --prob-threshold "$MOD_THRESHOLD"
fi

echo "TetraMod HeLa RNA002 modBAM: $SORTED_BAM"
