#!/usr/bin/env bash
set -Eeuo pipefail

# Build a maximized mAFiA Stage 1 control dataset covering all 6 DRACH motifs.
#
# Train merge includes:
#   WUE_splint_lig* + WUE_splint_batch2* + RL_Mix1-4
#
# Final heldout is prepared as per-run datasets only and is NOT merged into train:
#   Mix_1_A_RTA, Mix_2_m6A_RTA, Mix_3_A_RTA, Mix_4_m6A_RTA
#
# Edit the path block below for the remote server before running.

REPO="${REPO:-/path/to/TetraMod}"
MAFIA_ROOT="${MAFIA_ROOT:-/path/to/mAFia_RNA002_PRJEB74106/HEK293}"
WORK_ROOT="${WORK_ROOT:-/path/to/tetramod_mafia_rna002}"
DORADO_BIN="${DORADO_BIN:-/path/to/dorado}"
DORADO_MODEL="${DORADO_MODEL:-/path/to/dorado/rna002_70bps_sup@v3}"

DEVICE="${DEVICE:-cuda:0}"
POD5_JOBS="${POD5_JOBS:-4}"
DATASET_WORKERS="${DATASET_WORKERS:-8}"
CHUNK_LEN="${CHUNK_LEN:-5000}"
OVERLAP="${OVERLAP:-500}"
VALID_FRACTION="${VALID_FRACTION:-0.1}"
SEED="${SEED:-114514}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BUILD_HELDOUT="${BUILD_HELDOUT:-1}"
RUN_DATASET_CHECK="${RUN_DATASET_CHECK:-1}"

TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-stage1_train_mafia_6motif_wue_batch2_rlmix1_4}"
HELDOUT_ROOT_NAME="${HELDOUT_ROOT_NAME:-final_heldout_mix_1_4}"

OLIGO_MANIFEST="$REPO/gen_data/mafia_oligos.tsv"
RUN_MANIFEST="$REPO/gen_data/mafia_runs.tsv"
TRAIN_DATASET_DIR="$WORK_ROOT/chunks/$TRAIN_DATASET_NAME"
HELDOUT_DATASET_ROOT="$WORK_ROOT/chunks/$HELDOUT_ROOT_NAME"
CHECK_REPORT_DIR="$REPO/dataset_check_res/$TRAIN_DATASET_NAME/check_reports"

TRAIN_RUNS=(
  WUE_splint_lig_A_RTA
  WUE_splint_lig_m6A_RTA
  WUE_splint_batch2_A_RTA
  WUE_splint_batch2_m6A_RTA
  WUE_splint_batch2_m6A_RTA_1
  WUE_splint_batch2_m6A_RTA_2
  RL_Mix1_A_RTA
  RL_Mix2_A_RTA
  RL_Mix3_m6A_RTA
  RL_Mix4_m6A_RTA
)

HELDOUT_RUNS=(
  Mix_1_A_RTA
  Mix_2_m6A_RTA
  Mix_3_A_RTA
  Mix_4_m6A_RTA
)

fail_if_placeholder() {
  local name="$1"
  local value="$2"
  if [[ "$value" == /path/to/* || -z "$value" ]]; then
    echo "[error] Please set $name near the top of this script. Current value: $value" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[error] Missing required path: $path" >&2
    exit 1
  fi
}

run_dir_from_manifest() {
  local run_id="$1"
  local run_dir
  run_dir="$(awk -F'\t' -v id="$run_id" 'NR > 1 && $1 == id {print $3}' "$RUN_MANIFEST")"
  if [[ -z "$run_dir" ]]; then
    echo "[error] run_id not found in $RUN_MANIFEST: $run_id" >&2
    exit 1
  fi
  printf '%s\n' "$run_dir"
}

has_any_file() {
  local directory="$1"
  [[ -d "$directory" && -n "$(find "$directory" -type f -print -quit)" ]]
}

convert_run_to_pod5() {
  local run_id="$1"
  local output_dir="$WORK_ROOT/pod5/$run_id"
  local source_dir
  source_dir="$(run_dir_from_manifest "$run_id")"

  if [[ "$SKIP_EXISTING" == "1" ]] && has_any_file "$output_dir"; then
    echo "[skip pod5] $run_id -> $output_dir"
    return
  fi

  echo "[pod5] $run_id"
  python gen_data/convert_fast5_tar_to_pod5.py \
    "$MAFIA_ROOT/$source_dir" \
    --output-dir "$output_dir" \
    --recursive \
    --jobs "$POD5_JOBS"
}

basecall_and_sort_run() {
  local run_id="$1"
  local pod5_dir="$WORK_ROOT/pod5/$run_id"
  local bam="$WORK_ROOT/bam/$run_id.bam"
  local sorted_bam="$WORK_ROOT/bam/$run_id.sorted.bam"

  require_file "$pod5_dir"

  if [[ "$SKIP_EXISTING" == "1" && -s "$sorted_bam" && -e "$sorted_bam.bai" ]]; then
    echo "[skip bam] $run_id -> $sorted_bam"
    return
  fi

  if [[ "$SKIP_EXISTING" != "1" || ! -s "$bam" ]]; then
    echo "[dorado] $run_id"
    "$DORADO_BIN" basecaller "$DORADO_MODEL" "$pod5_dir" \
      --emit-moves \
      --device "$DEVICE" \
      > "$bam"
  else
    echo "[reuse bam] $run_id -> $bam"
  fi

  echo "[samtools sort/index] $run_id"
  samtools sort -o "$sorted_bam" "$bam"
  samtools index "$sorted_bam"
}

build_per_run_dataset() {
  local run_id="$1"
  local output_dir="$2"
  local sorted_bam="$WORK_ROOT/bam/$run_id.sorted.bam"
  local pod5_dir="$WORK_ROOT/pod5/$run_id"

  require_file "$sorted_bam"
  require_file "$sorted_bam.bai"
  require_file "$pod5_dir"

  if [[ "$SKIP_EXISTING" == "1" \
      && -s "$output_dir/chunks.npy" \
      && -s "$output_dir/references.npy" \
      && -s "$output_dir/reference_lengths.npy" \
      && -s "$output_dir/mod_targets.npy" \
      && -s "$output_dir/metadata.npz" ]]; then
    echo "[skip dataset] $run_id -> $output_dir"
    return
  fi

  echo "[dataset] $run_id -> $output_dir"
  python gen_data/create_mafia_synthetic_stage1_dataset.py \
    --bam-file "$sorted_bam" \
    --pod5-dir "$pod5_dir" \
    --output-dir "$output_dir" \
    --oligo-manifest "$OLIGO_MANIFEST" \
    --run-manifest "$RUN_MANIFEST" \
    --run-id "$run_id" \
    --sample-type rna \
    --rna002 \
    --chunk-len "$CHUNK_LEN" \
    --overlap "$OVERLAP" \
    --workers "$DATASET_WORKERS"
}

merge_train_dataset() {
  local merge_args=()
  local run_id

  for run_id in "${TRAIN_RUNS[@]}"; do
    merge_args+=(--dataset "$run_id:$WORK_ROOT/chunks/per_run/$run_id")
  done

  echo "[merge train] ${TRAIN_DATASET_DIR}"
  python gen_data/merge_mafia_stage1_datasets.py \
    "${merge_args[@]}" \
    --output-dir "$TRAIN_DATASET_DIR" \
    --valid-fraction "$VALID_FRACTION" \
    --seed "$SEED"
}

write_run_lists() {
  mkdir -p "$TRAIN_DATASET_DIR"
  printf '%s\n' "${TRAIN_RUNS[@]}" > "$TRAIN_DATASET_DIR/train_run_ids.txt"
  printf '%s\n' "${HELDOUT_RUNS[@]}" > "$TRAIN_DATASET_DIR/final_heldout_run_ids.txt"
}

run_dataset_check() {
  if [[ "$RUN_DATASET_CHECK" != "1" ]]; then
    return
  fi

  echo "[dataset check] ${CHECK_REPORT_DIR}"
  python dataset_check/check_mafia_stage1_dataset.py \
    "$TRAIN_DATASET_DIR" \
    --output-dir "$CHECK_REPORT_DIR"
}

main() {
  fail_if_placeholder REPO "$REPO"
  fail_if_placeholder MAFIA_ROOT "$MAFIA_ROOT"
  fail_if_placeholder WORK_ROOT "$WORK_ROOT"
  fail_if_placeholder DORADO_BIN "$DORADO_BIN"
  fail_if_placeholder DORADO_MODEL "$DORADO_MODEL"
  require_file "$REPO"
  require_file "$MAFIA_ROOT"
  require_file "$DORADO_BIN"

  mkdir -p \
    "$WORK_ROOT/manifests" \
    "$WORK_ROOT/pod5" \
    "$WORK_ROOT/bam" \
    "$WORK_ROOT/chunks/per_run" \
    "$WORK_ROOT/chunks" \
    "$WORK_ROOT/models"

  cd "$REPO"
  require_file "$OLIGO_MANIFEST"
  require_file "$RUN_MANIFEST"

  echo "[config]"
  echo "  REPO=$REPO"
  echo "  MAFIA_ROOT=$MAFIA_ROOT"
  echo "  WORK_ROOT=$WORK_ROOT"
  echo "  DORADO_BIN=$DORADO_BIN"
  echo "  DORADO_MODEL=$DORADO_MODEL"
  echo "  DEVICE=$DEVICE"
  echo "  CHUNK_LEN=$CHUNK_LEN OVERLAP=$OVERLAP"
  echo "  TRAIN_DATASET_DIR=$TRAIN_DATASET_DIR"
  echo "  HELDOUT_DATASET_ROOT=$HELDOUT_DATASET_ROOT"

  python gen_data/create_mafia_synthetic_stage1_dataset.py \
    --write-template-manifest "$WORK_ROOT/manifests"

  local all_runs=("${TRAIN_RUNS[@]}")
  if [[ "$BUILD_HELDOUT" == "1" ]]; then
    all_runs+=("${HELDOUT_RUNS[@]}")
    mkdir -p "$HELDOUT_DATASET_ROOT"
  fi

  local run_id
  for run_id in "${all_runs[@]}"; do
    convert_run_to_pod5 "$run_id"
  done

  for run_id in "${all_runs[@]}"; do
    basecall_and_sort_run "$run_id"
  done

  for run_id in "${TRAIN_RUNS[@]}"; do
    build_per_run_dataset "$run_id" "$WORK_ROOT/chunks/per_run/$run_id"
  done

  if [[ "$BUILD_HELDOUT" == "1" ]]; then
    for run_id in "${HELDOUT_RUNS[@]}"; do
      build_per_run_dataset "$run_id" "$HELDOUT_DATASET_ROOT/$run_id"
    done
  fi

  merge_train_dataset
  write_run_lists
  run_dataset_check

  echo "[done] 6 motif train dataset: $TRAIN_DATASET_DIR"
  if [[ "$BUILD_HELDOUT" == "1" ]]; then
    echo "[done] final heldout per-run datasets: $HELDOUT_DATASET_ROOT"
  fi
  echo
  echo "Suggested training directory:"
  echo "  $WORK_ROOT/models/${TRAIN_DATASET_NAME}"
  echo
  echo "Use this dataset for train_promote:"
  echo "  --directory \"$TRAIN_DATASET_DIR\""
}

main "$@"
