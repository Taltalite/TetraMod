#!/usr/bin/env bash
set -Eeuo pipefail

# Build a MoDiDeC RNA002 m6A Stage 1 control dataset from already-converted POD5.
#
# This is intentionally simpler than train_mix_stage1_6motif_dataset.sh:
#   - input is direct POD5 path(s), not ENA/FAST5/TAR names;
#   - no external run manifest is required;
#   - the script writes a small internal MoDiDeC m6A oligo table and per-run
#     run manifest under $WORK_ROOT/manifests;
#   - center m6A oligo sites are labeled, all other positions are ignored.
#
# Required inputs:
#   MODIDEC_CANONICAL_POD5_SPECS="unmod:/path/to/unmodified_pod5"
#   MODIDEC_M6A_POD5_SPECS="m6a:/path/to/m6A_pod5"
#
# Multiple inputs are separated by semicolons:
#   MODIDEC_CANONICAL_POD5_SPECS="unmod1:/pod5/a;unmod2:/pod5/b"
#   MODIDEC_M6A_POD5_SPECS="m6a1:/pod5/c;m6a2:/pod5/d"
#
# Optional heldout per-run datasets:
#   MODIDEC_HELDOUT_CANONICAL_POD5_SPECS="heldout_unmod:/path/to/pod5"
#   MODIDEC_HELDOUT_M6A_POD5_SPECS="heldout_m6a:/path/to/pod5"


REPO="${REPO:-/home/lijy/workspace/TetraMod/}"
WORK_ROOT="${WORK_ROOT:-/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002}"
DORADO_BIN="${DORADO_BIN:-/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado}"
DORADO_MODEL="${DORADO_MODEL:-/home/lijy/workspace/TetraMod/src/tetramod/models/rna002_70bps_sup@v3/rna002_70bps_sup@v3}"

STAGE="${STAGE:-all}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_WORKERS="${DATASET_WORKERS:-8}"
CHUNK_LEN="${CHUNK_LEN:-5000}"
OVERLAP="${OVERLAP:-500}"
VALID_FRACTION="${VALID_FRACTION:-0.1}"
SEED="${SEED:-114514}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_DATASET_CHECK="${RUN_DATASET_CHECK:-1}"
BUILD_HELDOUT="${BUILD_HELDOUT:-1}"
MAX_RECORDS="${MAX_RECORDS:--1}"
MAX_CHUNKS="${MAX_CHUNKS:--1}"
MIN_OLIGO_IDENTITY="${MIN_OLIGO_IDENTITY:-0.86}"
MAX_OLIGO_MISMATCHES="${MAX_OLIGO_MISMATCHES:-4}"

TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-stage1_train_modidec_m6a_rna002}"
HELDOUT_ROOT_NAME="${HELDOUT_ROOT_NAME:-heldout_modidec_m6a_rna002}"

MODIDEC_CANONICAL_POD5_SPECS="${MODIDEC_CANONICAL_POD5_SPECS:-}"
MODIDEC_M6A_POD5_SPECS="${MODIDEC_M6A_POD5_SPECS:-}"
MODIDEC_HELDOUT_CANONICAL_POD5_SPECS="${MODIDEC_HELDOUT_CANONICAL_POD5_SPECS:-}"
MODIDEC_HELDOUT_M6A_POD5_SPECS="${MODIDEC_HELDOUT_M6A_POD5_SPECS:-}"
MODIDEC_TRAIN_OLIGO_IDS="${MODIDEC_TRAIN_OLIGO_IDS:-}"
MODIDEC_HELDOUT_OLIGO_IDS="${MODIDEC_HELDOUT_OLIGO_IDS:-}"

MANIFEST_DIR="$WORK_ROOT/manifests"
OLIGO_MANIFEST="$MANIFEST_DIR/modidec_m6a_oligos.tsv"
TRAIN_DATASET_DIR="$WORK_ROOT/chunks/$TRAIN_DATASET_NAME"
HELDOUT_DATASET_ROOT="$WORK_ROOT/chunks/$HELDOUT_ROOT_NAME"
CHECK_REPORT_DIR="$REPO/dataset_check_res/$TRAIN_DATASET_NAME/check_reports"

fail() {
  echo "[error] $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -e "$path" ]] || fail "Missing required path: $path"
}

require_executable() {
  local command_or_path="$1"
  if [[ "$command_or_path" == */* ]]; then
    [[ -x "$command_or_path" ]] || fail "Missing executable: $command_or_path"
    return
  fi
  command -v "$command_or_path" >/dev/null 2>&1 || fail "Executable not found in PATH: $command_or_path"
}

has_any_file() {
  local directory="$1"
  [[ -d "$directory" && -n "$(find "$directory" -type f -print -quit)" ]]
}

sanitize_run_id() {
  local value="$1"
  value="${value//[^A-Za-z0-9_.-]/_}"
  [[ -n "$value" ]] || fail "Empty run id after sanitising input spec."
  printf '%s\n' "$value"
}

parse_specs() {
  local specs="$1"
  local status="$2"
  local split="$3"
  local raw name path

  [[ -n "$specs" ]] || return 0
  IFS=';' read -r -a raw_specs <<< "$specs"
  for raw in "${raw_specs[@]}"; do
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    [[ -n "$raw" ]] || continue
    if [[ "$raw" == *:* ]]; then
      name="${raw%%:*}"
      path="${raw#*:}"
    else
      path="$raw"
      name="$(basename "$path")"
    fi
    name="$(sanitize_run_id "$name")"
    printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$split" "$path"
  done
}

write_modidec_oligo_manifest() {
  mkdir -p "$MANIFEST_DIR"
  cat > "$OLIGO_MANIFEST" <<'EOF'
oligo_id	sequence	center_index	ligation_strategy	role
modidec_m6A_01	GAUACGGGAGACAGCCACCGGAAUACGGGAGm6ACAGCCACCUC	31	splint_ligation	train
modidec_m6A_02	GAGUGCCAGGACCGACCAUGGAAGUGCCAGGm6ACCGACCAUUC	31	splint_ligation	train
modidec_m6A_03	GACACCAGUGACUCCCAUAGGAACACCAGUGm6ACUCCCAUAUC	31	splint_ligation	train
modidec_m6A_04	GAUAAACGAGACCGUCUAGGGAAUAAACGAGm6ACCGUCUAGUC	31	splint_ligation	train
modidec_m6A_05	GAGUUAAGAGACUGAAUCUGGAAGUUAAGAGm6ACUGAAUCUUC	31	splint_ligation	train
modidec_m6A_06	GAAGUACAAAACAAUCAUUGGAAAGUACAAAm6ACAAUCAUUUC	31	splint_ligation	train
modidec_m6A_07	GAUUCAGAGAACCACUUGAGGAAUUCAGAGAm6ACCACUUGAUC	31	splint_ligation	train
modidec_m6A_08	GAUAAACUUAACUCCAAAAGGAAUAAACUUAm6ACUCCAAAAUC	31	splint_ligation	train
modidec_m6A_09	GAUCACUCGAACUUCAAGCGGAAUCACUCGAm6ACUUCAAGCUC	31	splint_ligation	train
modidec_m6A_10	GAUUGUGGUAACGUCCCCAGGAAUUGUGGUAm6ACGUCCCCAUC	31	splint_ligation	train
modidec_m6A_11	AACGCCUGGCm6AGCCGGAAGCC	10	splint_ligation	train
EOF
}

all_oligo_ids_csv() {
  awk -F'\t' 'NR > 1 && $1 !~ /^#/ { ids = ids ? ids "," $1 : $1 } END { print ids }' "$OLIGO_MANIFEST"
}

oligo_ids_for_split() {
  local split="$1"
  if [[ "$split" == "heldout" && -n "$MODIDEC_HELDOUT_OLIGO_IDS" ]]; then
    printf '%s\n' "$MODIDEC_HELDOUT_OLIGO_IDS"
    return
  fi
  if [[ "$split" == "train" && -n "$MODIDEC_TRAIN_OLIGO_IDS" ]]; then
    printf '%s\n' "$MODIDEC_TRAIN_OLIGO_IDS"
    return
  fi
  all_oligo_ids_csv
}

write_run_manifest() {
  local run_id="$1"
  local status="$2"
  local output="$3"
  local split="${4:-train}"
  local oligo_ids
  oligo_ids="$(oligo_ids_for_split "$split")"
  [[ -n "$oligo_ids" ]] || fail "No oligo ids found in $OLIGO_MANIFEST"
  cat > "$output" <<EOF
run_id	accession	local_name	modification_status	ligation_strategy	split	oligo_ids	modified_oligo_ids
$run_id	modidec	direct_pod5	$status	splint_ligation	train	$oligo_ids	
EOF
}

spec_file() {
  local path="$WORK_ROOT/manifests/pod5_specs.tsv"
  {
    parse_specs "$MODIDEC_CANONICAL_POD5_SPECS" "unmodified" "train"
    parse_specs "$MODIDEC_M6A_POD5_SPECS" "modified" "train"
    if [[ "$BUILD_HELDOUT" == "1" ]]; then
      parse_specs "$MODIDEC_HELDOUT_CANONICAL_POD5_SPECS" "unmodified" "heldout"
      parse_specs "$MODIDEC_HELDOUT_M6A_POD5_SPECS" "modified" "heldout"
    fi
  } > "$path"
  printf '%s\n' "$path"
}

train_run_ids() {
  awk -F'\t' '$3 == "train" { print $1 }' "$1"
}

heldout_run_ids() {
  awk -F'\t' '$3 == "heldout" { print $1 }' "$1"
}

pod5_path_for_run() {
  local specs_file="$1"
  local run_id="$2"
  awk -F'\t' -v id="$run_id" '$1 == id { print $4 }' "$specs_file"
}

pod5_input_dir_for_run() {
  local specs_file="$1"
  local run_id="$2"
  local path link_dir link_path
  path="$(pod5_path_for_run "$specs_file" "$run_id")"
  require_file "$path"
  if [[ -d "$path" ]]; then
    printf '%s\n' "$path"
    return
  fi
  case "${path,,}" in
    *.pod5) ;;
    *) fail "POD5 input for $run_id is a file but does not end with .pod5: $path" ;;
  esac
  link_dir="$WORK_ROOT/pod5_input_dirs/$run_id"
  mkdir -p "$link_dir"
  link_path="$link_dir/$(basename "$path")"
  if [[ -L "$link_path" ]]; then
    if [[ "$(readlink "$link_path")" != "$path" ]]; then
      ln -sfn "$path" "$link_path"
    fi
  elif [[ -e "$link_path" ]]; then
    fail "Cannot create POD5 symlink because a non-symlink path already exists: $link_path"
  else
    ln -s "$path" "$link_path"
  fi
  printf '%s\n' "$link_dir"
}

status_for_run() {
  local specs_file="$1"
  local run_id="$2"
  awk -F'\t' -v id="$run_id" '$1 == id { print $2 }' "$specs_file"
}

split_for_run() {
  local specs_file="$1"
  local run_id="$2"
  awk -F'\t' -v id="$run_id" '$1 == id { print $3 }' "$specs_file"
}

basecall_run() {
  local specs_file="$1"
  local run_id="$2"
  local pod5_dir bam
  pod5_dir="$(pod5_input_dir_for_run "$specs_file" "$run_id")"
  bam="$WORK_ROOT/bam/$run_id.bam"

  require_file "$pod5_dir"
  if [[ "$SKIP_EXISTING" == "1" && -s "$bam" ]]; then
    echo "[skip bam] $run_id -> $bam"
    return
  fi

  echo "[dorado] $run_id"
  "$DORADO_BIN" basecaller "$DORADO_MODEL" "$pod5_dir" \
    --emit-moves \
    --device "$DEVICE" \
    > "$bam"
}

build_run_dataset() {
  local specs_file="$1"
  local run_id="$2"
  local output_dir="$3"
  local pod5_dir bam status split run_manifest

  pod5_dir="$(pod5_input_dir_for_run "$specs_file" "$run_id")"
  status="$(status_for_run "$specs_file" "$run_id")"
  split="$(split_for_run "$specs_file" "$run_id")"
  bam="$WORK_ROOT/bam/$run_id.bam"
  run_manifest="$MANIFEST_DIR/${run_id}.runs.tsv"

  require_file "$pod5_dir"
  require_file "$bam"

  if [[ "$SKIP_EXISTING" == "1" \
      && -s "$output_dir/chunks.npy" \
      && -s "$output_dir/references.npy" \
      && -s "$output_dir/reference_lengths.npy" \
      && -s "$output_dir/mod_targets.npy" \
      && -s "$output_dir/metadata.npz" ]]; then
    echo "[skip dataset] $run_id -> $output_dir"
    return
  fi

  write_run_manifest "$run_id" "$status" "$run_manifest" "$split"

  echo "[dataset] $run_id -> $output_dir"
  python gen_data/create_mafia_synthetic_stage1_dataset.py \
    --bam-file "$bam" \
    --pod5-dir "$pod5_dir" \
    --output-dir "$output_dir" \
    --oligo-manifest "$OLIGO_MANIFEST" \
    --run-manifest "$run_manifest" \
    --run-id "$run_id" \
    --sample-type rna \
    --rna002 \
    --chunk-len "$CHUNK_LEN" \
    --overlap "$OVERLAP" \
    --workers "$DATASET_WORKERS" \
    --max-records "$MAX_RECORDS" \
    --max-chunks "$MAX_CHUNKS" \
    --min-oligo-identity "$MIN_OLIGO_IDENTITY" \
    --max-oligo-mismatches "$MAX_OLIGO_MISMATCHES"
}

merge_train_dataset() {
  local specs_file="$1"
  local merge_args=()
  local run_id

  while IFS= read -r run_id; do
    merge_args+=(--dataset "$run_id:$WORK_ROOT/chunks/per_run/$run_id")
  done < <(train_run_ids "$specs_file")

  [[ "${#merge_args[@]}" -gt 0 ]] || fail "No train POD5 specs were provided."

  echo "[merge train] $TRAIN_DATASET_DIR"
  python gen_data/merge_mafia_stage1_datasets.py \
    "${merge_args[@]}" \
    --output-dir "$TRAIN_DATASET_DIR" \
    --valid-fraction "$VALID_FRACTION" \
    --seed "$SEED"
}

write_run_lists() {
  local specs_file="$1"
  mkdir -p "$TRAIN_DATASET_DIR"
  train_run_ids "$specs_file" > "$TRAIN_DATASET_DIR/train_run_ids.txt"
  if [[ "$BUILD_HELDOUT" == "1" ]]; then
    mkdir -p "$HELDOUT_DATASET_ROOT"
    heldout_run_ids "$specs_file" > "$HELDOUT_DATASET_ROOT/heldout_run_ids.txt"
  fi
}

run_dataset_check() {
  if [[ "$RUN_DATASET_CHECK" != "1" ]]; then
    return
  fi

  echo "[dataset check] $CHECK_REPORT_DIR"
  python dataset_check/check_mafia_stage1_dataset.py \
    "$TRAIN_DATASET_DIR" \
    --output-dir "$CHECK_REPORT_DIR"
}

validate_stage() {
  case "$STAGE" in
    downstream|all) ;;
    *) fail "STAGE must be one of: downstream, all. Current: $STAGE" ;;
  esac
}

prepare_common_dirs() {
  mkdir -p \
    "$MANIFEST_DIR" \
    "$WORK_ROOT/pod5_input_dirs" \
    "$WORK_ROOT/bam" \
    "$WORK_ROOT/chunks/per_run" \
    "$WORK_ROOT/chunks" \
    "$WORK_ROOT/models"
}

prepare_repo_context() {
  require_file "$REPO"
  cd "$REPO"
  write_modidec_oligo_manifest
}

validate_inputs() {
  local specs_file="$1"
  require_executable "$DORADO_BIN"
  require_file "$DORADO_MODEL"
  [[ -s "$specs_file" ]] || fail "No POD5 specs were provided. Set MODIDEC_CANONICAL_POD5_SPECS and MODIDEC_M6A_POD5_SPECS."
  [[ -n "$MODIDEC_CANONICAL_POD5_SPECS" ]] || fail "Set MODIDEC_CANONICAL_POD5_SPECS for unmodified/control POD5."
  [[ -n "$MODIDEC_M6A_POD5_SPECS" ]] || fail "Set MODIDEC_M6A_POD5_SPECS for m6A POD5."
  while IFS=$'\t' read -r run_id _split _status path; do
    [[ -n "$run_id" ]] || continue
    require_file "$path"
    if [[ -d "$path" ]]; then
      has_any_file "$path" || fail "POD5 path has no files: $path"
    elif [[ "${path,,}" != *.pod5 ]]; then
      fail "POD5 input must be a directory or .pod5 file: $path"
    fi
  done < "$specs_file"
}

print_config() {
  echo "[config]"
  echo "  STAGE=$STAGE"
  echo "  REPO=$REPO"
  echo "  WORK_ROOT=$WORK_ROOT"
  echo "  DORADO_BIN=$DORADO_BIN"
  echo "  DORADO_MODEL=$DORADO_MODEL"
  echo "  DEVICE=$DEVICE"
  echo "  CHUNK_LEN=$CHUNK_LEN OVERLAP=$OVERLAP"
  echo "  TRAIN_DATASET_DIR=$TRAIN_DATASET_DIR"
  echo "  HELDOUT_DATASET_ROOT=$HELDOUT_DATASET_ROOT"
  echo "  OLIGO_MANIFEST=$OLIGO_MANIFEST"
  echo "  MODIDEC_TRAIN_OLIGO_IDS=${MODIDEC_TRAIN_OLIGO_IDS:-<all>}"
  echo "  MODIDEC_HELDOUT_OLIGO_IDS=${MODIDEC_HELDOUT_OLIGO_IDS:-<all>}"
}

run_downstream_stage() {
  local specs_file="$1"
  local run_id

  while IFS= read -r run_id; do
    basecall_run "$specs_file" "$run_id"
  done < <(awk -F'\t' '{ print $1 }' "$specs_file")

  while IFS= read -r run_id; do
    build_run_dataset "$specs_file" "$run_id" "$WORK_ROOT/chunks/per_run/$run_id"
  done < <(train_run_ids "$specs_file")

  if [[ "$BUILD_HELDOUT" == "1" ]]; then
    while IFS= read -r run_id; do
      build_run_dataset "$specs_file" "$run_id" "$HELDOUT_DATASET_ROOT/$run_id"
    done < <(heldout_run_ids "$specs_file")
  fi

  merge_train_dataset "$specs_file"
  write_run_lists "$specs_file"
  run_dataset_check

  echo "[done] MoDiDeC m6A Stage 1 train dataset: $TRAIN_DATASET_DIR"
  if [[ "$BUILD_HELDOUT" == "1" ]]; then
    echo "[done] MoDiDeC heldout per-run datasets: $HELDOUT_DATASET_ROOT"
  fi
  echo
  echo "Use this dataset for train_promote:"
  echo "  --directory \"$TRAIN_DATASET_DIR\""
}

main() {
  validate_stage
  prepare_common_dirs
  prepare_repo_context
  local specs_file
  specs_file="$(spec_file)"
  validate_inputs "$specs_file"
  print_config
  if [[ "$STAGE" == "downstream" || "$STAGE" == "all" ]]; then
    run_downstream_stage "$specs_file"
  fi
}

main "$@"
