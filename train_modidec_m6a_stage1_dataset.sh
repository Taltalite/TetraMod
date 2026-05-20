#!/usr/bin/env bash
set -Eeuo pipefail

# Build a MoDiDeC RNA002 m6A Stage 1 dataset from an already-basecalled BAM.
#
# Preprocessing chain:
#   POD5 raw signal + Dorado BAM with --emit-moves
#     -> match Supplementary Table 1 m6A oligo sequences in each read
#     -> positive chunks: windows containing explicit m6A oligo center A
#     -> internal negative chunks: A-containing windows outside matched m6A oligo intervals
#     -> balanced Stage 1 train/validation numpy dataset
#
# Required input format:
#   MODIDEC_BAM_SPECS="run_name:/path/to/basecalled.bam:/path/to/pod5_dir_or_file.pod5"
#
# Multiple already-basecalled runs can be separated by semicolons:
#   MODIDEC_BAM_SPECS="run1:/bam/a.bam:/pod5/a;run2:/bam/b.bam:/pod5/b"
#
# Optional heldout/test pass over the same BAM/POD5 or another pair:
#   MODIDEC_HELDOUT_BAM_SPECS="heldout:/path/to/basecalled.bam:/path/to/pod5"
#   MODIDEC_HELDOUT_OLIGO_IDS="modidec_m6A_11"

REPO="${REPO:-/home/lijy/workspace/TetraMod/}"
WORK_ROOT="${WORK_ROOT:-/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002}"

DATASET_WORKERS="${DATASET_WORKERS:-8}"
CHUNK_LEN="${CHUNK_LEN:-5000}"
OVERLAP="${OVERLAP:-500}"
VALID_FRACTION="${VALID_FRACTION:-0.1}"
SEED="${SEED:-114514}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_DATASET_CHECK="${RUN_DATASET_CHECK:-0}"
BUILD_HELDOUT="${BUILD_HELDOUT:-1}"
MERGE_BALANCE_TRAIN="${MERGE_BALANCE_TRAIN:-0}"
MERGE_BALANCE_MODE="${MERGE_BALANCE_MODE:-source-class}"
MERGE_BALANCE_VALIDATION="${MERGE_BALANCE_VALIDATION:-1}"
SOURCE_CLASS_CAP="${SOURCE_CLASS_CAP:-}"
MAX_RECORDS="${MAX_RECORDS:--1}"
MAX_CHUNKS="${MAX_CHUNKS:--1}"
MIN_OLIGO_IDENTITY="${MIN_OLIGO_IDENTITY:-0.86}"
MAX_OLIGO_MISMATCHES="${MAX_OLIGO_MISMATCHES:-4}"
NEGATIVE_CHUNKS_PER_POSITIVE="${NEGATIVE_CHUNKS_PER_POSITIVE:-2}"
NEGATIVE_EXCLUSION_BASES="${NEGATIVE_EXCLUSION_BASES:-0}"
NEGATIVE_LABEL_MODE="${NEGATIVE_LABEL_MODE:-center}"
NEGATIVE_MOTIF_MODE="${NEGATIVE_MOTIF_MODE:-positive-motifs}"

TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-stage1_train_modidec_m6a_rna002}"
HELDOUT_ROOT_NAME="${HELDOUT_ROOT_NAME:-heldout_modidec_m6a_rna002}"

MODIDEC_BAM_SPECS="${MODIDEC_BAM_SPECS:-modidec_train:/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/bam/modidec_train.bam:/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/m6A_pod5}"
MODIDEC_HELDOUT_BAM_SPECS="${MODIDEC_HELDOUT_BAM_SPECS:-modidec_h11:/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/bam/modidec_train.bam:/data/biolab-nvme-pcie2/lijy/tetramod_modidec_rna002/m6A_pod5}"
MODIDEC_TRAIN_OLIGO_IDS="${MODIDEC_TRAIN_OLIGO_IDS:-modidec_m6A_01,modidec_m6A_02,modidec_m6A_03,modidec_m6A_04,modidec_m6A_05,modidec_m6A_06,modidec_m6A_07,modidec_m6A_08,modidec_m6A_09,modidec_m6A_10}"
MODIDEC_HELDOUT_OLIGO_IDS="${MODIDEC_HELDOUT_OLIGO_IDS:-modidec_m6A_11}"

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
  local split="$2"
  local raw run_id bam_path pod5_path remainder

  [[ -n "$specs" ]] || return 0
  IFS=';' read -r -a raw_specs <<< "$specs"
  for raw in "${raw_specs[@]}"; do
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    [[ -n "$raw" ]] || continue
    IFS=':' read -r run_id bam_path pod5_path remainder <<< "$raw"
    [[ -z "${remainder:-}" ]] || fail "Invalid spec with too many ':' fields: $raw"
    [[ -n "$run_id" && -n "$bam_path" && -n "$pod5_path" ]] || fail "Spec must be run:bam:pod5, got: $raw"
    run_id="$(sanitize_run_id "$run_id")"
    printf '%s\t%s\t%s\t%s\n' "$run_id" "$split" "$bam_path" "$pod5_path"
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

spec_file() {
  local path="$WORK_ROOT/manifests/bam_specs.tsv"
  {
    parse_specs "$MODIDEC_BAM_SPECS" "train"
    if [[ "$BUILD_HELDOUT" == "1" ]]; then
      parse_specs "$MODIDEC_HELDOUT_BAM_SPECS" "heldout"
    fi
  } > "$path"
  printf '%s\n' "$path"
}

run_ids_for_split() {
  local specs_file="$1"
  local requested_split="$2"
  awk -F'\t' -v requested_split="$requested_split" '$2 == requested_split { print $1 }' "$specs_file"
}

field_for_run() {
  local specs_file="$1"
  local run_id="$2"
  local field="$3"
  awk -F'\t' -v id="$run_id" -v field="$field" '$1 == id { print $field }' "$specs_file"
}

split_for_run() {
  field_for_run "$1" "$2" 2
}

bam_path_for_run() {
  field_for_run "$1" "$2" 3
}

pod5_path_for_run() {
  field_for_run "$1" "$2" 4
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

build_run_dataset() {
  local specs_file="$1"
  local run_id="$2"
  local output_dir="$3"
  local pod5_dir bam split oligo_ids

  bam="$(bam_path_for_run "$specs_file" "$run_id")"
  pod5_dir="$(pod5_input_dir_for_run "$specs_file" "$run_id")"
  split="$(split_for_run "$specs_file" "$run_id")"
  oligo_ids="$(oligo_ids_for_split "$split")"

  require_file "$bam"
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
  python gen_data/create_modidec_m6a_stage1_dataset.py \
    --bam-file "$bam" \
    --pod5-dir "$pod5_dir" \
    --output-dir "$output_dir" \
    --oligo-manifest "$OLIGO_MANIFEST" \
    --oligo-ids "$oligo_ids" \
    --run-id "$run_id" \
    --sample-type rna \
    --rna002 \
    --chunk-len "$CHUNK_LEN" \
    --overlap "$OVERLAP" \
    --workers "$DATASET_WORKERS" \
    --max-records "$MAX_RECORDS" \
    --max-chunks "$MAX_CHUNKS" \
    --min-oligo-identity "$MIN_OLIGO_IDENTITY" \
    --max-oligo-mismatches "$MAX_OLIGO_MISMATCHES" \
    --negative-chunks-per-positive "$NEGATIVE_CHUNKS_PER_POSITIVE" \
    --negative-exclusion-bases "$NEGATIVE_EXCLUSION_BASES" \
    --negative-label-mode "$NEGATIVE_LABEL_MODE" \
    --negative-motif-mode "$NEGATIVE_MOTIF_MODE" \
    --seed "$SEED"
}

merge_train_dataset() {
  local specs_file="$1"
  local merge_args=()
  local run_id

  while IFS= read -r run_id; do
    merge_args+=(--dataset "$run_id:$WORK_ROOT/chunks/per_run/$run_id")
  done < <(run_ids_for_split "$specs_file" train)

  [[ "${#merge_args[@]}" -gt 0 ]] || fail "No train BAM specs were provided."

  echo "[merge train] $TRAIN_DATASET_DIR"
  local balance_args=()
  if [[ -n "$MERGE_BALANCE_MODE" ]]; then
    balance_args+=(--balance-mode "$MERGE_BALANCE_MODE")
  elif [[ "$MERGE_BALANCE_TRAIN" != "1" ]]; then
    balance_args+=(--no-balance-train)
  fi
  if [[ -n "$SOURCE_CLASS_CAP" ]]; then
    balance_args+=(--source-class-cap "$SOURCE_CLASS_CAP")
  fi
  if [[ "$MERGE_BALANCE_VALIDATION" == "1" ]]; then
    balance_args+=(--balance-validation)
  fi

  python gen_data/merge_mafia_stage1_datasets.py \
    "${merge_args[@]}" \
    --output-dir "$TRAIN_DATASET_DIR" \
    --valid-fraction "$VALID_FRACTION" \
    "${balance_args[@]}" \
    --seed "$SEED"

  if [[ -s "$TRAIN_DATASET_DIR/mafia_stage1_merge_summary.json" ]]; then
    cp "$TRAIN_DATASET_DIR/mafia_stage1_merge_summary.json" \
      "$TRAIN_DATASET_DIR/modidec_stage1_merge_summary.json"
  fi
}

write_run_lists() {
  local specs_file="$1"
  mkdir -p "$TRAIN_DATASET_DIR"
  run_ids_for_split "$specs_file" train > "$TRAIN_DATASET_DIR/train_run_ids.txt"
  if [[ "$BUILD_HELDOUT" == "1" ]]; then
    mkdir -p "$HELDOUT_DATASET_ROOT"
    run_ids_for_split "$specs_file" heldout > "$HELDOUT_DATASET_ROOT/heldout_run_ids.txt"
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

prepare_common_dirs() {
  mkdir -p \
    "$MANIFEST_DIR" \
    "$WORK_ROOT/pod5_input_dirs" \
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
  [[ -n "$MODIDEC_BAM_SPECS" ]] || fail "Set MODIDEC_BAM_SPECS to run:bam:pod5."
  [[ -s "$specs_file" ]] || fail "No BAM specs were parsed."
  while IFS=$'\t' read -r run_id _split bam path; do
    [[ -n "$run_id" ]] || continue
    require_file "$bam"
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
  echo "  REPO=$REPO"
  echo "  WORK_ROOT=$WORK_ROOT"
  echo "  CHUNK_LEN=$CHUNK_LEN OVERLAP=$OVERLAP"
  echo "  NEGATIVE_CHUNKS_PER_POSITIVE=$NEGATIVE_CHUNKS_PER_POSITIVE"
  echo "  NEGATIVE_EXCLUSION_BASES=$NEGATIVE_EXCLUSION_BASES"
  echo "  NEGATIVE_LABEL_MODE=$NEGATIVE_LABEL_MODE"
  echo "  NEGATIVE_MOTIF_MODE=$NEGATIVE_MOTIF_MODE"
  echo "  MERGE_BALANCE_TRAIN=$MERGE_BALANCE_TRAIN"
  echo "  MERGE_BALANCE_MODE=$MERGE_BALANCE_MODE"
  echo "  MERGE_BALANCE_VALIDATION=$MERGE_BALANCE_VALIDATION"
  echo "  SOURCE_CLASS_CAP=${SOURCE_CLASS_CAP:-<none>}"
  echo "  TRAIN_DATASET_DIR=$TRAIN_DATASET_DIR"
  echo "  HELDOUT_DATASET_ROOT=$HELDOUT_DATASET_ROOT"
  echo "  OLIGO_MANIFEST=$OLIGO_MANIFEST"
  echo "  MODIDEC_TRAIN_OLIGO_IDS=${MODIDEC_TRAIN_OLIGO_IDS:-<all>}"
  echo "  MODIDEC_HELDOUT_OLIGO_IDS=${MODIDEC_HELDOUT_OLIGO_IDS:-<all>}"
}

run_pipeline() {
  local specs_file="$1"
  local run_id

  while IFS= read -r run_id; do
    build_run_dataset "$specs_file" "$run_id" "$WORK_ROOT/chunks/per_run/$run_id"
  done < <(run_ids_for_split "$specs_file" train)

  if [[ "$BUILD_HELDOUT" == "1" ]]; then
    while IFS= read -r run_id; do
      build_run_dataset "$specs_file" "$run_id" "$HELDOUT_DATASET_ROOT/$run_id"
    done < <(run_ids_for_split "$specs_file" heldout)
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
  prepare_common_dirs
  prepare_repo_context
  local specs_file
  specs_file="$(spec_file)"
  validate_inputs "$specs_file"
  print_config
  run_pipeline "$specs_file"
}

main "$@"
