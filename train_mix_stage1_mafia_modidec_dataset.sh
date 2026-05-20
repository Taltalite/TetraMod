#!/usr/bin/env bash
set -Eeuo pipefail

# Merge already-built mAFiA and MoDiDeC Stage 1 datasets into one
# train_promote control dataset.
#
# Required inputs:
#   MAFIA_DATASET_DIR=/path/to/stage1_train_mafia_...
#   MODIDEC_DATASET_DIR=/path/to/stage1_train_modidec_...
#   WORK_ROOT=/path/to/output/work_root
#
# This script does not basecall or rebuild per-run chunks.  It only re-merges
# existing Stage 1 numpy datasets with source/class balancing:
#   mafia positive == mafia negative == modidec positive == modidec negative
# unless SOURCE_CLASS_CAP is set lower.

REPO="${REPO:-/home/lijy/workspace/TetraMod/}"
WORK_ROOT="${WORK_ROOT:-}"

MAFIA_DATASET_DIR="${MAFIA_DATASET_DIR:-}"
MODIDEC_DATASET_DIR="${MODIDEC_DATASET_DIR:-}"
MIX_DATASET_NAME="${MIX_DATASET_NAME:-stage1_train_mafia_modidec_m6a}"
MIX_DATASET_DIR="${MIX_DATASET_DIR:-$WORK_ROOT/chunks/$MIX_DATASET_NAME}"

VALID_FRACTION="${VALID_FRACTION:-0.1}"
SEED="${SEED:-114514}"
BALANCE_MODE="${BALANCE_MODE:-source-class}"
BALANCE_VALIDATION="${BALANCE_VALIDATION:-1}"
SOURCE_CLASS_CAP="${SOURCE_CLASS_CAP:-}"
SCAN_WORKERS="${SCAN_WORKERS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
RUN_DATASET_CHECK="${RUN_DATASET_CHECK:-1}"
CHECK_REPORT_DIR="${CHECK_REPORT_DIR:-$REPO/dataset_check_res/$MIX_DATASET_NAME/check_reports}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -e "$path" ]] || fail "Missing required path: $path"
}

require_dataset() {
  local name="$1"
  local path="$2"
  [[ -n "$path" ]] || fail "Set $name to an existing Stage 1 dataset directory."
  require_file "$path/chunks.npy"
  require_file "$path/references.npy"
  require_file "$path/reference_lengths.npy"
  require_file "$path/mod_targets.npy"
  require_file "$path/metadata.npz"
}

prepare_repo_context() {
  [[ -n "$WORK_ROOT" ]] || fail "Set WORK_ROOT to the output work directory."
  require_file "$REPO"
  cd "$REPO"
}

validate_inputs() {
  require_dataset MAFIA_DATASET_DIR "$MAFIA_DATASET_DIR"
  require_dataset MODIDEC_DATASET_DIR "$MODIDEC_DATASET_DIR"
  case "$BALANCE_MODE" in
    source-class|motif|none) ;;
    *) fail "BALANCE_MODE must be one of: source-class, motif, none. Current: $BALANCE_MODE" ;;
  esac
}

print_config() {
  echo "[config]"
  echo "  REPO=$REPO"
  echo "  WORK_ROOT=$WORK_ROOT"
  echo "  MAFIA_DATASET_DIR=$MAFIA_DATASET_DIR"
  echo "  MODIDEC_DATASET_DIR=$MODIDEC_DATASET_DIR"
  echo "  MIX_DATASET_DIR=$MIX_DATASET_DIR"
  echo "  VALID_FRACTION=$VALID_FRACTION"
  echo "  BALANCE_MODE=$BALANCE_MODE"
  echo "  BALANCE_VALIDATION=$BALANCE_VALIDATION"
  echo "  SOURCE_CLASS_CAP=${SOURCE_CLASS_CAP:-<none>}"
  echo "  SCAN_WORKERS=$SCAN_WORKERS"
}

merge_dataset() {
  if [[ "$SKIP_EXISTING" == "1" \
      && -s "$MIX_DATASET_DIR/chunks.npy" \
      && -s "$MIX_DATASET_DIR/references.npy" \
      && -s "$MIX_DATASET_DIR/reference_lengths.npy" \
      && -s "$MIX_DATASET_DIR/mod_targets.npy" \
      && -s "$MIX_DATASET_DIR/metadata.npz" ]]; then
    echo "[skip merge] $MIX_DATASET_DIR"
    return
  fi

  mkdir -p "$(dirname "$MIX_DATASET_DIR")"
  local balance_args=(--balance-mode "$BALANCE_MODE")
  if [[ -n "$SOURCE_CLASS_CAP" ]]; then
    balance_args+=(--source-class-cap "$SOURCE_CLASS_CAP")
  fi
  if [[ "$BALANCE_VALIDATION" == "1" ]]; then
    balance_args+=(--balance-validation)
  fi

  echo "[merge mixed stage1] $MIX_DATASET_DIR"
  python gen_data/merge_mafia_stage1_datasets.py \
    --dataset "mafia:$MAFIA_DATASET_DIR" \
    --dataset "modidec:$MODIDEC_DATASET_DIR" \
    --output-dir "$MIX_DATASET_DIR" \
    --valid-fraction "$VALID_FRACTION" \
    "${balance_args[@]}" \
    --scan-workers "$SCAN_WORKERS" \
    --seed "$SEED"

  if [[ -s "$MIX_DATASET_DIR/mafia_stage1_merge_summary.json" ]]; then
    cp "$MIX_DATASET_DIR/mafia_stage1_merge_summary.json" \
      "$MIX_DATASET_DIR/promote_stage1_mix_summary.json"
  fi
}

write_source_manifest() {
  mkdir -p "$MIX_DATASET_DIR"
  {
    printf 'source\tdirectory\n'
    printf 'mafia\t%s\n' "$MAFIA_DATASET_DIR"
    printf 'modidec\t%s\n' "$MODIDEC_DATASET_DIR"
  } > "$MIX_DATASET_DIR/source_datasets.tsv"
}

run_dataset_check() {
  if [[ "$RUN_DATASET_CHECK" != "1" ]]; then
    return
  fi

  echo "[dataset check] $CHECK_REPORT_DIR"
  python dataset_check/check_mafia_stage1_dataset.py \
    "$MIX_DATASET_DIR" \
    --output-dir "$CHECK_REPORT_DIR"
}

main() {
  prepare_repo_context
  validate_inputs
  print_config
  merge_dataset
  write_source_manifest
  run_dataset_check

  echo "[done] Mixed mAFiA + MoDiDeC Stage 1 dataset: $MIX_DATASET_DIR"
  echo "Use this dataset for train_promote:"
  echo "  --directory \"$MIX_DATASET_DIR\""
}

main "$@"
