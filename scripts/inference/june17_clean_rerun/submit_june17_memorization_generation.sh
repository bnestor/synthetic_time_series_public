#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

SBATCH_SCRIPT="./vllm_memorization_generation.sh"
RESULTS_ROOT="${RESULTS_ROOT:-/path/to/results}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$RESULTS_ROOT/slurm_logs}"
BASE_OUTPUT="${BASE_OUTPUT:-$RESULTS_ROOT/JUNE17_MEMORIZATION}"
SUBMIT_STAMP=$(date +"%Y%m%d_%H%M%S")
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_ORIGINAL="${INCLUDE_ORIGINAL:-1}"
INCLUDE_DEATH="${INCLUDE_DEATH:-0}"

TEMPS=(
  0
  0.1
  0.2
  0.3
  0.4
  0.5
  0.6
  0.7
  0.8
  0.9
  1.0
)

export MAX_TRAIN_PATIENTS=1000
export TRAIN_SELECTION_MODE=post_filter_shuffle
export RANDOM_SAMPLE_TRAIN=1
export MIN_EVENTS=8
export FRACTIONS=0,0.25,0.5,0.75
export BATCH_SIZE=4
export MAX_NEW_TOKENS=16384
export TOTAL_MAX_LENGTH=16384
export MAX_MODEL_LEN=16384
export TOP_P=0.9
export TOP_K=10
export REPETITION_PENALTY=1.1
export SEED=42
export DTYPE=bfloat16
export GPU_MEMORY_UTILIZATION=0.85
export CHECKPOINT=""


temp_tag() {
  python - "$1" <<'PY'
import sys

temp = float(sys.argv[1])
if abs(temp - 0.0) < 1e-9:
    print("0")
else:
    print(f"{int(round(temp * 10)):02d}")
PY
}

train_selection_tag() {
  python - "$1" <<'PY'
import sys

mode = sys.argv[1].strip().lower()
print({
    "head_ids": "headids",
    "random_ids": "randomids",
    "post_filter_shuffle": "postfilter",
}.get(mode, mode.replace("_", "")))
PY
}

fraction_tag() {
  python - "$1" <<'PY'
import sys

parts = []
for item in sys.argv[1].split(","):
    item = item.strip()
    if item:
        parts.append(str(int(round(float(item) * 100))))
print("frac" + "_".join(parts))
PY
}

length_tag() {
  python - "$1" "$2" <<'PY'
import sys

total = sys.argv[1].strip().lower()
max_new = sys.argv[2].strip()
if total in {"", "none", "null", "-1", "0"}:
    print(f"nototal_new{max_new}")
else:
    print(f"total{total}")
PY
}

submit_run() {
  local run_name="$1"
  local run_dir="$2"
  local data_override_path="${3:-}"
  local selection_tag
  local fractions_tag
  local total_length_tag
  selection_tag=$(train_selection_tag "$TRAIN_SELECTION_MODE")
  fractions_tag=$(fraction_tag "$FRACTIONS")
  total_length_tag=$(length_tag "$TOTAL_MAX_LENGTH" "$MAX_NEW_TOKENS")

  if [ ! -d "$run_dir" ]; then
    echo "ERROR: RUN_DIR does not exist: $run_dir"
    exit 1
  fi
  if [ ! -f "$run_dir/resolved_config.yaml" ]; then
    echo "ERROR: resolved_config.yaml not found: $run_dir/resolved_config.yaml"
    exit 1
  fi

  local output_root="$BASE_OUTPUT/$run_name"
  if [[ "$output_root"/ == "$run_dir"/* ]]; then
    echo "ERROR: OUTPUT_ROOT is inside RUN_DIR; refusing to risk modifying the model directory."
    echo "RUN_DIR=$run_dir"
    echo "OUTPUT_ROOT=$output_root"
    exit 1
  fi

  for temp in "${TEMPS[@]}"; do
    local tag
    tag=$(temp_tag "$temp")
    local temp_output_root="$output_root/temp_${tag}"
    local output_subdir="generation_${SUBMIT_STAMP}_n${MAX_TRAIN_PATIENTS}_${selection_tag}_min${MIN_EVENTS}_${fractions_tag}_${total_length_tag}_b${BATCH_SIZE}"
    local output_dir="$temp_output_root/$output_subdir"

    if [[ "$output_dir"/ == "$run_dir"/* ]]; then
      echo "ERROR: OUTPUT_DIR is inside RUN_DIR; refusing to risk modifying the model directory."
      echo "RUN_DIR=$run_dir"
      echo "OUTPUT_DIR=$output_dir"
      exit 1
    fi

    echo
    echo "Submitting clean June17 memorization generation"
    echo "RUN_NAME=$run_name"
    echo "TEMP=$temp"
    echo "TEMP_TAG=$tag"
    echo "RUN_DIR=$run_dir"
    echo "OUTPUT_BASE=$BASE_OUTPUT"
    echo "TEMP_OUTPUT_ROOT=$temp_output_root"
    echo "OUTPUT_DIR=$output_dir"
    echo "DATA_OVERRIDE_PATH=$data_override_path"
    echo "FRACTIONS=$FRACTIONS"

    if [ "$DRY_RUN" = "1" ]; then
      if [ -n "$data_override_path" ]; then
        printf 'DRY_RUN: sbatch %q %q %q %q %q %q %q\n' "$SBATCH_SCRIPT" "$temp" "$run_name" "$run_dir" "$BASE_OUTPUT" "$SUBMIT_STAMP" "$data_override_path"
      else
        printf 'DRY_RUN: sbatch %q %q %q %q %q %q\n' "$SBATCH_SCRIPT" "$temp" "$run_name" "$run_dir" "$BASE_OUTPUT" "$SUBMIT_STAMP"
      fi
    else
      if [ -n "$data_override_path" ]; then
        sbatch "$SBATCH_SCRIPT" "$temp" "$run_name" "$run_dir" "$BASE_OUTPUT" "$SUBMIT_STAMP" "$data_override_path"
      else
        sbatch "$SBATCH_SCRIPT" "$temp" "$run_name" "$run_dir" "$BASE_OUTPUT" "$SUBMIT_STAMP"
      fi
    fi

    submission_count=$((submission_count + 1))
  done
}

if [ ! -f "$SBATCH_SCRIPT" ]; then
  echo "ERROR: SBATCH_SCRIPT does not exist: $SBATCH_SCRIPT"
  exit 1
fi
mkdir -p "$SLURM_LOG_DIR"

echo "===== CLEAN vLLM JUNE17 MEMORIZATION BATCH SUBMIT ====="
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "SBATCH_SCRIPT=$SBATCH_SCRIPT"
echo "SLURM_LOG_DIR=$SLURM_LOG_DIR"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "BASE_OUTPUT=$BASE_OUTPUT"
echo "SUBMIT_STAMP=$SUBMIT_STAMP"
echo "DRY_RUN=$DRY_RUN"
echo "INCLUDE_ORIGINAL=$INCLUDE_ORIGINAL"
echo "INCLUDE_DEATH=$INCLUDE_DEATH"
echo "MAX_TRAIN_PATIENTS=$MAX_TRAIN_PATIENTS"
echo "TRAIN_SELECTION_MODE=$TRAIN_SELECTION_MODE"
echo "RANDOM_SAMPLE_TRAIN=$RANDOM_SAMPLE_TRAIN"
echo "MIN_EVENTS=$MIN_EVENTS"
echo "FRACTIONS=$FRACTIONS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "TOTAL_MAX_LENGTH=$TOTAL_MAX_LENGTH"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "TOP_P=$TOP_P"
echo "TOP_K=$TOP_K"
echo "REPETITION_PENALTY=$REPETITION_PENALTY"
echo "SEED=$SEED"
echo "DTYPE=$DTYPE"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"

submission_count=0

if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$BASE_OUTPUT"
fi

if [ "$INCLUDE_ORIGINAL" = "1" ]; then
  submit_run \
    "JUNE17_Original_checkpoint_memorization" \
    "/result/path/2026-mm-dd/hh-mm-ss_b4_g8_3691289/unnormalized/0"
fi

if [ "$INCLUDE_DEATH" = "1" ]; then
  submit_run \
    "JUNE17_Death_checkpoint_memorization" \
    "/result/path/2026-mm-dd/hh-mm-ss_b4_g8_3971026/unnormalized/0"
fi


echo
echo "===== SUBMIT COMPLETE ====="
if [ "$DRY_RUN" = "1" ]; then
  echo "Total jobs planned: $submission_count"
else
  echo "Total jobs submitted: $submission_count"
fi
