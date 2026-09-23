#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

SBATCH_SCRIPT="./vllm_synthetic_data_generation.sh"
RESULTS_ROOT="${RESULTS_ROOT:-/path/to/results}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$RESULTS_ROOT/slurm_logs}"
BASE_OUTPUT="${BASE_OUTPUT:-$RESULTS_ROOT/JUNE17_GENERATE}"
SUBMIT_STAMP=$(date +"%Y%m%d_%H%M%S")
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_ORIGINAL="${INCLUDE_ORIGINAL:-1}"
INCLUDE_DEATH="${INCLUDE_DEATH:-0}"

ORIGINAL_TEMPS=(
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

DEATH_TEMPS=(
  0
)

export EVAL_DATASET=held_out
export MAX_EVAL_PATIENTS=3000
export RANDOM_SAMPLE_EVAL=1
export MAX_TRAIN_PATIENTS=""
export BATCH_SIZE=4
export SYNTHETIC_SAMPLES_PER_EVAL=1
export MAX_NEW_TOKENS=16384
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

submit_run() {
  local run_name="$1"
  local run_dir="$2"
  shift 2
  local temps=("$@")

  if [ ! -d "$run_dir" ]; then
    echo "ERROR: RUN_DIR does not exist: $run_dir"
    exit 1
  fi
  if [ ! -f "$run_dir/resolved_config.yaml" ]; then
    echo "ERROR: resolved_config.yaml not found: $run_dir/resolved_config.yaml"
    exit 1
  fi

  local output_root="$BASE_OUTPUT/$run_name"
  if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$output_root"
  fi

  for temp in "${temps[@]}"; do
    local tag
    tag=$(temp_tag "$temp")
    local output_dir="$output_root/temp_${tag}_${SUBMIT_STAMP}"

    if [[ "$output_dir"/ == "$run_dir"/* ]]; then
      echo "ERROR: OUTPUT_DIR is inside RUN_DIR; refusing to risk modifying the model directory."
      echo "RUN_DIR=$run_dir"
      echo "OUTPUT_DIR=$output_dir"
      exit 1
    fi

    echo
    echo "Submitting clean June17 synthetic generation"
    echo "RUN_NAME=$run_name"
    echo "TEMP=$temp"
    echo "TEMP_TAG=$tag"
    echo "RUN_DIR=$run_dir"
    echo "OUTPUT_DIR=$output_dir"

    if [ "$DRY_RUN" = "1" ]; then
      printf 'DRY_RUN: sbatch %q %q %q %q\n' "$SBATCH_SCRIPT" "$temp" "$run_dir" "$output_dir"
    else
      sbatch "$SBATCH_SCRIPT" "$temp" "$run_dir" "$output_dir"
    fi

    submission_count=$((submission_count + 1))
  done
}

if [ ! -f "$SBATCH_SCRIPT" ]; then
  echo "ERROR: SBATCH_SCRIPT does not exist: $SBATCH_SCRIPT"
  exit 1
fi
mkdir -p "$SLURM_LOG_DIR"

echo "===== CLEAN vLLM JUNE17 SYNTHETIC BATCH SUBMIT ====="
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "SBATCH_SCRIPT=$SBATCH_SCRIPT"
echo "SLURM_LOG_DIR=$SLURM_LOG_DIR"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "BASE_OUTPUT=$BASE_OUTPUT"
echo "SUBMIT_STAMP=$SUBMIT_STAMP"
echo "DRY_RUN=$DRY_RUN"
echo "INCLUDE_ORIGINAL=$INCLUDE_ORIGINAL"
echo "INCLUDE_DEATH=$INCLUDE_DEATH"
echo "EVAL_DATASET=$EVAL_DATASET"
echo "MAX_EVAL_PATIENTS=$MAX_EVAL_PATIENTS"
echo "RANDOM_SAMPLE_EVAL=$RANDOM_SAMPLE_EVAL"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "TOP_P=$TOP_P"
echo "TOP_K=$TOP_K"
echo "REPETITION_PENALTY=$REPETITION_PENALTY"
echo "SEED=$SEED"
echo "DTYPE=$DTYPE"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"

submission_count=0

if [ "$INCLUDE_ORIGINAL" = "1" ]; then
  submit_run \
    "original_checkpoint" \
    "/result/path/2026-mm-dd/hh-mm-ss_b4_g8_3691289/unnormalized/0" \
    "${ORIGINAL_TEMPS[@]}"
fi

if [ "$INCLUDE_DEATH" = "1" ]; then
  submit_run \
    "original_checkpoint_addtion_iteration_death_data_for_km" \
    "/result/path/2026-mm-dd/hh-mm-ss_b4_g8_3971026/unnormalized/0" \
    "${DEATH_TEMPS[@]}"
fi

echo
echo "===== SUBMIT COMPLETE ====="
if [ "$DRY_RUN" = "1" ]; then
  echo "Total jobs planned: $submission_count"
else
  echo "Total jobs submitted: $submission_count"
fi
