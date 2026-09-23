#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/inference/vllm_generate.yaml}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a training run directory.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for memorization generation outputs.}"
CHECKPOINT="${CHECKPOINT:-}"
RUN_NAME="${RUN_NAME:-}"
TEMP_TAG="${TEMP_TAG:-}"
MAX_TRAIN_PATIENTS="${MAX_TRAIN_PATIENTS:-1000}"
TRAIN_SELECTION_MODE="${TRAIN_SELECTION_MODE:-random_ids}"
RANDOM_SAMPLE_TRAIN="${RANDOM_SAMPLE_TRAIN:-1}"
MIN_EVENTS="${MIN_EVENTS:-8}"
FRACTIONS="${FRACTIONS:-0,0.25,0.5,0.75}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
TOTAL_MAX_LENGTH="${TOTAL_MAX_LENGTH:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
DO_SAMPLE="${DO_SAMPLE:-1}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-10}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

ARGS=(
  --config "$CONFIG"
  --run-dir "$RUN_DIR"
  --output-dir "$OUTPUT_DIR"
  --max-train-patients "$MAX_TRAIN_PATIENTS"
  --train-selection-mode "$TRAIN_SELECTION_MODE"
  --min-events "$MIN_EVENTS"
  --fractions "$FRACTIONS"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --total-max-length "$TOTAL_MAX_LENGTH"
  --max-model-len "$MAX_MODEL_LEN"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --repetition-penalty "$REPETITION_PENALTY"
  --seed "$SEED"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)
if [ -n "$CHECKPOINT" ]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [ -n "$RUN_NAME" ]; then
  ARGS+=(--run-name "$RUN_NAME")
fi
if [ -n "$TEMP_TAG" ]; then
  ARGS+=(--temp-tag "$TEMP_TAG")
fi
if [ "$RANDOM_SAMPLE_TRAIN" = "1" ]; then
  ARGS+=(--random-sample-train)
else
  ARGS+=(--no-random-sample-train)
fi
if [ "$DO_SAMPLE" = "1" ]; then
  ARGS+=(--do-sample)
else
  ARGS+=(--no-sample)
fi

echo "===== MEMORIZATION vLLM GENERATION ====="
echo "CONFIG=$CONFIG"
echo "RUN_DIR=$RUN_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DATA_SPLIT_USED=train"
echo "MAX_TRAIN_PATIENTS=$MAX_TRAIN_PATIENTS"
echo "TRAIN_SELECTION_MODE=$TRAIN_SELECTION_MODE"
echo "RANDOM_SAMPLE_TRAIN=$RANDOM_SAMPLE_TRAIN"
echo "MIN_EVENTS=$MIN_EVENTS"
echo "FRACTIONS=$FRACTIONS"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "TOTAL_MAX_LENGTH=$TOTAL_MAX_LENGTH"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
"$PYTHON" -m synthetic_time_series_clean.inference.generate_memorization "${ARGS[@]}"
