#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/inference/vllm_generate.yaml}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a training run directory.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for vLLM generation outputs.}"
EVAL_DATASET="${EVAL_DATASET:-held_out}"
CHECKPOINT="${CHECKPOINT:-}"

ARGS=(--config "$CONFIG" --run-dir "$RUN_DIR" --output-dir "$OUTPUT_DIR" --eval-dataset "$EVAL_DATASET")
if [ -n "$CHECKPOINT" ]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

echo "===== vLLM GENERATION ====="
echo "CONFIG=$CONFIG"
echo "RUN_DIR=$RUN_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "EVAL_DATASET=$EVAL_DATASET"
"$PYTHON" -m synthetic_time_series_clean.inference.generate_vllm "${ARGS[@]}"

