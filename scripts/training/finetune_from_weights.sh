#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/training/mamba_train.yaml}"
MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:?Set MODEL_LOCAL_PATH to the model/checkpoint weights to fine-tune from.}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
export USE_HUB_KERNELS="${USE_HUB_KERNELS:-0}"

ARGS=(--config "$CONFIG" --model-local-path "$MODEL_LOCAL_PATH")
if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

echo "===== FINE-TUNE FROM WEIGHTS ====="
echo "This starts a new Trainer run and does not restore optimizer/scheduler/global_step."
echo "CONFIG=$CONFIG"
echo "MODEL_LOCAL_PATH=$MODEL_LOCAL_PATH"
echo "OUTPUT_DIR=${OUTPUT_DIR:-from_config}"
echo "USE_HUB_KERNELS=$USE_HUB_KERNELS"
"$PYTHON" -m synthetic_time_series_clean.training.train_mamba "${ARGS[@]}"
