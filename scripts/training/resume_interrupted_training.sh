#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/training/mamba_train.yaml}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to checkpoint-* for strict Trainer resume.}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
export USE_HUB_KERNELS="${USE_HUB_KERNELS:-0}"

ARGS=(--config "$CONFIG" --resume-from-checkpoint "$CHECKPOINT")
if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

echo "===== RESUME INTERRUPTED TRAINING ====="
echo "This restores Trainer checkpoint state through training.resume_from_checkpoint."
echo "CONFIG=$CONFIG"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_DIR=${OUTPUT_DIR:-from_config}"
echo "USE_HUB_KERNELS=$USE_HUB_KERNELS"
"$PYTHON" -m synthetic_time_series_clean.training.train_mamba "${ARGS[@]}"
