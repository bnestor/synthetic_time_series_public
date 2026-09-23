#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/evaluation/sliced_wd.yaml}"
TASK_TYPE="${TASK_TYPE:-synthetic}"  # synthetic or memorization
MANIFEST="${MANIFEST:?Set MANIFEST to a CSV with a source_path column.}"
EMBEDDING_MANIFEST="${EMBEDDING_MANIFEST:?Set EMBEDDING_MANIFEST explicitly. This script never creates it.}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT for sliced WD outputs.}"

WORK_DIR="${WORK_DIR:-$PROJECT_DIR}"
EMBEDDING_NAMES="${EMBEDDING_NAMES:-bge_m3}"
DEVICE="${DEVICE:-auto}"
N_PROJECTIONS="${N_PROJECTIONS:-128}"
SEED="${SEED:-42}"
MAX_ROWS="${MAX_ROWS:-}"
EXPECTED_INPUT_COUNT="${EXPECTED_INPUT_COUNT:-}"
FILTERS="${FILTERS:-}"
SAVE_EMBEDDINGS="${SAVE_EMBEDDINGS:-0}"
DRY_RUN="${DRY_RUN:-0}"
REPAIR_EVENT_TEXT="${REPAIR_EVENT_TEXT:-}"

ARGS=(
  --config "$CONFIG"
  --work-dir "$WORK_DIR"
  --task-type "$TASK_TYPE"
  --manifest "$MANIFEST"
  --embedding-manifest "$EMBEDDING_MANIFEST"
  --embedding-names "$EMBEDDING_NAMES"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
  --n-projections "$N_PROJECTIONS"
  --seed "$SEED"
)

case "$REPAIR_EVENT_TEXT" in
  1|true|TRUE|yes|YES)
    ARGS+=(--repair-event-text)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-repair-event-text)
    ;;
  "")
    ;;
  *)
    echo "ERROR: REPAIR_EVENT_TEXT must be 1/0, true/false, yes/no, or empty." >&2
    exit 1
    ;;
esac

if [ -n "$MAX_ROWS" ]; then
  ARGS+=(--max-rows "$MAX_ROWS")
fi
if [ -n "$EXPECTED_INPUT_COUNT" ]; then
  ARGS+=(--expected-input-count "$EXPECTED_INPUT_COUNT")
fi
if [ "$SAVE_EMBEDDINGS" = "1" ]; then
  ARGS+=(--save-embeddings)
fi
if [ "$DRY_RUN" = "1" ]; then
  ARGS+=(--dry-run)
fi
for filter_expr in $FILTERS; do
  ARGS+=(--filter "$filter_expr")
done

echo "===== SLICED WD ====="
echo "TASK_TYPE=$TASK_TYPE"
echo "CONFIG=$CONFIG"
echo "MANIFEST=$MANIFEST"
echo "EMBEDDING_MANIFEST=$EMBEDDING_MANIFEST"
echo "EMBEDDING_NAMES=$EMBEDDING_NAMES"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "DEVICE=$DEVICE"
echo "N_PROJECTIONS=$N_PROJECTIONS"
echo "SEED=$SEED"
echo "REPAIR_EVENT_TEXT=${REPAIR_EVENT_TEXT:-from_config}"

"$PYTHON" -m synthetic_time_series_clean.evaluation.sliced_wd "${ARGS[@]}"
