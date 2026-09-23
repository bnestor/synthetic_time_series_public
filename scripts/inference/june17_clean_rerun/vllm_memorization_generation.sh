#!/bin/bash
#SBATCH --job-name=vllm_j17_clean_mem
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "ERROR: missing arguments."
  echo "Usage: bash/sbatch $0 <temperature> <run_name> <run_dir> <output_base> [submit_stamp] [data_override_path]"
  exit 1
fi

export TEMP="$1"
export RUN_NAME="$2"
export RUN_DIR="$3"
export OUTPUT_BASE="$4"
export SUBMIT_STAMP="${5:-$(date +"%Y%m%d_%H%M%S")}"
export DATA_OVERRIDE_PATH="${6:-}"

TEMP_TAG=$(python - "$TEMP" <<'PY'
import sys

temp = float(sys.argv[1])
if abs(temp - 0.0) < 1e-9:
    print("0")
else:
    print(f"{int(round(temp * 10)):02d}")
PY
)
export TEMP_TAG

if [ "$TEMP_TAG" = "0" ]; then
  export SAMPLE_FLAG="--no-sample"
else
  export SAMPLE_FLAG="--do-sample"
fi

module load apptainer

# ==============================================================================
# Runtime environment: same container and venv as the original June17 scripts.
# ==============================================================================

export MAMBA_APPTAINER_ROOT="${MAMBA_APPTAINER_ROOT:-/scratch/<user>/MAMBA_APPTAINER}"
export CONTAINER="${CONTAINER:-$MAMBA_APPTAINER_ROOT/containers/vllm_openai_0.11.0.sif}"
export VENV_DIR="${VENV_DIR:-$MAMBA_APPTAINER_ROOT/venvs/vllm_011_mamba_venv}"

export PROJECT_DIR="${PROJECT_DIR:-/path/to/synthetic_time_series_public}"
export CONFIG="${CONFIG:-$PROJECT_DIR/configs/inference/vllm_generate.yaml}"
export CHECKPOINT="${CHECKPOINT:-}"

# ==============================================================================
# Stage 4A train memorization settings.
# ==============================================================================

export MAX_TRAIN_PATIENTS="${MAX_TRAIN_PATIENTS:-1000}"
export TRAIN_SELECTION_MODE="${TRAIN_SELECTION_MODE:-post_filter_shuffle}"
export RANDOM_SAMPLE_TRAIN="${RANDOM_SAMPLE_TRAIN:-1}"
export MIN_EVENTS="${MIN_EVENTS:-8}"
export FRACTIONS="${FRACTIONS:-0,0.25,0.5,0.75}"

export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
export TOTAL_MAX_LENGTH="${TOTAL_MAX_LENGTH:-16384}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export TOP_P="${TOP_P:-0.9}"
export TOP_K="${TOP_K:-10}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
export SEED="${SEED:-42}"
export DTYPE="${DTYPE:-bfloat16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

FRACTION_TAG=$(python - "$FRACTIONS" <<'PY'
import sys

parts = []
for item in sys.argv[1].split(","):
    item = item.strip()
    if item:
        parts.append(str(int(round(float(item) * 100))))
print("frac" + "_".join(parts))
PY
)
export FRACTION_TAG

TRAIN_SELECTION_TAG=$(python - "$TRAIN_SELECTION_MODE" <<'PY'
import sys

mode = sys.argv[1].strip().lower()
print({
    "head_ids": "headids",
    "random_ids": "randomids",
    "post_filter_shuffle": "postfilter",
}.get(mode, mode.replace("_", "")))
PY
)
export TRAIN_SELECTION_TAG

TOTAL_MAX_LENGTH_NORMALIZED=$(python - "$TOTAL_MAX_LENGTH" <<'PY'
import sys

value = sys.argv[1].strip().lower()
print("none" if value in {"", "none", "null", "-1", "0"} else value)
PY
)
export TOTAL_MAX_LENGTH_NORMALIZED

if [ "$TOTAL_MAX_LENGTH_NORMALIZED" = "none" ]; then
  export LENGTH_TAG="nototal_new${MAX_NEW_TOKENS}"
else
  export LENGTH_TAG="total${TOTAL_MAX_LENGTH_NORMALIZED}"
fi

export OUTPUT_SUBDIR="generation_${SUBMIT_STAMP}_n${MAX_TRAIN_PATIENTS}_${TRAIN_SELECTION_TAG}_min${MIN_EVENTS}_${FRACTION_TAG}_${LENGTH_TAG}_b${BATCH_SIZE}"
export OUTPUT_DIR="$OUTPUT_BASE/$RUN_NAME/temp_${TEMP_TAG}/$OUTPUT_SUBDIR"

if [[ "$OUTPUT_DIR"/ == "$RUN_DIR"/* ]]; then
  echo "ERROR: OUTPUT_DIR is inside RUN_DIR; refusing to risk modifying the model directory."
  echo "RUN_DIR=$RUN_DIR"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 1
fi

export TRAINING_CONFIG_OVERRIDE=""
if [ -n "$DATA_OVERRIDE_PATH" ]; then
  TRAINING_CONFIG_OVERRIDE=$(mktemp -t "june17_clean_data_override_XXXXXX.yaml")
  export TRAINING_CONFIG_OVERRIDE
  sed 's|path: .*/DEATH_PATIENTS/death_sampled_data|path: '"$DATA_OVERRIDE_PATH"'|' \
    "$RUN_DIR/resolved_config.yaml" > "$TRAINING_CONFIG_OVERRIDE"
fi

cleanup_override() {
  if [ -n "${TRAINING_CONFIG_OVERRIDE:-}" ] && [ -f "$TRAINING_CONFIG_OVERRIDE" ]; then
    rm -f "$TRAINING_CONFIG_OVERRIDE"
  fi
}
trap cleanup_override EXIT

# ==============================================================================
# Cache setup, copied from the original June17 scripts.
# ==============================================================================

export APPTAINER_CACHEDIR="$MAMBA_APPTAINER_ROOT/apptainer_cache"
export APPTAINER_TMPDIR="$MAMBA_APPTAINER_ROOT/apptainer_tmp"

export VLLM_CACHE_ROOT="$MAMBA_APPTAINER_ROOT/VLLM_CACHE"
export HF_HOME="$VLLM_CACHE_ROOT/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$VLLM_CACHE_ROOT/torch_home"
export TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE_ROOT/torchinductor_cache"
export TRITON_CACHE_DIR="$VLLM_CACHE_ROOT/triton_cache"
export XDG_CACHE_HOME="$VLLM_CACHE_ROOT/xdg_cache"

export USE_HUB_KERNELS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p \
  "$APPTAINER_CACHEDIR" \
  "$APPTAINER_TMPDIR" \
  "$VLLM_CACHE_ROOT" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$XDG_CACHE_HOME/torch" \
  "$XDG_CACHE_HOME/torch/kernels" \
  "$OUTPUT_DIR"

echo "===== HOST CHECK ====="
hostname
date
echo "TEMP=$TEMP"
echo "TEMP_TAG=$TEMP_TAG"
echo "SAMPLE_FLAG=$SAMPLE_FLAG"
echo "RUN_NAME=$RUN_NAME"
echo "RUN_DIR=$RUN_DIR"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_BASE=$OUTPUT_BASE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SUBMIT_STAMP=$SUBMIT_STAMP"
echo "CONTAINER=$CONTAINER"
echo "VENV_DIR=$VENV_DIR"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONFIG=$CONFIG"
echo "DATA_SPLIT_USED=train"
echo "DATA_OVERRIDE_PATH=$DATA_OVERRIDE_PATH"
echo "TRAINING_CONFIG_OVERRIDE=$TRAINING_CONFIG_OVERRIDE"
echo "MAX_TRAIN_PATIENTS=$MAX_TRAIN_PATIENTS"
echo "TRAIN_SELECTION_MODE=$TRAIN_SELECTION_MODE"
echo "RANDOM_SAMPLE_TRAIN=$RANDOM_SAMPLE_TRAIN"
echo "MIN_EVENTS=$MIN_EVENTS"
echo "FRACTIONS=$FRACTIONS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "TOTAL_MAX_LENGTH=$TOTAL_MAX_LENGTH"
echo "TOTAL_MAX_LENGTH_NORMALIZED=$TOTAL_MAX_LENGTH_NORMALIZED"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "LENGTH_TAG=$LENGTH_TAG"
echo "TOP_P=$TOP_P"
echo "TOP_K=$TOP_K"
echo "REPETITION_PENALTY=$REPETITION_PENALTY"
echo "SEED=$SEED"
echo "DTYPE=$DTYPE"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
if [ -n "$TRAINING_CONFIG_OVERRIDE" ]; then
  echo "Effective data path override:"
  grep 'path:' "$TRAINING_CONFIG_OVERRIDE" | head -1 || true
fi

test -f "$CONTAINER"
test -d "$VENV_DIR"
test -d "$PROJECT_DIR"
test -f "$CONFIG"
test -d "$RUN_DIR"
test -f "$RUN_DIR/resolved_config.yaml"
if [ -n "$TRAINING_CONFIG_OVERRIDE" ]; then
  test -f "$TRAINING_CONFIG_OVERRIDE"
fi
if [ -n "$CHECKPOINT" ]; then
  test -d "$RUN_DIR/$CHECKPOINT"
fi

echo "===== NVIDIA CHECK ====="
nvidia-smi || true

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /project:/project \
  --bind /datasets:/datasets \
  --bind /tmp:/tmp \
  "$CONTAINER" \
  bash --norc -c "
set -eo pipefail

source \"$VENV_DIR/bin/activate\"

export PROJECT_DIR=\"$PROJECT_DIR\"
export PROJECT_ROOT=\"$PROJECT_DIR\"
export PYTHONPATH=\"$PROJECT_DIR:\${PYTHONPATH:-}\"

export VLLM_CACHE_ROOT=\"$VLLM_CACHE_ROOT\"
export HF_HOME=\"$HF_HOME\"
export HF_HUB_CACHE=\"$HF_HUB_CACHE\"
export HUGGINGFACE_HUB_CACHE=\"$HUGGINGFACE_HUB_CACHE\"
export TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\"
export TORCH_HOME=\"$TORCH_HOME\"
export TORCHINDUCTOR_CACHE_DIR=\"$TORCHINDUCTOR_CACHE_DIR\"
export TRITON_CACHE_DIR=\"$TRITON_CACHE_DIR\"
export XDG_CACHE_HOME=\"$XDG_CACHE_HOME\"
export USE_HUB_KERNELS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=\"$OMP_NUM_THREADS\"

cd \"$PROJECT_DIR\"

echo '===== PYTHON CHECK ====='
which python
python --version

python - <<'PY'
import torch
import vllm
import transformers
import polars

print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
print('vllm:', vllm.__version__)
print('transformers:', transformers.__version__)
print('polars:', polars.__version__)
PY

EXTRA_ARGS=()
if [ -n \"$CHECKPOINT\" ]; then
  EXTRA_ARGS+=(--checkpoint \"$CHECKPOINT\")
fi
if [ \"$RANDOM_SAMPLE_TRAIN\" = \"1\" ]; then
  EXTRA_ARGS+=(--random-sample-train)
else
  EXTRA_ARGS+=(--no-random-sample-train)
fi
if [ \"$TOTAL_MAX_LENGTH_NORMALIZED\" != \"none\" ]; then
  EXTRA_ARGS+=(--total-max-length \"$TOTAL_MAX_LENGTH_NORMALIZED\")
fi
if [ -n \"$TRAINING_CONFIG_OVERRIDE\" ]; then
  EXTRA_ARGS+=(--training-config \"$TRAINING_CONFIG_OVERRIDE\")
fi

echo '===== START CLEAN vLLM JUNE17 TRAIN MEMORIZATION GENERATION ====='

python -m synthetic_time_series_clean.inference.generate_memorization \
  --config \"$CONFIG\" \
  --run-name \"$RUN_NAME\" \
  --temp-tag \"$TEMP_TAG\" \
  --run-dir \"$RUN_DIR\" \
  --output-dir \"$OUTPUT_DIR\" \
  --max-train-patients \"$MAX_TRAIN_PATIENTS\" \
  --train-selection-mode \"$TRAIN_SELECTION_MODE\" \
  --min-events \"$MIN_EVENTS\" \
  --fractions \"$FRACTIONS\" \
  --batch-size \"$BATCH_SIZE\" \
  --max-new-tokens \"$MAX_NEW_TOKENS\" \
  --temperature \"$TEMP\" \
  --top-p \"$TOP_P\" \
  --top-k \"$TOP_K\" \
  --repetition-penalty \"$REPETITION_PENALTY\" \
  --seed \"$SEED\" \
  --dtype \"$DTYPE\" \
  --max-model-len \"$MAX_MODEL_LEN\" \
  --gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\" \
  --trust-remote-code \
  \"$SAMPLE_FLAG\" \
  \"\${EXTRA_ARGS[@]}\"

echo '===== DONE CLEAN vLLM JUNE17 TRAIN MEMORIZATION GENERATION ====='
"

echo "===== JOB COMPLETE ====="
date
echo "Output dir: $OUTPUT_DIR"
echo "Selected train rows: $OUTPUT_DIR/selected_train_rows.jsonl"
echo "Phase candidates: $OUTPUT_DIR/phases/<fraction>/train_memorization_candidates.jsonl"
echo "Phase generations: $OUTPUT_DIR/phases/<fraction>/train_memorization_generations.jsonl"
echo "Summary: $OUTPUT_DIR/stage_4a_summary.json"
