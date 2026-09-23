#!/bin/bash
#SBATCH --job-name=vllm_j17_clean_gen
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --output=/%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "ERROR: missing temperature argument."
  echo "Usage: bash/sbatch $0 <temperature> [run_dir] [output_dir]"
  exit 1
fi

export TEMP="$1"
export RUN_DIR="${2:-${RUN_DIR:-/result/path/2026-mm-dd/hh-mm-ss_b4_g8_3691289/unnormalized/0}}"

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

RUN_STAMP=$(date +"%Y%m%d_%H%M%S")
export RUN_STAMP

export RESULTS_ROOT="${RESULTS_ROOT:-/path/to/results}"
export OUTPUT_DIR="${3:-${OUTPUT_DIR:-$RESULTS_ROOT/JUNE17_GENERATE/temp_${TEMP_TAG}_${RUN_STAMP}}}"
export SAMPLES_FILENAME="${SAMPLES_FILENAME:-temp_${TEMP_TAG}_${RUN_STAMP}_real_and_synthetic_data.jsonl}"

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
# Stage 1-like generation settings, copied from the original June17 submit jobs.
# ==============================================================================

export EVAL_DATASET="${EVAL_DATASET:-held_out}"
export MAX_EVAL_PATIENTS="${MAX_EVAL_PATIENTS:-3000}"
export RANDOM_SAMPLE_EVAL="${RANDOM_SAMPLE_EVAL:-1}"
export MAX_TRAIN_PATIENTS="${MAX_TRAIN_PATIENTS:-}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export SYNTHETIC_SAMPLES_PER_EVAL="${SYNTHETIC_SAMPLES_PER_EVAL:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export TOP_P="${TOP_P:-0.9}"
export TOP_K="${TOP_K:-10}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
export SEED="${SEED:-42}"
export DTYPE="${DTYPE:-bfloat16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

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

if [[ "$OUTPUT_DIR"/ == "$RUN_DIR"/* ]]; then
  echo "ERROR: OUTPUT_DIR is inside RUN_DIR; refusing to risk modifying the model directory."
  echo "RUN_DIR=$RUN_DIR"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 1
fi

echo "===== HOST CHECK ====="
hostname
date
echo "TEMP=$TEMP"
echo "TEMP_TAG=$TEMP_TAG"
echo "SAMPLE_FLAG=$SAMPLE_FLAG"
echo "RUN_STAMP=$RUN_STAMP"
echo "SAMPLES_FILENAME=$SAMPLES_FILENAME"
echo "CONTAINER=$CONTAINER"
echo "VENV_DIR=$VENV_DIR"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONFIG=$CONFIG"
echo "RUN_DIR=$RUN_DIR"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "EVAL_DATASET=$EVAL_DATASET"
echo "MAX_EVAL_PATIENTS=$MAX_EVAL_PATIENTS"
echo "RANDOM_SAMPLE_EVAL=$RANDOM_SAMPLE_EVAL"
echo "MAX_TRAIN_PATIENTS=$MAX_TRAIN_PATIENTS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "SYNTHETIC_SAMPLES_PER_EVAL=$SYNTHETIC_SAMPLES_PER_EVAL"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "TOP_P=$TOP_P"
echo "TOP_K=$TOP_K"
echo "REPETITION_PENALTY=$REPETITION_PENALTY"
echo "SEED=$SEED"
echo "DTYPE=$DTYPE"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"

test -f "$CONTAINER"
test -d "$VENV_DIR"
test -d "$PROJECT_DIR"
test -f "$CONFIG"
test -d "$RUN_DIR"
test -f "$RUN_DIR/resolved_config.yaml"
if [ -n "$CHECKPOINT" ]; then
  test -d "$RUN_DIR/$CHECKPOINT"
fi

echo "===== NVIDIA CHECK ====="
nvidia-smi || true

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /project:/project \
  --bind /datasets:/datasets \
  "$CONTAINER" \
  bash --norc -c "
set -eo pipefail

source \"$VENV_DIR/bin/activate\"

export PROJECT_DIR=\"$PROJECT_DIR\"
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
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
print('vllm:', vllm.__version__)
PY

EXTRA_ARGS=()
if [ -n \"$CHECKPOINT\" ]; then
  EXTRA_ARGS+=(--checkpoint \"$CHECKPOINT\")
fi
if [ -n \"$MAX_EVAL_PATIENTS\" ]; then
  EXTRA_ARGS+=(--max-eval-patients \"$MAX_EVAL_PATIENTS\")
fi
if [ \"$RANDOM_SAMPLE_EVAL\" = \"1\" ]; then
  EXTRA_ARGS+=(--random-sample-eval)
fi
if [ -n \"$MAX_TRAIN_PATIENTS\" ]; then
  EXTRA_ARGS+=(--max-train-patients \"$MAX_TRAIN_PATIENTS\")
fi
if [ -n \"$MAX_MODEL_LEN\" ]; then
  EXTRA_ARGS+=(--max-model-len \"$MAX_MODEL_LEN\")
fi

echo '===== START CLEAN vLLM JUNE17 SYNTHETIC GENERATION ====='

python -m synthetic_time_series_clean.inference.generate_vllm \
  --config \"$CONFIG\" \
  --run-dir \"$RUN_DIR\" \
  --output-dir \"$OUTPUT_DIR\" \
  --samples-filename \"$SAMPLES_FILENAME\" \
  --eval-dataset \"$EVAL_DATASET\" \
  --batch-size \"$BATCH_SIZE\" \
  --synthetic-samples-per-eval \"$SYNTHETIC_SAMPLES_PER_EVAL\" \
  --max-new-tokens \"$MAX_NEW_TOKENS\" \
  --temperature \"$TEMP\" \
  --top-p \"$TOP_P\" \
  --top-k \"$TOP_K\" \
  --repetition-penalty \"$REPETITION_PENALTY\" \
  --seed \"$SEED\" \
  --dtype \"$DTYPE\" \
  --gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\" \
  --trust-remote-code \
  \"$SAMPLE_FLAG\" \
  \"\${EXTRA_ARGS[@]}\"

echo '===== DONE CLEAN vLLM JUNE17 SYNTHETIC GENERATION ====='
"

echo "===== JOB COMPLETE ====="
date
echo "Temperature: $TEMP"
echo "Output JSONL: $OUTPUT_DIR/$SAMPLES_FILENAME"
