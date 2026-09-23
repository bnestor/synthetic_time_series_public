#!/bin/bash
#SBATCH --job-name=aug11_ppl_base
#SBATCH --partition=partition
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=1-12:00:00
#SBATCH --exclude=kn001,kn050
#SBATCH --output=/%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

DEFAULT_SCRIPT_DIR="/path/to/your/synthetic_time_series_public/scripts/evaluation/EVAL_PPL_WD"
SCRIPT_DIR="${EVAL_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$DEFAULT_SCRIPT_DIR}}"
if [ ! -f "$SCRIPT_DIR/config.sh" ]; then
  SCRIPT_DIR="$DEFAULT_SCRIPT_DIR"
fi
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

require_file "$BASELINE_PPL_MANIFEST"

BASELINE_PPL_RUN_NAME="${BASELINE_PPL_RUN_NAME:?BASELINE_PPL_RUN_NAME is required, e.g. real_test, real_train, wiki_full_16384, or wiki_uniform16_16384}"
case "$BASELINE_PPL_RUN_NAME" in
  real_test|real_train|wiki_full_16384|wiki_uniform16_16384)
    ;;
  *)
    echo "ERROR: unsupported BASELINE_PPL_RUN_NAME: $BASELINE_PPL_RUN_NAME" >&2
    exit 1
    ;;
esac

OUTPUT_ROOT="$PPL_RESULTS_ROOT/baselines/full_text/$SYNTHETIC_RUN_NAME/$BASELINE_PPL_RUN_NAME/baselines__${SYNTHETIC_RUN_NAME}__${BASELINE_PPL_RUN_NAME}__slurm_${SLURM_JOB_ID:-local}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

export TASK_TYPE="synthetic"
export MANIFEST="$BASELINE_PPL_MANIFEST"
export OUTPUT_ROOT
export RUN_DIR="$ORIGINAL_RUN_DIR"
export CONTEXT_FIELD="__empty_context__"
export TARGET_FIELD="static_event"
export MAX_LENGTH
export BATCH_SIZE
export DEVICE
export DTYPE
export MAX_ROWS
export EXPECTED_INPUT_COUNT="1"
export FILTERS="run_name=$BASELINE_PPL_RUN_NAME"
export DRY_RUN
export PPL_WRAPPER
export PYTHON="python"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

module load apptainer

export MAMBA_APPTAINER_ROOT="${MAMBA_APPTAINER_ROOT:-/scratch/user/MAMBA_APPTAINER}"
export CONTAINER="${CONTAINER:-$MAMBA_APPTAINER_ROOT/containers/vllm_openai_0.11.0.sif}"
export VENV_DIR="${VENV_DIR:-$MAMBA_APPTAINER_ROOT/venvs/vllm_011_mamba_venv}"
PPL_PROJECT_DIR=$(cd -- "$(dirname -- "$PPL_WRAPPER")" && pwd)
export PPL_PROJECT_DIR

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
  "$XDG_CACHE_HOME"

echo "===== AUG11 BASELINE PPL ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "BASELINE_PPL_RUN_NAME=$BASELINE_PPL_RUN_NAME"
echo "MANIFEST=$MANIFEST"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "CONTEXT_FIELD=$CONTEXT_FIELD"
echo "TARGET_FIELD=$TARGET_FIELD"
echo "MAX_LENGTH=$MAX_LENGTH"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EXPECTED_INPUT_COUNT=$EXPECTED_INPUT_COUNT"
echo "FILTERS=$FILTERS"
echo "CONTAINER=$CONTAINER"
echo "VENV_DIR=$VENV_DIR"
echo "PPL_PROJECT_DIR=$PPL_PROJECT_DIR"
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

test -f "$CONTAINER"
test -d "$VENV_DIR"
test -f "$PPL_WRAPPER"
nvidia-smi || true

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /project:/project \
  --bind /datasets:/datasets \
  "$CONTAINER" \
  bash --norc -lc '
set -euo pipefail
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PPL_PROJECT_DIR:${PYTHONPATH:-}"
python --version
echo "Container CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Container SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
python - <<'"'"'PY'"'"'
import sys
import torch

print("torch version:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
print("torch cuda device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available inside the Apptainer container.", file=sys.stderr)
    sys.exit(42)
print("torch cuda device 0:", torch.cuda.get_device_name(0))
PY
bash "$PPL_WRAPPER"
'
