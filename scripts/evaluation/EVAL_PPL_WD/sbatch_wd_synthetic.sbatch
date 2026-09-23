#!/bin/bash
#SBATCH --job-name=aug11_wd_syn
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
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

TEMP_TAG="${TEMP_TAG:?TEMP_TAG is required, e.g. temp_08}"
require_file "$SYNTHETIC_MANIFEST"
require_file "$EMBEDDING_MANIFEST"

OUTPUT_ROOT="$WD_RESULTS_ROOT/slices/synthetic/$SYNTHETIC_RUN_NAME/synthetic__${SYNTHETIC_RUN_NAME}__${TEMP_TAG}__bge_m3__slurm_${SLURM_JOB_ID:-local}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

export HF_HOME="${HF_HOME:-/scratch/${USER:-<user>}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export USE_HUB_KERNELS="${USE_HUB_KERNELS:-0}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

EMBEDDING_VENV="${EMBEDDING_VENV:-/scratch/${USER:-<user>}/SDG_PROJECT_DATA/embedding_env}"
if [ "${SKIP_VENV:-0}" != "1" ] && [ -d "$EMBEDDING_VENV" ]; then
  # shellcheck disable=SC1091
  source "$EMBEDDING_VENV/bin/activate"
fi

export TASK_TYPE="synthetic"
export MANIFEST="$SYNTHETIC_MANIFEST"
export OUTPUT_ROOT
export WORK_DIR="$ORGANIZED_ROOT/vllm_inf_eval"
export EMBEDDING_MANIFEST
export EMBEDDING_NAMES="bge_m3"
export DEVICE
export N_PROJECTIONS
export SEED
export MAX_ROWS
export EXPECTED_INPUT_COUNT="1"
export FILTERS="run_name=$SYNTHETIC_RUN_NAME temp_tag=$TEMP_TAG"
export SAVE_EMBEDDINGS
export DRY_RUN
export PYTHON="${PYTHON:-python}"

echo "===== AUG11 SYNTHETIC WD / BGE-M3 ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "TEMP_TAG=$TEMP_TAG"
echo "MANIFEST=$MANIFEST"
echo "EMBEDDING_MANIFEST=$EMBEDDING_MANIFEST"
echo "EMBEDDING_NAMES=$EMBEDDING_NAMES"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "FILTERS=$FILTERS"
echo "N_PROJECTIONS=$N_PROJECTIONS"
echo "SEED=$SEED"
echo "SAVE_EMBEDDINGS=$SAVE_EMBEDDINGS"
echo "TIME_REQUEST=1-00:00:00"
echo "PYTHON=$(command -v "$PYTHON" || true)"

test -f "$WD_WRAPPER"
test -f "$EMBEDDING_MANIFEST"
nvidia-smi || true

bash "$WD_WRAPPER"
