#!/bin/bash
#SBATCH --job-name=aug11_wd_base
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=kn001
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

BASELINE_VARIANT="${BASELINE_VARIANT:?BASELINE_VARIANT is required, e.g. full_16384}"
require_file "$EMBEDDING_MANIFEST"

variant_dir="$MANIFEST_DIR/baselines/$BASELINE_VARIANT"
variant_manifest="$variant_dir/baseline_datasets_manifest.csv"
require_file "$variant_manifest"

OUTPUT_ROOT="$WD_RESULTS_ROOT/baselines/$BASELINE_VARIANT/slurm_${SLURM_JOB_ID:-local}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

export PREPARED_ROOT="$variant_dir"
export MANIFEST="$variant_manifest"
export OUTPUT_ROOT
export EMBEDDING_MANIFEST
export EMBEDDING_NAMES
export TEXT_VIEWS="$BASELINE_TEXT_VIEWS"
export DEVICE
export N_PROJECTIONS
export SEED
export MAX_ROWS
export SAVE_EMBEDDINGS
export PYTHON="python"
export BASELINE_WD_WRAPPER

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

module load apptainer

export MAMBA_APPTAINER_ROOT="${MAMBA_APPTAINER_ROOT:-/scratch/user/MAMBA_APPTAINER}"
export CONTAINER="${CONTAINER:-$MAMBA_APPTAINER_ROOT/containers/vllm_openai_0.11.0.sif}"
export VENV_DIR="${VENV_DIR:-$MAMBA_APPTAINER_ROOT/venvs/vllm_011_mamba_venv}"
BASELINE_WD_PROJECT_DIR=$(cd -- "$(dirname -- "$BASELINE_WD_WRAPPER")/.." && pwd)
export BASELINE_WD_PROJECT_DIR

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

echo "===== AUG11 BASELINE WD ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "BASELINE_VARIANT=$BASELINE_VARIANT"
echo "PREPARED_ROOT=$PREPARED_ROOT"
echo "MANIFEST=$MANIFEST"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "EMBEDDING_MANIFEST=$EMBEDDING_MANIFEST"
echo "EMBEDDING_NAMES=$EMBEDDING_NAMES"
echo "TEXT_VIEWS=$TEXT_VIEWS"
echo "N_PROJECTIONS=$N_PROJECTIONS"
echo "SEED=$SEED"
echo "CONTAINER=$CONTAINER"
echo "VENV_DIR=$VENV_DIR"
echo "BASELINE_WD_PROJECT_DIR=$BASELINE_WD_PROJECT_DIR"

test -f "$CONTAINER"
test -d "$VENV_DIR"
test -f "$BASELINE_WD_WRAPPER"
nvidia-smi || true

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /project:/project \
  --bind /datasets:/datasets \
  "$CONTAINER" \
  bash --norc -lc '
set -euo pipefail
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$BASELINE_WD_PROJECT_DIR:${PYTHONPATH:-}"
python --version
bash "$BASELINE_WD_WRAPPER"
'
