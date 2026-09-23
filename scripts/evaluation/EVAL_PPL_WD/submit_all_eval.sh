#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
export EVAL_SCRIPT_DIR="$SCRIPT_DIR"

SUBMIT_SET="${SUBMIT_SET:-all}"
SUBMIT_DRY_RUN="${SUBMIT_DRY_RUN:-0}"

bash "$SCRIPT_DIR/build_aug11_manifests.sh"
bash "$SCRIPT_DIR/build_baseline_manifests.sh"

should_submit() {
  local job="$1"
  case "$SUBMIT_SET" in
    all) return 0 ;;
    ppl) [[ "$job" == ppl_* ]]; return ;;
    wd) [[ "$job" == wd_* ]]; return ;;
    synthetic) [[ "$job" == *_synthetic ]]; return ;;
    memorization) [[ "$job" == *_memorization ]]; return ;;
    baseline|baselines) [[ "$job" == *_baselines ]]; return ;;
    *) [[ ",$SUBMIT_SET," == *",$job,"* ]]; return ;;
  esac
}

submit_job() {
  local job="$1"
  local script="$2"
  local temp_tag="$3"
  if ! should_submit "$job"; then
    return 0
  fi
  local cmd=(sbatch --export=ALL,EVAL_SCRIPT_DIR="$SCRIPT_DIR",TEMP_TAG="$temp_tag" "$SCRIPT_DIR/$script")
  if [ "$SUBMIT_DRY_RUN" = "1" ]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

echo "===== SUBMIT AUG11 PPL/WD EVAL ====="
echo "SUBMIT_SET=$SUBMIT_SET"
echo "SUBMIT_TEMP_TAGS=$SUBMIT_TEMP_TAGS"
echo "MANIFEST_TEMP_TAGS=$MANIFEST_TEMP_TAGS"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "PPL_RESULTS_ROOT=$PPL_RESULTS_ROOT"
echo "WD_RESULTS_ROOT=$WD_RESULTS_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "EMBEDDING_NAMES=bge_m3"
echo "N_PROJECTIONS=$N_PROJECTIONS"
echo "BASELINE_VARIANTS=$BASELINE_VARIANTS"
echo "BASELINE_PPL_RUN_NAMES=$BASELINE_PPL_RUN_NAMES"

for temp_tag in $SUBMIT_TEMP_TAGS; do
  submit_job "ppl_synthetic" "sbatch_ppl_synthetic.sbatch" "$temp_tag"
  submit_job "ppl_memorization" "sbatch_ppl_memorization.sbatch" "$temp_tag"
  submit_job "wd_synthetic" "sbatch_wd_synthetic.sbatch" "$temp_tag"
  submit_job "wd_memorization" "sbatch_wd_memorization.sbatch" "$temp_tag"
done

should_submit_baseline_ppl() {
  local baseline_ppl_run_name="$1"
  should_submit "ppl_baselines" || should_submit "ppl_baseline_$baseline_ppl_run_name"
}

for baseline_ppl_run_name in $BASELINE_PPL_RUN_NAMES; do
  if ! should_submit_baseline_ppl "$baseline_ppl_run_name"; then
    continue
  fi
  job_name="aug11_ppl_${baseline_ppl_run_name}"
  if [ "$SUBMIT_DRY_RUN" = "1" ]; then
    printf 'DRY_RUN: %q %q %q %q\n' \
      sbatch \
      "--job-name=$job_name" \
      "--export=ALL,EVAL_SCRIPT_DIR=$SCRIPT_DIR,BASELINE_PPL_RUN_NAME=$baseline_ppl_run_name" \
      "$SCRIPT_DIR/sbatch_ppl_baselines.sbatch"
  else
    sbatch \
      --job-name="$job_name" \
      --export=ALL,EVAL_SCRIPT_DIR="$SCRIPT_DIR",BASELINE_PPL_RUN_NAME="$baseline_ppl_run_name" \
      "$SCRIPT_DIR/sbatch_ppl_baselines.sbatch"
  fi
done

if should_submit "wd_baselines"; then
  for variant in $BASELINE_VARIANTS; do
    if [ "$SUBMIT_DRY_RUN" = "1" ]; then
      printf 'DRY_RUN: %q %q %q\n' sbatch "--export=ALL,EVAL_SCRIPT_DIR=$SCRIPT_DIR,BASELINE_VARIANT=$variant" "$SCRIPT_DIR/sbatch_wd_baseline.sbatch"
    else
      sbatch --export=ALL,EVAL_SCRIPT_DIR="$SCRIPT_DIR",BASELINE_VARIANT="$variant" "$SCRIPT_DIR/sbatch_wd_baseline.sbatch"
    fi
  done
fi
