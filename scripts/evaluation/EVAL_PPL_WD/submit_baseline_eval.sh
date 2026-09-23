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
    all|baseline|baselines) return 0 ;;
    ppl|ppl_baselines) [[ "$job" == "ppl_baselines" ]]; return ;;
    wd|wd_baselines) [[ "$job" == "wd_baselines" ]]; return ;;
    *) [[ ",$SUBMIT_SET," == *",$job,"* ]]; return ;;
  esac
}

submit_cmd() {
  if [ "$SUBMIT_DRY_RUN" = "1" ]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

echo "===== SUBMIT AUG11 BASELINE PPL/WD EVAL ====="
echo "SUBMIT_SET=$SUBMIT_SET"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "PPL_RESULTS_ROOT=$PPL_RESULTS_ROOT"
echo "WD_RESULTS_ROOT=$WD_RESULTS_ROOT"
echo "BASELINE_VARIANTS=$BASELINE_VARIANTS"
echo "BASELINE_PPL_RUN_NAMES=$BASELINE_PPL_RUN_NAMES"
echo "BASELINE_PPL_MANIFEST=$BASELINE_PPL_MANIFEST"
echo "BASELINE_PREPARED_ROOTS_MANIFEST=$BASELINE_PREPARED_ROOTS_MANIFEST"

should_submit_baseline_ppl() {
  local baseline_ppl_run_name="$1"
  should_submit "ppl_baselines" || should_submit "ppl_baseline_$baseline_ppl_run_name"
}

for baseline_ppl_run_name in $BASELINE_PPL_RUN_NAMES; do
  if should_submit_baseline_ppl "$baseline_ppl_run_name"; then
    submit_cmd \
      sbatch \
      --job-name="aug11_ppl_${baseline_ppl_run_name}" \
      --export=ALL,EVAL_SCRIPT_DIR="$SCRIPT_DIR",BASELINE_PPL_RUN_NAME="$baseline_ppl_run_name" \
      "$SCRIPT_DIR/sbatch_ppl_baselines.sbatch"
  fi
done

if should_submit "wd_baselines"; then
  for variant in $BASELINE_VARIANTS; do
    submit_cmd sbatch --export=ALL,EVAL_SCRIPT_DIR="$SCRIPT_DIR",BASELINE_VARIANT="$variant" "$SCRIPT_DIR/sbatch_wd_baseline.sbatch"
  done
fi
