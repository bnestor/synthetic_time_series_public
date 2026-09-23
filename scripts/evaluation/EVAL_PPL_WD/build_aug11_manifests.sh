#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)


sample_file_from_dir() {
  local temp_dir="$1"
  require_dir "$temp_dir"
  shopt -s nullglob
  local matches=("$temp_dir"/*_real_and_synthetic_data.jsonl)
  shopt -u nullglob
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one *_real_and_synthetic_data.jsonl under $temp_dir, found ${#matches[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

generation_dir_for_temp() {
  local temp_tag="$1"
  local temp_root="$MEMORIZATION_ROOT/$temp_tag"
  require_dir "$temp_root"
  shopt -s nullglob
  local matches=("$temp_root"/generation_*)
  shopt -u nullglob
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one generation_* dir under $temp_root, found ${#matches[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

mkdir -p "$MANIFEST_DIR" "$PPL_RESULTS_ROOT" "$WD_RESULTS_ROOT" "$LOG_DIR"

synthetic_tmp=$(mktemp "${SYNTHETIC_MANIFEST}.tmp.XXXXXX")
memorization_tmp=$(mktemp "${MEMORIZATION_MANIFEST}.tmp.XXXXXX")
embedding_tmp=$(mktemp "${EMBEDDING_MANIFEST}.tmp.XXXXXX")
cleanup() {
  rm -f "$synthetic_tmp" "$memorization_tmp" "$embedding_tmp"
}
trap cleanup EXIT

{
  echo "run_name,source_label,temp_tag,temperature,source_path"
  for temp_tag in $MANIFEST_TEMP_TAGS; do
    source_path=$(sample_file_from_dir "$SYNTHETIC_ROOT/${temp_tag}_20260811_163116")
    printf '%s,%s,%s,%s,%s\n' \
      "$SYNTHETIC_RUN_NAME" \
      "$SYNTHETIC_SOURCE_LABEL" \
      "$temp_tag" \
      "$(temp_value "$temp_tag")" \
      "$source_path"
  done
} > "$synthetic_tmp"

{
  echo "run_name,source_label,temp_tag,temperature,fraction_tag,fraction,source_path"
  for temp_tag in $MANIFEST_TEMP_TAGS; do
    generation_dir=$(generation_dir_for_temp "$temp_tag")
    for fraction_tag in $FRACTION_TAGS; do
      source_path="$generation_dir/phases/$fraction_tag/train_memorization_generations.jsonl"
      require_file "$source_path"
      printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$MEMORIZATION_RUN_NAME" \
        "$MEMORIZATION_RUN_NAME" \
        "$temp_tag" \
        "$(temp_value "$temp_tag")" \
        "$fraction_tag" \
        "$(fraction_value "$fraction_tag")" \
        "$source_path"
    done
  done
} > "$memorization_tmp"

{
  echo "embedding_name,model_path,max_length,batch_size"
  printf 'bge_m3,%s,%s,%s\n' "$BGEM3_MODEL_PATH" "$EMBEDDING_MAX_LENGTH" "$EMBEDDING_BATCH_SIZE"
} > "$embedding_tmp"

mv "$synthetic_tmp" "$SYNTHETIC_MANIFEST"
mv "$memorization_tmp" "$MEMORIZATION_MANIFEST"
mv "$embedding_tmp" "$EMBEDDING_MANIFEST"
trap - EXIT

echo "Wrote synthetic manifest: $SYNTHETIC_MANIFEST"
echo "Wrote memorization manifest: $MEMORIZATION_MANIFEST"
echo "Wrote BGE-M3 embedding manifest: $EMBEDDING_MANIFEST"
echo "PPL results root: $PPL_RESULTS_ROOT"
echo "WD results root: $WD_RESULTS_ROOT"
echo "Slurm log dir: $LOG_DIR"
