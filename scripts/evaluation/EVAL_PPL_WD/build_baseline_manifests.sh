#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

manifest_dir_for_variant() {
  local variant="$1"
  printf '%s\n' "$MANIFEST_DIR/baselines/$variant"
}

wiki_path_for_variant() {
  local variant="$1"
  case "$variant" in
    full_16384)
      printf '%s\n' "$BASELINE_FULL_WIKI_ROOT/random_wikitext103_validation.jsonl"
      ;;
    uniform16_16384)
      printf '%s\n' "$BASELINE_UNIFORM_WIKI_ROOT/random_wikitext103_validation.jsonl"
      ;;
    *)
      echo "ERROR: unsupported BASELINE_VARIANTS entry: $variant" >&2
      exit 1
      ;;
  esac
}

wiki_role_for_variant() {
  local variant="$1"
  case "$variant" in
    full_16384)
      printf '%s\n' "unrelated_random_text_full_16384"
      ;;
    uniform16_16384)
      printf '%s\n' "unrelated_random_text_uniform16_16384"
      ;;
    *)
      echo "ERROR: unsupported BASELINE_VARIANTS entry: $variant" >&2
      exit 1
      ;;
  esac
}

real_test_path="$BASELINE_REAL_ROOT/real_test.jsonl"
real_train_path="$BASELINE_REAL_ROOT/real_train.jsonl"
require_file "$real_test_path"
require_file "$real_train_path"
mkdir -p "$MANIFEST_DIR"

baseline_ppl_tmp=$(mktemp "${BASELINE_PPL_MANIFEST}.tmp.XXXXXX")
baseline_prepared_tmp=$(mktemp "${BASELINE_PREPARED_ROOTS_MANIFEST}.tmp.XXXXXX")
cleanup() {
  rm -f "$baseline_ppl_tmp" "$baseline_prepared_tmp"
}
trap cleanup EXIT

{
  echo "run_name,source_label,temp_tag,temperature,source_path"
  printf '%s,%s,%s,%s,%s\n' \
    "real_test" \
    "real_test_reference" \
    "baseline" \
    "" \
    "$real_test_path"
  printf '%s,%s,%s,%s,%s\n' \
    "real_train" \
    "real_train_reference" \
    "baseline" \
    "" \
    "$real_train_path"
} > "$baseline_ppl_tmp"

{
  echo "variant,prepared_root,manifest,wiki_jsonl,count"
  for variant in $BASELINE_VARIANTS; do
    wiki_path="$(wiki_path_for_variant "$variant")"
    require_file "$wiki_path"
    variant_dir="$(manifest_dir_for_variant "$variant")"
    mkdir -p "$variant_dir"
    variant_manifest="$variant_dir/baseline_datasets_manifest.csv"
    variant_manifest_tmp=$(mktemp "${variant_manifest}.tmp.XXXXXX")
    wiki_count="$(wc -l < "$wiki_path")"
    real_test_count="$(wc -l < "$real_test_path")"
    real_train_count="$(wc -l < "$real_train_path")"
    wiki_role="$(wiki_role_for_variant "$variant")"

    {
      echo "dataset,dataset_role,path,count"
      printf '%s,%s,%s,%s\n' "real_test" "real_test_reference" "$real_test_path" "$real_test_count"
      printf '%s,%s,%s,%s\n' "real_train" "real_train_reference" "$real_train_path" "$real_train_count"
      printf '%s,%s,%s,%s\n' "random_wikitext103_validation" "$wiki_role" "$wiki_path" "$wiki_count"
    } > "$variant_manifest_tmp"
    mv "$variant_manifest_tmp" "$variant_manifest"

    printf '%s,%s,%s,%s,%s\n' "$variant" "$variant_dir" "$variant_manifest" "$wiki_path" "$wiki_count"
    printf '%s,%s,%s,%s,%s\n' \
      "wiki_${variant}" \
      "$wiki_role" \
      "$variant" \
      "" \
      "$wiki_path" >> "$baseline_ppl_tmp"
  done
} > "$baseline_prepared_tmp"

mv "$baseline_ppl_tmp" "$BASELINE_PPL_MANIFEST"
mv "$baseline_prepared_tmp" "$BASELINE_PREPARED_ROOTS_MANIFEST"
trap - EXIT

echo "Wrote baseline PPL manifest: $BASELINE_PPL_MANIFEST"
echo "Wrote baseline prepared roots manifest: $BASELINE_PREPARED_ROOTS_MANIFEST"
echo "Baseline variants: $BASELINE_VARIANTS"
