# AUG11 PPL and WD Evaluation

This directory provides batch evaluation scripts for the AUG11 clean rerun. It covers two metric families:

```text
PPL     -> generated continuation perplexity
WD      -> BGE-M3 embedding + Sliced Wasserstein Distance
```

Working directory:

```bash
cd /path/to/your/synthetic_time_series_public/scripts/evaluation/EVAL_PPL_WD
```


## Manifests

Build AUG11 synthetic, memorization, and embedding manifests:

```bash
bash build_aug11_manifests.sh
```

Build baseline PPL and baseline WD manifests:

```bash
bash build_baseline_manifests.sh
```

Main manifest files:

```text
manifests/aug11_synthetic_manifest.csv
manifests/aug11_memorization_manifest.csv
manifests/embedding_manifest_bgem3.csv
manifests/baseline_ppl_manifest.csv
manifests/baseline_prepared_roots_manifest.csv
```

Default input counts:

```text
aug11_synthetic_manifest.csv:       11 inputs
aug11_memorization_manifest.csv:    44 inputs
embedding_manifest_bgem3.csv:        1 model
baseline_ppl_manifest.csv:           4 inputs
baseline_prepared_roots_manifest:    2 variants
```

## Default Temperature Grid

```text
temp_0
temp_01
temp_02
temp_03
temp_04
temp_05
temp_06
temp_07
temp_08
temp_09
temp_10
```

These tags correspond to temperatures:

```text
0.0, 0.1, 0.2, ..., 1.0
```

## Submit Jobs

Preview only, without submitting jobs:

```bash
SUBMIT_DRY_RUN=1 bash submit_all_eval.sh
```

Submit all PPL and WD jobs:

```bash
bash submit_all_eval.sh
```