# synthetic_time_series_public

`synthetic_time_series_public` is a clean release repository for medical time-series text modeling. It organizes deterministic data rendering, Mamba causal language model training, vLLM generation, memorization probing, and embedding-based Sliced Wasserstein Distance evaluation into a modular and reusable pipeline.

Core pipeline:

```text
MEDS/MIMIC-style time series
  -> deterministic text rendering
  -> Mamba causal LM training
  -> vLLM synthetic or memorization generation
  -> PPL / Sliced WD evaluation
```

## Repository Layout

```text
synthetic_time_series_public/
  configs/
    data/
      mimic_meds.yaml
    training/
      mamba_train.yaml
    inference/
      vllm_generate.yaml
    evaluation/
      sliced_wd.yaml

  scripts/
    training/
      train_mamba.sh
      finetune_from_weights.sh
      resume_interrupted_training.sh
    inference/
      generate_vllm.sh
      generate_memorization_vllm.sh
      june17_clean_rerun/
    evaluation/
      run_sliced_wd.sh
      EVAL_PPL_WD/

  synthetic_time_series_public/
    common/
    data/
    training/
    inference/
    evaluation/
    tests/

  pyproject.toml
  README.md
```

## Installation


Recommended installation:

```bash
cd /path/to/your/synthetic_time_series_public
python -m pip install -e .
```

Install with vLLM support:

```bash
python -m pip install -e ".[vllm]"
```

# Data

## Access the MIMIC-IV data
Complete CITI training and download MIMIC-IV from [https://physionet.org/content/mimiciv/3.1/](https://physionet.org/content/mimiciv/3.1/)

Extract all zipped files in the folder:
```
gunzip *.gz
```

## Extract the data into MEDS format:
[https://github.com/mmcdermott/MEDS_extract](https://github.com/mmcdermott/MEDS_extract)

Use the event_configs.yaml file from the accompanying MIMICIV_MEDS example: [https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS/blob/main/src/MIMIC_IV_MEDS/event_configs.yaml](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS/blob/main/src/MIMIC_IV_MEDS/event_configs.yaml)

1) Modify the `configs/data/event_configs.yaml`
2) Run the command: `meds-extract-run spec=./event_configs.yaml output_dir=$MEDS_OUTPUT do_download=false input_dir=$RAW_DIR`


## MEDS Rendering Format

Each patient is rendered into structured text for causal language modeling. The project uses a fixed set of special tokens:

```text
<static>
</static>
<event>
</event>
<comma>
<newline>
```

The event block CSV header is fixed to:

```text
time,code,numeric_value
```

Rendering outputs these core views:

```text
static_body
event_csv
normalized_text
prompt_prefix
event_count
```

`normalized_text` is the full model input text, `prompt_prefix` is the prefix used for generation, and `event_csv` is the event-level representation consumed by evaluation.

Default text length limit:

```text
max_text_length: 16384
```

Default row limit:

```text
max_rows: 512
```

# Training

Training entrypoint:

```bash
scripts/training/train_mamba.sh
```

Default config:

```text
configs/training/mamba_train.yaml
```

Start fresh training:

```bash
cd /path/to/your/synthetic_time_series_public
CONFIG=configs/training/mamba_train.yaml \
OUTPUT_DIR=/path/to/your/synthetic_time_series_public/training_runs \
scripts/training/train_mamba.sh
```

Start fine-tuning from an existing model or checkpoint:

```bash
MODEL_LOCAL_PATH=/path/to/model_or_checkpoint \
OUTPUT_DIR=/path/to/your/synthetic_time_series_public/finetune_runs \
scripts/training/finetune_from_weights.sh
```

Strictly resume an interrupted run:

```bash
CHECKPOINT=/path/to/run/checkpoint-3000 \
OUTPUT_DIR=/path/to/your/synthetic_time_series_public/resume_runs \
scripts/training/resume_interrupted_training.sh
```

Training outputs include:

```text
config.json / model weights
tokenizer files
resolved_config.yaml
training_summary.json
final_tuning_eval_metrics.json
loss_history.json
loss_history.csv
logs/
checkpoint-*/
```

# Synthetic Generation

Synthetic generation uses vLLM:

```bash
scripts/inference/generate_vllm.sh
```

Default config:

```text
configs/inference/vllm_generate.yaml
```

Default split:

```text
eval_dataset: held_out
```

Default generation hyperparameters:

```text
batch_size: 128
synthetic_samples_per_eval: 1
max_new_tokens: 16384
do_sample: true
temperature: 0.8
top_p: 0.9
top_k: 10
repetition_penalty: 1.1
seed: 42
```


Run:

```bash
RUN_DIR=/path/to/training/run \
OUTPUT_DIR=/path/to/your/synthetic_time_series_public/vllm_heldout_generation \
EVAL_DATASET=held_out \
scripts/inference/generate_vllm.sh
```

Optional checkpoint:

```bash
CHECKPOINT=checkpoint-3000 \
RUN_DIR=/path/to/training/run \
OUTPUT_DIR=/path/to/your/synthetic_time_series_public/vllm_heldout_generation \
scripts/inference/generate_vllm.sh
```

Main outputs:

```text
samples.jsonl
summary.json
args.json
run_spec.json
```

Each row in `samples.jsonl` contains:

```text
original_events
original_full_text
prompt_prefix
synthetic_continuation_raw
synthetic_continuation
synthetic_raw_full_text
synthetic_events
synthetic_full_text
status
valid
is_complete
synthetic_finish_reason
synthetic_stop_reason
synthetic_raw_contains_closing_event_tag
synthetic_event_count
original_event_count
```

# Sliced WD Evaluation

General Sliced Wasserstein Distance evaluation entrypoint:

```bash
scripts/evaluation/run_sliced_wd.sh
```

Default config:

```text
configs/evaluation/sliced_wd.yaml
```

`MANIFEST` must contain `source_path`. It may also include metadata for filtering and grouping:

```csv
run_name,source_label,temp_tag,temperature,fraction_tag,fraction,source_path
my_run,my_model,temp_08,0.8,,,/path/to/samples.jsonl
```

`EMBEDDING_MANIFEST` explicitly defines the embedding model:

```csv
embedding_name,model_path,max_length,batch_size
bge_m3,/path/to/your/VLLM_INF_EVAL/model_cache/bge-m3-safetensors,8192,16
```

Synthetic evaluation:

```bash
TASK_TYPE=synthetic \
MANIFEST=/path/to/synthetic_manifest.csv \
EMBEDDING_MANIFEST=/path/to/embedding_manifest.csv \
OUTPUT_ROOT=/path/to/your/synthetic_time_series_public/wd_synthetic \
scripts/evaluation/run_sliced_wd.sh
```

## AUG11 PPL and WD Batch Recipes

`scripts/evaluation/EVAL_PPL_WD/` provides Slurm batch evaluation recipes for the AUG11 clean rerun. It covers:

```text
PPL synthetic
PPL memorization
PPL baselines
WD synthetic
WD memorization
WD baselines
```

Entrypoint:

```bash
cd /path/to/your/synthetic_time_series_public/scripts/evaluation/EVAL_PPL_WD
SUBMIT_DRY_RUN=1 bash submit_all_eval.sh
bash submit_all_eval.sh
```
