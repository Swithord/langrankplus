# Slurm scripts

This directory contains Slurm entry points for LangRankPlus.

The main same-benchmark script is:

```bash
sbatch slurm/run_eval_nested.slurm
```

It runs the full comparison:

```text
random
single_new_gen
single_new_typ
single_new_geo
single_script
composite_equal
rrf_equal
composite_offline_nested
rrf_offline_nested
lightgbm_lambdarank
mlp_listnet
```

By default:

```text
VAL_SIZE=0
INCLUDE_CONFORMAL=1
N_CALIBRATION_SAMPLES=500
```

So nested composite/RRF and supervised LangRank-style rankers use the same fitting rows in each fold, and conformal outputs are included.

---

## Scripts

| Script | Use |
|---|---|
| `run_eval_nested.slurm` | Main same-benchmark evaluation with nested fitted composite/RRF |
| `run_eval.slurm` | Faster diagnostic evaluation without fitted composite/RRF |
| `run_train_offline.slurm` | Fit frozen composite/RRF weights and save JSON files |
| `run_eval_frozen.slurm` | Evaluate frozen fitted composite/RRF on a dataset |
| `run_infer.slurm` | Run source selection on an unlabelled CSV |

---

## Main same-benchmark evaluation

Run:

```bash
sbatch slurm/run_eval_nested.slurm
```

Default input:

```text
data/sib200.csv
PERFORMANCE_COLS="accuracy f1_score"
FEATURES="new_gen new_typ new_geo script"
```

Outputs:

```text
artifacts/evaluation_sib200_nested/accuracy/
artifacts/evaluation_sib200_nested/f1_score/
```

Each output directory contains:

```text
summary_<performance_col>.csv
per_fold_<performance_col>.csv
pairwise_<performance_col>.csv
```

Increase the nested calibration search budget:

```bash
N_CALIBRATION_SAMPLES=5000 sbatch slurm/run_eval_nested.slurm
```

Disable conformal evaluation:

```bash
INCLUDE_CONFORMAL=0 sbatch slurm/run_eval_nested.slurm
```

---

## Fast diagnostic evaluation

Run this to check the environment and the non-fitted baselines before running nested composite/RRF:

```bash
sbatch slurm/run_eval.slurm
```

This compares:

```text
random
single_*
composite_equal
rrf_equal
lightgbm_lambdarank
mlp_listnet
```

It does not include:

```text
composite_offline_nested
rrf_offline_nested
```

---

## Fit frozen composite/RRF weights

Single-dataset fitting:

```bash
sbatch slurm/run_train_offline.slurm
```

This writes:

```text
artifacts/offline_calibration/composite_accuracy.json
artifacts/offline_calibration/rrf_accuracy.json
artifacts/offline_calibration/composite_f1_score.json
artifacts/offline_calibration/rrf_f1_score.json
```

Multi-dataset fitting:

```bash
CSVS="data/sib200.csv data/masakhaner.csv data/tydiqa.csv" \
OUT_DIR="artifacts/offline_calibration_multi" \
sbatch slurm/run_train_offline.slurm
```

Use frozen weights when fitting data and evaluation data are separate.

---

## Evaluate frozen fitted composite/RRF

Evaluate frozen weights on a new dataset:

```bash
CSVS="data/new_dataset.csv" \
CALIBRATION_DIR="artifacts/offline_calibration_multi" \
OUT_DIR="artifacts/evaluation_new_dataset_frozen" \
sbatch slurm/run_eval_frozen.slurm
```

The script expects:

```text
$CALIBRATION_DIR/composite_<performance_col>.json
$CALIBRATION_DIR/rrf_<performance_col>.json
```

For same-benchmark fitted composite/RRF, use nested evaluation instead:

```bash
sbatch slurm/run_eval_nested.slurm
```

---

## Inference on an unlabelled dataset

Equal composite:

```bash
CSVS="data/new_task.csv" \
METHOD="composite_equal" \
OUT_DIR="artifacts/inference_new_task" \
sbatch slurm/run_infer.slurm
```

Equal RRF:

```bash
CSVS="data/new_task.csv" \
METHOD="rrf_equal" \
OUT_DIR="artifacts/inference_new_task_rrf" \
sbatch slurm/run_infer.slurm
```

Calibrated selector:

```bash
CSVS="data/new_task.csv" \
METHOD="calibrated" \
CALIBRATION_JSON="artifacts/offline_calibration_multi/rrf_accuracy.json" \
OUT_DIR="artifacts/inference_new_task_calibrated" \
sbatch slurm/run_infer.slurm
```

Inference does not compute evaluation metrics because no transfer-performance labels are required.

---

## Common overrides

### Performance columns

Default:

```text
PERFORMANCE_COLS="accuracy f1_score"
#SBATCH --array=0-1
```

Use only accuracy:

```bash
PERFORMANCE_COLS="accuracy" \
sbatch --array=0-0 slurm/run_eval_nested.slurm
```

Use three performance columns:

```bash
PERFORMANCE_COLS="accuracy f1_score matthews_corr" \
sbatch --array=0-2 slurm/run_eval_nested.slurm
```

### Datasets

Single dataset:

```bash
CSVS="data/sib200.csv" sbatch slurm/run_eval_nested.slurm
```

Multiple datasets:

```bash
CSVS="data/sib200.csv data/masakhaner.csv data/tydiqa.csv" \
sbatch slurm/run_eval_nested.slurm
```

With explicit names:

```bash
CSVS="data/sib200.csv data/masakhaner.csv data/tydiqa.csv" \
DATASET_NAMES="sib200 masakhaner tydiqa" \
sbatch slurm/run_eval_nested.slurm
```

### Conformal and validation splits

Default:

```text
VAL_SIZE=0
INCLUDE_CONFORMAL=1
CONFORMAL_CAL_SIZE=0.2
```

Disable conformal evaluation:

```bash
INCLUDE_CONFORMAL=0 sbatch slurm/run_eval_nested.slurm
```

Use an internal validation split inside the fitting pool:

```bash
VAL_SIZE=0.1 sbatch slurm/run_eval_nested.slurm
```

For the main comparison, keep:

```text
VAL_SIZE=0
```

so nested composite/RRF and supervised LangRank-style rankers use the same fitting rows.

---

## Recommended workflow

Same-benchmark comparison:

```bash
sbatch slurm/run_eval_nested.slurm
```

Cross-dataset frozen fitting:

```bash
CSVS="data/sib200.csv data/masakhaner.csv data/tydiqa.csv" \
OUT_DIR="artifacts/offline_calibration_multi" \
sbatch slurm/run_train_offline.slurm
```

```bash
CSVS="data/new_dataset.csv" \
CALIBRATION_DIR="artifacts/offline_calibration_multi" \
OUT_DIR="artifacts/evaluation_new_dataset_frozen" \
sbatch slurm/run_eval_frozen.slurm
```

Deployment-style inference:

```bash
CSVS="data/new_task.csv" \
METHOD="calibrated" \
CALIBRATION_JSON="artifacts/offline_calibration_multi/rrf_accuracy.json" \
OUT_DIR="artifacts/inference_new_task" \
sbatch slurm/run_infer.slurm
```

---

## Notes

- These are CPU jobs.
- `run_eval_nested.slurm` is the main same-benchmark evaluation script.
- `run_eval.slurm` is mainly for fast diagnostics.
- `run_eval_frozen.slurm` is for cross-dataset or deployment-style evaluation.
- `INCLUDE_CONFORMAL=1` is enabled by default in evaluation scripts.
- `VAL_SIZE=0` makes nested composite/RRF and supervised LangRank-style rankers use the same fitting rows.


## Running the Llama eval uses:

```bash
STAGE=train bash slurm/submit_llama.sh sib200     --base-model models/base/Llama-3.1-8B
```

for training and then 

```bash
EVAL_SHARD_SIZE=205 \
STAGE=eval \
bash slurm/submit_llama.sh sib200 \
    --base-model models/base/Llama-3.1-8B


EVAL_SHARD_SIZE=15 STAGE=eval bash slurm/submit_llama.sh xnli \
    --base-model models/base/Llama-3.1-8B

EVAL_SHARD_SIZE=176 STAGE=eval bash slurm/submit_llama.sh wikiann \
    --base-model models/base/Llama-3.1-8B
```

for evaluation.