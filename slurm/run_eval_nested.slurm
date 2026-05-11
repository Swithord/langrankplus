#!/bin/bash
#SBATCH --job-name=langrank-nested
#SBATCH --account=def-annielee
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --output=artifacts/logs/%x-%A_%a.out
#SBATCH --error=artifacts/logs/%x-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON_MODULE="${PYTHON_MODULE:-python}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-venv}"

CSVS="${CSVS:-data/sib200.csv}"
DATASET_NAMES="${DATASET_NAMES:-}"
FEATURES="${FEATURES:-new_gen new_typ new_geo script}"
PERFORMANCE_COLS="${PERFORMANCE_COLS:-accuracy f1_score}"

TARGET_COL="${TARGET_COL:-task_lang}"
SOURCE_COL="${SOURCE_COL:-transfer_lang}"
DATASET_COL="${DATASET_COL:-dataset}"

K="${K:-3}"
TOP_K_RELEVANCE="${TOP_K_RELEVANCE:-10}"
VAL_SIZE="${VAL_SIZE:-0}"
RANDOM_STATE="${RANDOM_STATE:-42}"
NORMALIZER="${NORMALIZER:-minmax}"

OUT_DIR="${OUT_DIR:-artifacts/evaluation_sib200_nested}"

N_CALIBRATION_SAMPLES="${N_CALIBRATION_SAMPLES:-500}"
RRF_K_GRID="${RRF_K_GRID:-1 5 10 20 40 60 100}"

INCLUDE_CONFORMAL="${INCLUDE_CONFORMAL:-1}"
CONFORMAL_ALPHA="${CONFORMAL_ALPHA:-0.1}"
CONFORMAL_CAL_SIZE="${CONFORMAL_CAL_SIZE:-0.2}"

SKIP_LEARNED="${SKIP_LEARNED:-0}"
SKIP_RANDOM="${SKIP_RANDOM:-0}"
SKIP_SINGLE="${SKIP_SINGLE:-0}"

VERBOSE="${VERBOSE:-1}"

read -r -a PERF_ARRAY <<< "$PERFORMANCE_COLS"
read -r -a CSV_ARRAY <<< "$CSVS"
read -r -a FEATURE_ARRAY <<< "$FEATURES"
read -r -a RRF_K_ARRAY <<< "$RRF_K_GRID"

if [[ "$SLURM_ARRAY_TASK_ID" -ge "${#PERF_ARRAY[@]}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID but only ${#PERF_ARRAY[@]} performance columns were provided."
  echo "PERFORMANCE_COLS=$PERFORMANCE_COLS"
  exit 1
fi

PERFORMANCE_COL="${PERF_ARRAY[$SLURM_ARRAY_TASK_ID]}"
TASK_OUT_DIR="${OUT_DIR}/${PERFORMANCE_COL}"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p artifacts/logs "$TASK_OUT_DIR"

if [[ -n "$PYTHON_MODULE" ]]; then
  module load "$PYTHON_MODULE"
fi

for mod in $EXTRA_MODULES; do
  module load "$mod"
done

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "No virtualenv found at $VENV_PATH"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "which python: $(which python)"
python --version

echo "CSVS: $CSVS"
echo "FEATURES: $FEATURES"
echo "PERFORMANCE_COL: $PERFORMANCE_COL"
echo "OUT_DIR: $TASK_OUT_DIR"
echo "VAL_SIZE: $VAL_SIZE"
echo "N_CALIBRATION_SAMPLES: $N_CALIBRATION_SAMPLES"
echo "RRF_K_GRID: $RRF_K_GRID"
echo "INCLUDE_CONFORMAL: $INCLUDE_CONFORMAL"

python -c "import numpy; print('numpy ok')"
python -c "import pandas; print('pandas ok')"
python -c "import sklearn; print('sklearn ok')"
python -c "import lightgbm; print('lightgbm ok')"
python -c "import torch; print('torch ok')"

EXTRA_FLAGS=(
  --include_nested_offline
  --n_calibration_samples "$N_CALIBRATION_SAMPLES"
  --rrf_k_grid "${RRF_K_ARRAY[@]}"
)

if [[ -n "$DATASET_NAMES" ]]; then
  read -r -a DATASET_NAME_ARRAY <<< "$DATASET_NAMES"
  EXTRA_FLAGS+=(--dataset_names "${DATASET_NAME_ARRAY[@]}")
fi

if [[ "$INCLUDE_CONFORMAL" == "1" ]]; then
  EXTRA_FLAGS+=(
    --include_conformal
    --conformal_alpha "$CONFORMAL_ALPHA"
    --conformal_cal_size "$CONFORMAL_CAL_SIZE"
  )
fi

if [[ "$SKIP_LEARNED" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_learned)
fi

if [[ "$SKIP_RANDOM" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_random)
fi

if [[ "$SKIP_SINGLE" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_single)
fi

if [[ "$VERBOSE" == "1" ]]; then
  EXTRA_FLAGS+=(--verbose)
fi

python main.py evaluate \
  --csv "${CSV_ARRAY[@]}" \
  --performance_col "$PERFORMANCE_COL" \
  --features "${FEATURE_ARRAY[@]}" \
  --target_col "$TARGET_COL" \
  --source_col "$SOURCE_COL" \
  --dataset_col "$DATASET_COL" \
  --normalizer "$NORMALIZER" \
  --k "$K" \
  --top_k_relevance "$TOP_K_RELEVANCE" \
  --val_size "$VAL_SIZE" \
  --random_state "$RANDOM_STATE" \
  --outdir "$TASK_OUT_DIR" \
  "${EXTRA_FLAGS[@]}"

echo "Finished nested evaluation for performance_col=$PERFORMANCE_COL"
echo "Outputs:"
echo "  $TASK_OUT_DIR/summary_${PERFORMANCE_COL}.csv"
echo "  $TASK_OUT_DIR/per_fold_${PERFORMANCE_COL}.csv"
echo "  $TASK_OUT_DIR/pairwise_${PERFORMANCE_COL}.csv"