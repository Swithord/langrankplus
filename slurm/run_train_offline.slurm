#!/bin/bash
#SBATCH --job-name=langrank-train
#SBATCH --account=def-annielee
#SBATCH --time=04:00:00
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

NORMALIZER="${NORMALIZER:-minmax}"
RANDOM_STATE="${RANDOM_STATE:-42}"

OUT_DIR="${OUT_DIR:-artifacts/offline_calibration}"

N_OPT_STEPS="${N_OPT_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
MAX_PAIRS_PER_QUERY="${MAX_PAIRS_PER_QUERY:-5000}"
SCORE_SCALE="${SCORE_SCALE:-10.0}"
RRF_K_GRID="${RRF_K_GRID:-1 5 10 20 40 60 100}"

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

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p artifacts/logs "$OUT_DIR"

module load StdEnv/2023
module load arrow/24.0.0

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

echo "Training offline fitted composite/RRF"
echo "CSVS: $CSVS"
echo "FEATURES: $FEATURES"
echo "PERFORMANCE_COL: $PERFORMANCE_COL"
echo "OUT_DIR: $OUT_DIR"
echo "N_OPT_STEPS: $N_OPT_STEPS"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "MAX_PAIRS_PER_QUERY: $MAX_PAIRS_PER_QUERY"
echo "SCORE_SCALE: $SCORE_SCALE"
echo "RRF_K_GRID: $RRF_K_GRID"

python -c "import numpy; print('numpy ok')"
python -c "import pandas; print('pandas ok')"
python -c "import sklearn; print('sklearn ok')"
python -c "import pyarrow; print('pyarrow ok')"
python -c "import torch; print('torch ok')"

EXTRA_FLAGS=(
  --n_opt_steps "$N_OPT_STEPS"
  --learning_rate "$LEARNING_RATE"
  --max_pairs_per_query "$MAX_PAIRS_PER_QUERY"
  --score_scale "$SCORE_SCALE"
  --rrf_k_grid "${RRF_K_ARRAY[@]}"
)

if [[ -n "$DATASET_NAMES" ]]; then
  read -r -a DATASET_NAME_ARRAY <<< "$DATASET_NAMES"
  EXTRA_FLAGS+=(--dataset_names "${DATASET_NAME_ARRAY[@]}")
fi

if [[ "$VERBOSE" == "1" ]]; then
  EXTRA_FLAGS+=(--verbose)
fi

python main.py train \
  --csv "${CSV_ARRAY[@]}" \
  --performance_col "$PERFORMANCE_COL" \
  --features "${FEATURE_ARRAY[@]}" \
  --target_col "$TARGET_COL" \
  --source_col "$SOURCE_COL" \
  --dataset_col "$DATASET_COL" \
  --normalizer "$NORMALIZER" \
  --random_state "$RANDOM_STATE" \
  --outdir "$OUT_DIR" \
  "${EXTRA_FLAGS[@]}"

echo "Finished offline fitting for performance_col=$PERFORMANCE_COL"
echo "Outputs:"
echo "  $OUT_DIR/composite_${PERFORMANCE_COL}.json"
echo "  $OUT_DIR/rrf_${PERFORMANCE_COL}.json"