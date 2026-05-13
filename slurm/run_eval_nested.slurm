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

N_OPT_STEPS="${N_OPT_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
MAX_PAIRS_PER_QUERY="${MAX_PAIRS_PER_QUERY:-5000}"
SCORE_SCALE="${SCORE_SCALE:-10.0}"
RRF_K_GRID="${RRF_K_GRID:-1 5 10 20 40 60 100}"

INCLUDE_CONFORMAL="${INCLUDE_CONFORMAL:-1}"
CONFORMAL_ALPHA="${CONFORMAL_ALPHA:-0.1}"
CONFORMAL_CAL_SIZE="${CONFORMAL_CAL_SIZE:-0.2}"
CONFORMAL_MODE="${CONFORMAL_MODE:-rank_near_best}"
NEAR_BEST_RULE="${NEAR_BEST_RULE:-relative}"
CONFORMAL_NEAR_BEST_RULES="${CONFORMAL_NEAR_BEST_RULES:-relative}"
NEAR_BEST_EPSILON="${NEAR_BEST_EPSILON:-0.05}"
NEAR_BEST_STD_MULTIPLIER="${NEAR_BEST_STD_MULTIPLIER:-0.5}"
CONFORMAL_MAX_SET_SIZES="${CONFORMAL_MAX_SET_SIZES:-10}"

OPERATIONAL_STD_MULTIPLIERS="${OPERATIONAL_STD_MULTIPLIERS:-0.25 0.5}"
OPERATIONAL_TOP_K="${OPERATIONAL_TOP_K:-5 10}"
IR_CUTOFFS="${IR_CUTOFFS:-10}"

SAVE_RANKINGS="${SAVE_RANKINGS:-1}"
SAVE_CALIBRATION_RANKINGS="${SAVE_CALIBRATION_RANKINGS:-1}"
RANKINGS_FORMAT="${RANKINGS_FORMAT:-parquet}"

SKIP_LEARNED="${SKIP_LEARNED:-0}"
SKIP_RANDOM="${SKIP_RANDOM:-0}"
SKIP_SINGLE="${SKIP_SINGLE:-0}"

VERBOSE="${VERBOSE:-1}"

read -r -a PERF_ARRAY <<< "$PERFORMANCE_COLS"
read -r -a CSV_ARRAY <<< "$CSVS"
read -r -a FEATURE_ARRAY <<< "$FEATURES"
read -r -a RRF_K_ARRAY <<< "$RRF_K_GRID"
read -r -a CONFORMAL_RULE_ARRAY <<< "$CONFORMAL_NEAR_BEST_RULES"
read -r -a CONFORMAL_MAX_SET_SIZE_ARRAY <<< "$CONFORMAL_MAX_SET_SIZES"
read -r -a OPERATIONAL_STD_ARRAY <<< "$OPERATIONAL_STD_MULTIPLIERS"
read -r -a OPERATIONAL_TOP_K_ARRAY <<< "$OPERATIONAL_TOP_K"
read -r -a IR_CUTOFF_ARRAY <<< "$IR_CUTOFFS"

if [[ "$SLURM_ARRAY_TASK_ID" -ge "${#PERF_ARRAY[@]}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID but only ${#PERF_ARRAY[@]} performance columns were provided."
  echo "PERFORMANCE_COLS=$PERFORMANCE_COLS"
  exit 1
fi

PERFORMANCE_COL="${PERF_ARRAY[$SLURM_ARRAY_TASK_ID]}"
TASK_OUT_DIR="${OUT_DIR}/${PERFORMANCE_COL}"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p artifacts/logs "$TASK_OUT_DIR"

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

echo "CSVS: $CSVS"
echo "FEATURES: $FEATURES"
echo "PERFORMANCE_COL: $PERFORMANCE_COL"
echo "OUT_DIR: $TASK_OUT_DIR"
echo "VAL_SIZE: $VAL_SIZE"
echo "N_OPT_STEPS: $N_OPT_STEPS"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "MAX_PAIRS_PER_QUERY: $MAX_PAIRS_PER_QUERY"
echo "SCORE_SCALE: $SCORE_SCALE"
echo "RRF_K_GRID: $RRF_K_GRID"
echo "INCLUDE_CONFORMAL: $INCLUDE_CONFORMAL"
echo "CONFORMAL_MODE: $CONFORMAL_MODE"
echo "CONFORMAL_NEAR_BEST_RULES: $CONFORMAL_NEAR_BEST_RULES"
echo "NEAR_BEST_EPSILON: $NEAR_BEST_EPSILON"
echo "NEAR_BEST_STD_MULTIPLIER: $NEAR_BEST_STD_MULTIPLIER"
echo "CONFORMAL_MAX_SET_SIZES: $CONFORMAL_MAX_SET_SIZES"
echo "OPERATIONAL_STD_MULTIPLIERS: $OPERATIONAL_STD_MULTIPLIERS"
echo "OPERATIONAL_TOP_K: $OPERATIONAL_TOP_K"
echo "IR_CUTOFFS: $IR_CUTOFFS"
echo "SAVE_RANKINGS: $SAVE_RANKINGS"
echo "SAVE_CALIBRATION_RANKINGS: $SAVE_CALIBRATION_RANKINGS"
echo "RANKINGS_FORMAT: $RANKINGS_FORMAT"

python -c "import numpy; print('numpy ok')"
python -c "import pandas; print('pandas ok')"
python -c "import sklearn; print('sklearn ok')"
python -c "import pyarrow; print('pyarrow ok')"
python -c "import lightgbm; print('lightgbm ok')"
python -c "import torch; print('torch ok')"

EXTRA_FLAGS=(
  --include_nested_offline
  --n_opt_steps "$N_OPT_STEPS"
  --learning_rate "$LEARNING_RATE"
  --max_pairs_per_query "$MAX_PAIRS_PER_QUERY"
  --score_scale "$SCORE_SCALE"
  --rrf_k_grid "${RRF_K_ARRAY[@]}"
  --operational_std_multipliers "${OPERATIONAL_STD_ARRAY[@]}"
  --operational_top_k "${OPERATIONAL_TOP_K_ARRAY[@]}"
  --ir_cutoffs "${IR_CUTOFF_ARRAY[@]}"
  --rankings_format "$RANKINGS_FORMAT"
)

if [[ "$SAVE_RANKINGS" == "1" ]]; then
  EXTRA_FLAGS+=(--save_rankings)
else
  EXTRA_FLAGS+=(--no_save_rankings)
fi

if [[ "$SAVE_CALIBRATION_RANKINGS" == "1" ]]; then
  EXTRA_FLAGS+=(--save_calibration_rankings)
else
  EXTRA_FLAGS+=(--no_save_calibration_rankings)
fi

if [[ -n "$DATASET_NAMES" ]]; then
  read -r -a DATASET_NAME_ARRAY <<< "$DATASET_NAMES"
  EXTRA_FLAGS+=(--dataset_names "${DATASET_NAME_ARRAY[@]}")
fi

if [[ "$INCLUDE_CONFORMAL" == "1" ]]; then
  EXTRA_FLAGS+=(
    --include_conformal
    --conformal_alpha "$CONFORMAL_ALPHA"
    --conformal_cal_size "$CONFORMAL_CAL_SIZE"
    --conformal_mode "$CONFORMAL_MODE"
    --near_best_rule "$NEAR_BEST_RULE"
    --conformal_near_best_rules "${CONFORMAL_RULE_ARRAY[@]}"
    --near_best_epsilon "$NEAR_BEST_EPSILON"
    --near_best_std_multiplier "$NEAR_BEST_STD_MULTIPLIER"
    --conformal_max_set_sizes "${CONFORMAL_MAX_SET_SIZE_ARRAY[@]}"
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
echo "  $TASK_OUT_DIR/rankings_${PERFORMANCE_COL}.${RANKINGS_FORMAT}"
echo "  $TASK_OUT_DIR/fold_metadata_${PERFORMANCE_COL}.${RANKINGS_FORMAT}"