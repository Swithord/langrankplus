#!/bin/bash
#SBATCH --job-name=mondrian-ctc
#SBATCH --account=def-mere
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5%6
#SBATCH --output=artifacts/logs/%x-%A_%a.out
#SBATCH --error=artifacts/logs/%x-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON_MODULE="${PYTHON_MODULE:-python}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-$SLURM_SUBMIT_DIR/venv}"

FEATURES="${FEATURES:-new_gen new_typ new_geo script distals_asjp distals_wiki_size}"
TARGET_COL="${TARGET_COL:-task_lang}"
SOURCE_COL="${SOURCE_COL:-transfer_lang}"
DATASET_COL="${DATASET_COL:-dataset}"
PERFORMANCE_COL="${PERFORMANCE_COL:-f1_score}"

K="${K:-3}"
TOP_K_RELEVANCE="${TOP_K_RELEVANCE:-10}"
VAL_SIZE="${VAL_SIZE:-0}"
RANDOM_STATE="${RANDOM_STATE:-42}"
NORMALIZER="${NORMALIZER:-minmax}"

CNOTC_ALPHA="${CNOTC_ALPHA:-0.1}"
CNOTC_EPSILON="${CNOTC_EPSILON:-0.05}"
CNOTC_CAL_SIZE="${CNOTC_CAL_SIZE:-0.2}"
CNOTC_UNKNOWN_POLICY="${CNOTC_UNKNOWN_POLICY:-error}"
BUDGET_KS="${BUDGET_KS:-10}"

OUT_ROOT="${OUT_ROOT:-artifacts/mondrian_ctc}"
VERBOSE="${VERBOSE:-1}"

TASKS=(
  sib200
  sib200
  taxi1500
  taxi1500
  ud_pos
  ud_pos
)

MODELS=(
  xlm-r
  mt5
  xlm-r
  mt5
  xlm-r
  mt5
)

CSVS=(
  data/sib200_xlm-r.csv
  data/sib200_mt5.csv
  data/taxi1500_xlm-r.csv
  data/taxi1500_mt5.csv
  data/ud_pos_xlm-r.csv
  data/ud_pos_mt5.csv
)

IDX="${SLURM_ARRAY_TASK_ID}"
if (( IDX < 0 || IDX >= ${#CSVS[@]} )); then
  echo "Invalid array index: $IDX" >&2
  exit 1
fi

TASK="${TASKS[$IDX]}"
MODEL="${MODELS[$IDX]}"
CSV="${CSVS[$IDX]}"
DATASET_NAME="${TASK}_${MODEL}"
OUT_DIR="${OUT_ROOT}/${DATASET_NAME}"

if [[ ! -f "$CSV" ]]; then
  echo "Missing input CSV: $CSV" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$RANDOM_STATE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ -n "$PYTHON_MODULE" ]]; then
  module load "$PYTHON_MODULE"
fi
for mod in $EXTRA_MODULES; do
  module load "$mod"
done

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "No virtual environment found at: $VENV_PATH" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

read -r -a FEATURE_ARRAY <<< "$FEATURES"
read -r -a BUDGET_ARRAY <<< "$BUDGET_KS"

EXTRA_FLAGS=(
  --include_cnotc
  --cnotc_alpha "$CNOTC_ALPHA"
  --cnotc_epsilon "$CNOTC_EPSILON"
  --cnotc_cal_size "$CNOTC_CAL_SIZE"
  --cnotc_grouping resource_level
  --cnotc_unknown_policy "$CNOTC_UNKNOWN_POLICY"
  --budget_ks "${BUDGET_ARRAY[@]}"
  --skip_random
)

if [[ "$VERBOSE" == "1" ]]; then
  EXTRA_FLAGS+=(--verbose)
fi

echo "HOST: $(hostname)"
echo "PWD: $(pwd)"
echo "PYTHON: $(which python)"
python --version
echo "TASK: $TASK"
echo "MODEL: $MODEL"
echo "CSV: $CSV"
echo "OUT_DIR: $OUT_DIR"
echo "FEATURES: $FEATURES"
echo "MONDRIAN GROUP: target resource level"
echo "ALPHA: $CNOTC_ALPHA"
echo "EPSILON: $CNOTC_EPSILON"
echo "CALIBRATION FRACTION: $CNOTC_CAL_SIZE"

python -c "import lightgbm, numpy, pandas, sklearn, torch; print('dependencies ok')"

python main.py evaluate \
  --csv "$CSV" \
  --dataset_names "$DATASET_NAME" \
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
  --outdir "$OUT_DIR" \
  "${EXTRA_FLAGS[@]}"

echo "Finished ${TASK}/${MODEL}"
echo "Mondrian summary: ${OUT_DIR}/mondrian_ctc_${PERFORMANCE_COL}.csv"
echo "Per-fold diagnostics: ${OUT_DIR}/per_fold_${PERFORMANCE_COL}.csv"
