#!/bin/bash
#SBATCH --job-name=langrank-cross-task
#SBATCH --account=def-annielee
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=artifacts/logs/%x-%A_%a.out
#SBATCH --error=artifacts/logs/%x-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON_MODULE="${PYTHON_MODULE:-python}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-venv}"

FEATURES="${FEATURES:-new_gen new_typ new_geo script distals_asjp distals_wiki_size}"
TARGET_COL="${TARGET_COL:-task_lang}"
SOURCE_COL="${SOURCE_COL:-transfer_lang}"
DATASET_COL="${DATASET_COL:-dataset}"
NORMALIZER="${NORMALIZER:-minmax}"

K="${K:-3}"
TOP_K_RELEVANCE="${TOP_K_RELEVANCE:-10}"
VAL_SIZE="${VAL_SIZE:-0.2}"
VALIDATION_UNIT="${VALIDATION_UNIT:-task}"
RANDOM_STATE="${RANDOM_STATE:-42}"

# Fit Composite-Pairwise and RRF-Pairwise separately inside every outer
# leave-one-task-out fold. This is leakage-safe because the held-out task is
# excluded before these methods are fit.
INCLUDE_NESTED_OFFLINE="${INCLUDE_NESTED_OFFLINE:-0}"
N_OPT_STEPS="${N_OPT_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
MAX_PAIRS_PER_QUERY="${MAX_PAIRS_PER_QUERY:-5000}"
SCORE_SCALE="${SCORE_SCALE:-10.0}"
RRF_K_GRID="${RRF_K_GRID:-1 5 10 20 40 60 100}"

INCLUDE_IR_METRICS="${INCLUDE_IR_METRICS:-1}"
IR_CUTOFFS="${IR_CUTOFFS:-1 3 5 10}"

SKIP_RANDOM="${SKIP_RANDOM:-0}"
SKIP_SINGLE="${SKIP_SINGLE:-0}"
SKIP_LEARNED="${SKIP_LEARNED:-0}"
VERBOSE="${VERBOSE:-1}"

BASE_OUT_DIR="${BASE_OUT_DIR:-artifacts/cross_task_generalization}"

read -r -a FEATURE_ARRAY <<< "$FEATURES"
read -r -a RRF_K_ARRAY <<< "$RRF_K_GRID"
read -r -a IR_CUTOFF_ARRAY <<< "$IR_CUTOFFS"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    MODEL_NAME="mt5"
    CSV_ARRAY=(
      "data/masakhaner_mt5.csv"
      "data/opus100_mt5.csv"
      "data/sib200_mt5.csv"
      "data/taxi1500_mt5.csv"
      "data/tydiqa_mt5.csv"
      "data/ud_pos_mt5.csv"
      "data/wikiann_mt5.csv"
      "data/xnli_mt5.csv"
      "data/xquad_mt5.csv"
    )
    DATASET_NAME_ARRAY=(
      "masakhaner"
      "opus100"
      "sib200"
      "taxi1500"
      "tydiqa"
      "ud_pos"
      "wikiann"
      "xnli"
      "xquad"
    )
    PERFORMANCE_COL_ARRAY=(
      "f1_score"
      "bleu"
      "f1_score"
      "f1_score"
      "f1_score"
      "f1_score"
      "f1_score"
      "f1_score"
      "f1_score"
    )
    ;;

  1)
    MODEL_NAME="xlm-r"
    CSV_ARRAY=(
      "data/masakhaner_xlm-r.csv"
      "data/sib200_xlm-r.csv"
      "data/taxi1500_xlm-r.csv"
      "data/tydiqa_xlm-r.csv"
      "data/ud_dep_xlm-r.csv"
      "data/ud_pos_xlm-r.csv"
      "data/wikiann_xlm-r.csv"
      "data/xnli_xlm-r.csv"
      "data/xquad_xlm-r.csv"
    )
    DATASET_NAME_ARRAY=(
      "masakhaner"
      "sib200"
      "taxi1500"
      "tydiqa"
      "ud_dep"
      "ud_pos"
      "wikiann"
      "xnli"
      "xquad"
    )
    PERFORMANCE_COL_ARRAY=(
      "f1_score"
      "f1_score"
      "f1_score"
      "f1_score"
      "las"
      "f1_score"
      "f1_score"
      "f1_score"
      "f1_score"
    )
    ;;

  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID; expected 0 or 1."
    exit 1
    ;;
esac

if [[ "${#CSV_ARRAY[@]}" -ne "${#DATASET_NAME_ARRAY[@]}" ]]; then
  echo "Internal configuration error: CSV and task-name arrays differ in length."
  exit 1
fi

if [[ "${#CSV_ARRAY[@]}" -ne "${#PERFORMANCE_COL_ARRAY[@]}" ]]; then
  echo "Internal configuration error: CSV and performance-column arrays differ in length."
  exit 1
fi

for csv_path in "${CSV_ARRAY[@]}"; do
  if [[ ! -f "$csv_path" ]]; then
    echo "Required input CSV not found: $csv_path"
    exit 1
  fi
done

OUT_DIR="${BASE_OUT_DIR}/${MODEL_NAME}"
mkdir -p artifacts/logs "$OUT_DIR"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [[ -n "$PYTHON_MODULE" ]]; then
  module load "$PYTHON_MODULE"
fi

for module_name in $EXTRA_MODULES; do
  module load "$module_name"
done

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "No virtual environment found at $VENV_PATH"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

printf '%s\n' \
  "Started: $(date)" \
  "PWD: $(pwd)" \
  "Host: $(hostname)" \
  "Python: $(which python)" \
  "Model: $MODEL_NAME" \
  "Tasks: ${DATASET_NAME_ARRAY[*]}" \
  "Features: ${FEATURE_ARRAY[*]}" \
  "Validation unit: $VALIDATION_UNIT" \
  "Validation size: $VAL_SIZE" \
  "Output directory: $OUT_DIR"

python --version
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import lightgbm; print('lightgbm:', lightgbm.__version__)"
python -c "import torch; print('torch:', torch.__version__)"

EXTRA_FLAGS=(
  --rrf_k_grid "${RRF_K_ARRAY[@]}"
)

if [[ "$INCLUDE_NESTED_OFFLINE" == "1" ]]; then
  EXTRA_FLAGS+=(--include_nested_offline)
fi

if [[ "$INCLUDE_IR_METRICS" == "1" ]]; then
  EXTRA_FLAGS+=(
    --include_ir_metrics
    --ir_cutoffs "${IR_CUTOFF_ARRAY[@]}"
  )
fi

if [[ "$SKIP_RANDOM" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_random)
fi

if [[ "$SKIP_SINGLE" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_single)
fi

if [[ "$SKIP_LEARNED" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_learned)
fi

if [[ "$VERBOSE" == "1" ]]; then
  EXTRA_FLAGS+=(--verbose)
fi

python scripts/evaluate_cross_task.py \
  --csv "${CSV_ARRAY[@]}" \
  --dataset_names "${DATASET_NAME_ARRAY[@]}" \
  --performance_cols "${PERFORMANCE_COL_ARRAY[@]}" \
  --model_name "$MODEL_NAME" \
  --features "${FEATURE_ARRAY[@]}" \
  --target_col "$TARGET_COL" \
  --source_col "$SOURCE_COL" \
  --dataset_col "$DATASET_COL" \
  --normalizer "$NORMALIZER" \
  --k "$K" \
  --top_k_relevance "$TOP_K_RELEVANCE" \
  --val_size "$VAL_SIZE" \
  --validation_unit "$VALIDATION_UNIT" \
  --random_state "$RANDOM_STATE" \
  --n_opt_steps "$N_OPT_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --max_pairs_per_query "$MAX_PAIRS_PER_QUERY" \
  --score_scale "$SCORE_SCALE" \
  --outdir "$OUT_DIR" \
  "${EXTRA_FLAGS[@]}"

printf '%s\n' \
  "Finished: $(date)" \
  "Model: $MODEL_NAME" \
  "Generated files:" \
  "  $OUT_DIR/summary_macro_task.csv" \
  "  $OUT_DIR/summary_by_heldout_task.csv" \
  "  $OUT_DIR/summary_pooled_query.csv" \
  "  $OUT_DIR/per_query.csv" \
  "  $OUT_DIR/pairwise_macro_task.csv" \
  "  $OUT_DIR/pairwise_pooled_query.csv" \
  "  $OUT_DIR/transfer_type_loss.csv" \
  "  $OUT_DIR/run_manifest.json"
