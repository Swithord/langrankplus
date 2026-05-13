#!/bin/bash
#SBATCH --job-name=langrank-shortlists
#SBATCH --account=def-annielee
#SBATCH --time=06:00:00
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

PERFORMANCE_COLS="${PERFORMANCE_COLS:-accuracy f1_score}"

EVAL_DIR="${EVAL_DIR:-artifacts/evaluation_sib200_default}"
RANKING_INPUTS="${RANKING_INPUTS:-}"

OUT_DIR="${OUT_DIR:-artifacts/shortlists_sib200_default}"

METHOD_COL="${METHOD_COL:-auto}"
DATASET_COL="${DATASET_COL:-auto}"
TARGET_COL="${TARGET_COL:-auto}"
SOURCE_COL="${SOURCE_COL:-auto}"
SCORE_COL="${SCORE_COL:-auto}"
RANK_COL="${RANK_COL:-auto}"

FIXED_TOP_K="${FIXED_TOP_K:-3 5 10}"

RELATIVE_EPSILONS="${RELATIVE_EPSILONS:-0.05}"
STD_MULTIPLIERS="${STD_MULTIPLIERS:-0 0.25 0.5 1.0}"

CONFORMAL_ALPHAS="${CONFORMAL_ALPHAS:-0.1}"
CONFORMAL_CAPS="${CONFORMAL_CAPS:-3 5 10}"

CALIBRATED_TARGETS="${CALIBRATED_TARGETS:-0.75 0.8 0.9}"
CALIBRATED_MAX_K="${CALIBRATED_MAX_K:-3 5 10}"

SCORE_GAP_BUDGETS="${SCORE_GAP_BUDGETS:-3 5 10}"
SCORE_GAP_TOLERANCE="${SCORE_GAP_TOLERANCE:-0.01}"

ELBOW_MAX_K="${ELBOW_MAX_K:-5 10}"

CONSENSUS_TOP_K0="${CONSENSUS_TOP_K0:-5 10}"
CONSENSUS_VOTE_THRESHOLDS="${CONSENSUS_VOTE_THRESHOLDS:-2 3 4}"
CONSENSUS_CAPS="${CONSENSUS_CAPS:-3 5 10}"
CONSENSUS_EXCLUDE_METHODS="${CONSENSUS_EXCLUDE_METHODS:-random}"

DISTANCE_FEATURES="${DISTANCE_FEATURES:-new_gen new_typ new_geo script}"
PARETO_CAPS="${PARETO_CAPS:-3 5 10}"

SKIP_FIXED_TOPK="${SKIP_FIXED_TOPK:-0}"
SKIP_CONFORMAL="${SKIP_CONFORMAL:-0}"
SKIP_CALIBRATED_MIN_K="${SKIP_CALIBRATED_MIN_K:-0}"

SKIP_SCORE_GAP="${SKIP_SCORE_GAP:-1}"
SKIP_CONSENSUS="${SKIP_CONSENSUS:-1}"
SKIP_PARETO="${SKIP_PARETO:-1}"

SKIP_ELBOW="${SKIP_ELBOW:-0}"

MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_FORMATS="${PLOT_FORMATS:-png pdf}"

VERBOSE="${VERBOSE:-1}"

read -r -a PERF_ARRAY <<< "$PERFORMANCE_COLS"

if [[ "$SLURM_ARRAY_TASK_ID" -ge "${#PERF_ARRAY[@]}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID but only ${#PERF_ARRAY[@]} performance columns were provided."
  echo "PERFORMANCE_COLS=$PERFORMANCE_COLS"
  exit 1
fi

PERFORMANCE_COL="${PERF_ARRAY[$SLURM_ARRAY_TASK_ID]}"
TASK_EVAL_DIR="${EVAL_DIR}/${PERFORMANCE_COL}"
TASK_OUT_DIR="${OUT_DIR}/${PERFORMANCE_COL}"

if [[ -n "$RANKING_INPUTS" ]]; then
  read -r -a RANKING_ARRAY <<< "$RANKING_INPUTS"
else
  if [[ ! -d "$TASK_EVAL_DIR" ]]; then
    echo "Could not find evaluation directory: $TASK_EVAL_DIR"
    echo "Set EVAL_DIR or RANKING_INPUTS explicitly."
    exit 1
  fi

  mapfile -t RANKING_ARRAY < <(
    find "$TASK_EVAL_DIR" \
      -maxdepth 2 \
      -type f \
      -name "*.parquet" \
      \( -iname "*ranking*.parquet" -o -iname "*rankings*.parquet" \) \
      | sort
  )

  if [[ "${#RANKING_ARRAY[@]}" -eq 0 ]]; then
    echo "No ranking parquet files found under: $TASK_EVAL_DIR"
    echo ""
    echo "Available parquet files under this directory are:"
    find "$TASK_EVAL_DIR" -maxdepth 2 -type f -name "*.parquet" | sort || true
    echo ""
    echo "Set RANKING_INPUTS explicitly, for example:"
    echo "  RANKING_INPUTS=\"artifacts/evaluation_sib200_default/${PERFORMANCE_COL}/rankings_${PERFORMANCE_COL}.parquet\" sbatch --array=${SLURM_ARRAY_TASK_ID}-${SLURM_ARRAY_TASK_ID} slurm/run_shortlists_posthoc.slurm"
    exit 1
  fi
fi

read -r -a FIXED_TOP_K_ARRAY <<< "$FIXED_TOP_K"
read -r -a RELATIVE_EPSILON_ARRAY <<< "$RELATIVE_EPSILONS"
read -r -a STD_MULTIPLIER_ARRAY <<< "$STD_MULTIPLIERS"
read -r -a CONFORMAL_ALPHA_ARRAY <<< "$CONFORMAL_ALPHAS"
read -r -a CONFORMAL_CAP_ARRAY <<< "$CONFORMAL_CAPS"
read -r -a CALIBRATED_TARGET_ARRAY <<< "$CALIBRATED_TARGETS"
read -r -a CALIBRATED_MAX_K_ARRAY <<< "$CALIBRATED_MAX_K"
read -r -a SCORE_GAP_BUDGET_ARRAY <<< "$SCORE_GAP_BUDGETS"
read -r -a ELBOW_MAX_K_ARRAY <<< "$ELBOW_MAX_K"
read -r -a CONSENSUS_TOP_K0_ARRAY <<< "$CONSENSUS_TOP_K0"
read -r -a CONSENSUS_VOTE_THRESHOLD_ARRAY <<< "$CONSENSUS_VOTE_THRESHOLDS"
read -r -a CONSENSUS_CAP_ARRAY <<< "$CONSENSUS_CAPS"
read -r -a CONSENSUS_EXCLUDE_METHOD_ARRAY <<< "$CONSENSUS_EXCLUDE_METHODS"
read -r -a DISTANCE_FEATURE_ARRAY <<< "$DISTANCE_FEATURES"
read -r -a PARETO_CAP_ARRAY <<< "$PARETO_CAPS"
read -r -a PLOT_FORMAT_ARRAY <<< "$PLOT_FORMATS"

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

echo "PERFORMANCE_COL: $PERFORMANCE_COL"
echo "EVAL_DIR: $EVAL_DIR"
echo "TASK_EVAL_DIR: $TASK_EVAL_DIR"
echo "RANKING_INPUTS:"
printf '  %s\n' "${RANKING_ARRAY[@]}"
echo "OUT_DIR: $TASK_OUT_DIR"
echo "FIXED_TOP_K: $FIXED_TOP_K"
echo "RELATIVE_EPSILONS: $RELATIVE_EPSILONS"
echo "STD_MULTIPLIERS: $STD_MULTIPLIERS"
echo "CONFORMAL_ALPHAS: $CONFORMAL_ALPHAS"
echo "CONFORMAL_CAPS: $CONFORMAL_CAPS"
echo "CALIBRATED_TARGETS: $CALIBRATED_TARGETS"
echo "CALIBRATED_MAX_K: $CALIBRATED_MAX_K"
echo "SCORE_GAP_BUDGETS: $SCORE_GAP_BUDGETS"
echo "ELBOW_MAX_K: $ELBOW_MAX_K"
echo "CONSENSUS_TOP_K0: $CONSENSUS_TOP_K0"
echo "CONSENSUS_VOTE_THRESHOLDS: $CONSENSUS_VOTE_THRESHOLDS"
echo "CONSENSUS_CAPS: $CONSENSUS_CAPS"
echo "CONSENSUS_EXCLUDE_METHODS: $CONSENSUS_EXCLUDE_METHODS"
echo "DISTANCE_FEATURES: $DISTANCE_FEATURES"
echo "PARETO_CAPS: $PARETO_CAPS"
echo "SKIP_SCORE_GAP: $SKIP_SCORE_GAP"
echo "SKIP_CONSENSUS: $SKIP_CONSENSUS"
echo "SKIP_PARETO: $SKIP_PARETO"
echo "MAKE_PLOTS: $MAKE_PLOTS"
echo "PLOT_FORMATS: $PLOT_FORMATS"

python -c "import numpy; print('numpy ok')"
python -c "import pandas; print('pandas ok')"
python -c "import pyarrow; print('pyarrow ok')"
python -c "import tqdm; print('tqdm ok')"
python -c "import matplotlib; print('matplotlib ok')"

EXTRA_FLAGS=()

if [[ "$SKIP_FIXED_TOPK" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_fixed_topk)
fi

if [[ "$SKIP_CONFORMAL" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_conformal)
fi

if [[ "$SKIP_CALIBRATED_MIN_K" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_calibrated_min_k)
fi

if [[ "$SKIP_SCORE_GAP" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_score_gap)
fi

if [[ "$SKIP_ELBOW" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_elbow)
fi

if [[ "$SKIP_CONSENSUS" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_consensus)
fi

if [[ "$SKIP_PARETO" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_pareto)
fi

if [[ "$MAKE_PLOTS" == "0" ]]; then
  EXTRA_FLAGS+=(--skip_plots)
else
  EXTRA_FLAGS+=(--plot_formats "${PLOT_FORMAT_ARRAY[@]}")
fi

if [[ "$VERBOSE" == "1" ]]; then
  EXTRA_FLAGS+=(--verbose)
fi

python -m scripts.evaluate_shortlists \
  --rankings "${RANKING_ARRAY[@]}" \
  --performance_col "$PERFORMANCE_COL" \
  --method_col "$METHOD_COL" \
  --dataset_col "$DATASET_COL" \
  --target_col "$TARGET_COL" \
  --source_col "$SOURCE_COL" \
  --score_col "$SCORE_COL" \
  --rank_col "$RANK_COL" \
  --fixed_top_k "${FIXED_TOP_K_ARRAY[@]}" \
  --relative_epsilons "${RELATIVE_EPSILON_ARRAY[@]}" \
  --std_multipliers "${STD_MULTIPLIER_ARRAY[@]}" \
  --conformal_alphas "${CONFORMAL_ALPHA_ARRAY[@]}" \
  --conformal_caps "${CONFORMAL_CAP_ARRAY[@]}" \
  --calibrated_targets "${CALIBRATED_TARGET_ARRAY[@]}" \
  --calibrated_max_k "${CALIBRATED_MAX_K_ARRAY[@]}" \
  --score_gap_budgets "${SCORE_GAP_BUDGET_ARRAY[@]}" \
  --score_gap_tolerance "$SCORE_GAP_TOLERANCE" \
  --elbow_max_k "${ELBOW_MAX_K_ARRAY[@]}" \
  --consensus_top_k0 "${CONSENSUS_TOP_K0_ARRAY[@]}" \
  --consensus_vote_thresholds "${CONSENSUS_VOTE_THRESHOLD_ARRAY[@]}" \
  --consensus_caps "${CONSENSUS_CAP_ARRAY[@]}" \
  --consensus_exclude_methods "${CONSENSUS_EXCLUDE_METHOD_ARRAY[@]}" \
  --distance_features "${DISTANCE_FEATURE_ARRAY[@]}" \
  --pareto_caps "${PARETO_CAP_ARRAY[@]}" \
  --outdir "$TASK_OUT_DIR" \
  "${EXTRA_FLAGS[@]}"

echo "Finished post-hoc shortlist evaluation for performance_col=$PERFORMANCE_COL"
echo "Outputs:"
echo "  $TASK_OUT_DIR/shortlist_summary_${PERFORMANCE_COL}.csv"
echo "  $TASK_OUT_DIR/shortlist_per_query_${PERFORMANCE_COL}.csv"
echo "  $TASK_OUT_DIR/shortlist_summary_${PERFORMANCE_COL}.parquet"
echo "  $TASK_OUT_DIR/shortlist_per_query_${PERFORMANCE_COL}.parquet"
echo "  $TASK_OUT_DIR/plots/"