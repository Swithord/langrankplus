#!/bin/bash
#SBATCH --job-name=langrank-infer
#SBATCH --account=def-annielee
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON_MODULE="${PYTHON_MODULE:-python}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-venv}"

CSVS="${CSVS:-data/new_task.csv}"
DATASET_NAMES="${DATASET_NAMES:-}"
FEATURES="${FEATURES:-new_gen new_typ new_geo script}"

TARGET_COL="${TARGET_COL:-task_lang}"
SOURCE_COL="${SOURCE_COL:-transfer_lang}"
DATASET_COL="${DATASET_COL:-dataset}"

NORMALIZER="${NORMALIZER:-minmax}"
METHOD="${METHOD:-composite_equal}"
SINGLE_FEATURE="${SINGLE_FEATURE:-}"
CALIBRATION_JSON="${CALIBRATION_JSON:-}"

OUT_DIR="${OUT_DIR:-artifacts/inference}"

read -r -a CSV_ARRAY <<< "$CSVS"
read -r -a FEATURE_ARRAY <<< "$FEATURES"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p artifacts/logs "$OUT_DIR"

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

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "which python: $(which python)"
python --version

echo "Inference"
echo "CSVS: $CSVS"
echo "FEATURES: $FEATURES"
echo "METHOD: $METHOD"
echo "OUT_DIR: $OUT_DIR"

EXTRA_FLAGS=()

if [[ -n "$DATASET_NAMES" ]]; then
  read -r -a DATASET_NAME_ARRAY <<< "$DATASET_NAMES"
  EXTRA_FLAGS+=(--dataset_names "${DATASET_NAME_ARRAY[@]}")
fi

if [[ "$METHOD" == "single" ]]; then
  if [[ -z "$SINGLE_FEATURE" ]]; then
    echo "SINGLE_FEATURE is required when METHOD=single"
    exit 1
  fi
  EXTRA_FLAGS+=(--single_feature "$SINGLE_FEATURE")
fi

if [[ "$METHOD" == "calibrated" ]]; then
  if [[ -z "$CALIBRATION_JSON" ]]; then
    echo "CALIBRATION_JSON is required when METHOD=calibrated"
    exit 1
  fi
  EXTRA_FLAGS+=(--calibration_json "$CALIBRATION_JSON")
fi

python main.py infer \
  --csv "${CSV_ARRAY[@]}" \
  --features "${FEATURE_ARRAY[@]}" \
  --target_col "$TARGET_COL" \
  --source_col "$SOURCE_COL" \
  --dataset_col "$DATASET_COL" \
  --normalizer "$NORMALIZER" \
  --method "$METHOD" \
  --outdir "$OUT_DIR" \
  "${EXTRA_FLAGS[@]}"

echo "Finished inference"
echo "Outputs are in $OUT_DIR"