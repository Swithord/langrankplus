#!/bin/bash
#SBATCH --job-name=mondrian-table
#SBATCH --account=def-mere
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON_MODULE="${PYTHON_MODULE:-python}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-$SLURM_SUBMIT_DIR/venv}"
ROOT="${ROOT:-artifacts/mondrian_ctc}"
OUTDIR="${OUTDIR:-artifacts/mondrian_ctc/table}"
PERFORMANCE_COL="${PERFORMANCE_COL:-f1_score}"

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

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTDIR"

python post-analysis/build_mondrian_ctc_table.py \
  --root "$ROOT" \
  --outdir "$OUTDIR" \
  --performance_col "$PERFORMANCE_COL"
