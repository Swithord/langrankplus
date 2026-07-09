#!/bin/bash
#SBATCH --job-name=langrank-mdtables
#SBATCH --account=def-mere
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON_MODULE="${PYTHON_MODULE:-python}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-venv}"

OUT_DIR="${OUT_DIR:-post-analysis/outputs}"
TABLES_DIR="${TABLES_DIR:-$OUT_DIR/tables}"
MARKDOWN_DIR="${MARKDOWN_DIR:-$OUT_DIR/markdown}"
MARKDOWN_MAX_ROWS="${MARKDOWN_MAX_ROWS:-0}"
MARKDOWN_DIGITS="${MARKDOWN_DIGITS:-3}"

export PYTHONPATH="$SLURM_SUBMIT_DIR/post-analysis:$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p artifacts/logs "$MARKDOWN_DIR"

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

python post-analysis/write_markdown_tables.py \
  --tables-dir "$TABLES_DIR" \
  --outdir "$MARKDOWN_DIR" \
  --max-rows "$MARKDOWN_MAX_ROWS" \
  --digits "$MARKDOWN_DIGITS"

echo "Finished Markdown table export."
echo "Markdown tables: $MARKDOWN_DIR"
echo "Combined Markdown: $MARKDOWN_DIR/all_tables.md"