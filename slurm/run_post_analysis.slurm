#!/bin/bash
#SBATCH --job-name=langrank-post
#SBATCH --account=def-mere
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON_MODULE="${PYTHON_MODULE:-python}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-venv}"

OUT_DIR="${OUT_DIR:-post-analysis/outputs}"
N_BOOTSTRAP="${N_BOOTSTRAP:-20000}"
SEED="${SEED:-42}"
INCLUDE_NNRANK="${INCLUDE_NNRANK:-1}"
MIN_CTC_TARGETS="${MIN_CTC_TARGETS:-50}"
MARKDOWN_MAX_ROWS="${MARKDOWN_MAX_ROWS:-0}"
MARKDOWN_DIGITS="${MARKDOWN_DIGITS:-3}"

export PYTHONPATH="$SLURM_SUBMIT_DIR/post-analysis:$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p artifacts/logs "$OUT_DIR" "$OUT_DIR/tables" "$OUT_DIR/markdown"

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

python -c "import numpy; print('numpy ok')"
python -c "import pandas; print('pandas ok')"

python - <<'PY'
try:
    import statsmodels
    print("statsmodels ok")
except ImportError:
    print("WARNING: statsmodels not installed. LMM table will contain an error row.")
PY

EXTRA_FLAGS=()
if [[ "$INCLUDE_NNRANK" == "1" ]]; then
  EXTRA_FLAGS+=(--include-nnrank)
fi

python post-analysis/run_revision_analysis.py \
  --root . \
  --outdir "$OUT_DIR" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED" \
  --min-ctc-targets "$MIN_CTC_TARGETS" \
  "${EXTRA_FLAGS[@]}"

python post-analysis/write_markdown_tables.py \
  --tables-dir "$OUT_DIR/tables" \
  --outdir "$OUT_DIR/markdown" \
  --max-rows "$MARKDOWN_MAX_ROWS" \
  --digits "$MARKDOWN_DIGITS"

echo "Finished post-analysis."
echo "CSV tables: $OUT_DIR/tables"
echo "Markdown tables: $OUT_DIR/markdown"
echo "Combined Markdown: $OUT_DIR/markdown/all_tables.md"