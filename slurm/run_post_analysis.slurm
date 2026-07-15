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

export PYTHONPATH="$SLURM_SUBMIT_DIR/post-analysis:$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR"

if [[ -n "$PYTHON_MODULE" ]]; then
  module load "$PYTHON_MODULE"
fi

for mod in $EXTRA_MODULES; do
  module load "$mod"
done

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "No virtualenv found at: $VENV_PATH" >&2
  exit 1
fi

source "$VENV_PATH/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Started: $(date)"
echo "PWD: $(pwd)"
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version

python -m py_compile \
  post-analysis/analysis_utils.py \
  post-analysis/run_revision_analysis.py

python - <<'PY'
import numpy
import pandas

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)

try:
    import statsmodels
    print("statsmodels:", statsmodels.__version__)
except ImportError:
    print("WARNING: statsmodels is not installed.")
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

echo "Finished: $(date)"
echo
echo "Generated files:"
find "$OUT_DIR" -maxdepth 4 -type f | sort