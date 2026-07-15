#!/bin/bash
#SBATCH --job-name=langrank-cross-task-post
#SBATCH --account=def-mere
#SBATCH --time=00:30:00
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

ROOT_DIR="${ROOT_DIR:-$SLURM_SUBMIT_DIR}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/artifacts/cross_task_generalization}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/post-analysis/cross-task-outputs}"
N_BOOTSTRAP="${N_BOOTSTRAP:-20000}"
SEED="${SEED:-42}"

export PYTHONPATH="$ROOT_DIR/post-analysis:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "$PYTHON_MODULE" ]]; then
  module load "$PYTHON_MODULE"
fi

for mod in $EXTRA_MODULES; do
  module load "$mod"
done

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "ERROR: no virtualenv found at $VENV_PATH" >&2
  exit 1
fi

source "$VENV_PATH/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p artifacts/logs
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Started: $(date)"
echo "PWD: $(pwd)"
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version

echo "Root directory: $ROOT_DIR"
echo "Artifact root: $ARTIFACT_ROOT"
echo "Output directory: $OUT_DIR"
echo "Bootstrap replicates: $N_BOOTSTRAP"

python -m py_compile \
  post-analysis/analysis_utils.py \
  post-analysis/cross_task_analysis_utils.py \
  post-analysis/run_cross_task_revision_analysis.py

python - <<'PY'
import numpy
import pandas
import analysis_utils
import cross_task_analysis_utils

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("analysis_utils:", analysis_utils.__file__)
print("cross_task_analysis_utils:", cross_task_analysis_utils.__file__)

try:
    import statsmodels
    print("statsmodels:", statsmodels.__version__)
except ImportError:
    print("WARNING: statsmodels is not installed; the LMM table will contain an error row.")
PY

python -u post-analysis/run_cross_task_revision_analysis.py \
  --root "$ROOT_DIR" \
  --artifact-root "$ARTIFACT_ROOT" \
  --outdir "$OUT_DIR" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED"

N_FILES="$(find "$OUT_DIR" -type f | wc -l)"
if [[ "$N_FILES" -eq 0 ]]; then
  echo "ERROR: analysis completed without generating files." >&2
  exit 1
fi

echo "Finished: $(date)"
echo "Generated $N_FILES files:"
find "$OUT_DIR" -maxdepth 4 -type f | sort
