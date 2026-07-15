#!/bin/bash
#SBATCH --job-name=llama-eval
#SBATCH --account=rrg-annielee
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=artifacts/logs/%x-%A_%a.out
#SBATCH --error=artifacts/logs/%x-%A_%a.err

set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "Usage: sbatch --array=... $0 TASK TRANSFER_MANIFEST TASK_MANIFEST SHARD_SIZE ARRAY_OFFSET [PYTHON_ARGS...]" >&2
    exit 2
fi

TASK="$1"
TRANSFER_MANIFEST="$2"
TASK_MANIFEST="$3"
SHARD_SIZE="$4"
ARRAY_OFFSET="$5"
shift 5

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}"

module load python/3.11
module load gcc
module load cuda
module load arrow

VENV_DIR="${VENV_PATH:-$SLURM_SUBMIT_DIR/venv}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment activation script not found:" >&2
    echo "  $VENV_DIR/bin/activate" >&2
    echo "Set VENV_PATH or create the repository virtual environment." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

for MANIFEST in "$TRANSFER_MANIFEST" "$TASK_MANIFEST"; do
    if [[ ! -f "$MANIFEST" ]]; then
        echo "Language manifest not found: $MANIFEST" >&2
        exit 1
    fi
done

if [[ ! "$SHARD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "SHARD_SIZE must be a positive integer; received: $SHARD_SIZE" >&2
    exit 2
fi

if [[ ! "$ARRAY_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "ARRAY_OFFSET must be a non-negative integer; received: $ARRAY_OFFSET" >&2
    exit 2
fi

case "$TASK" in
    sib200)
        PYTHON_SCRIPT="scripts/finetune_llama_sib200.py"
        ;;
    wikiann)
        PYTHON_SCRIPT="scripts/finetune_llama_wikiann.py"
        ;;
    xnli)
        PYTHON_SCRIPT="scripts/finetune_llama_xnli.py"
        ;;
    *)
        echo "Unknown task: $TASK" >&2
        exit 2
        ;;
esac

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Python entry point not found: $PYTHON_SCRIPT" >&2
    exit 1
fi

TRANSFER_COUNT="$(awk 'END {print NR}' "$TRANSFER_MANIFEST")"
TASK_LANGUAGE_COUNT="$(awk 'END {print NR}' "$TASK_MANIFEST")"

if (( TRANSFER_COUNT == 0 || TASK_LANGUAGE_COUNT == 0 )); then
    echo "A language manifest is empty." >&2
    exit 1
fi

SHARD_COUNT=$(((TASK_LANGUAGE_COUNT + SHARD_SIZE - 1) / SHARD_SIZE))

LOCAL_ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This file must be submitted as a job array.}"
GLOBAL_EVAL_INDEX=$((ARRAY_OFFSET + LOCAL_ARRAY_INDEX))

# Each global index represents:
#   one transfer language
#   one shard of task languages
TRANSFER_INDEX=$((GLOBAL_EVAL_INDEX / SHARD_COUNT))
SHARD_INDEX=$((GLOBAL_EVAL_INDEX % SHARD_COUNT))

if (( TRANSFER_INDEX >= TRANSFER_COUNT )); then
    echo "Evaluation index $GLOBAL_EVAL_INDEX is outside the manifest range." >&2
    exit 2
fi

TRANSFER_LANGUAGE="$(
    sed -n "$((TRANSFER_INDEX + 1))p" "$TRANSFER_MANIFEST"
)"
TRANSFER_LANGUAGE="${TRANSFER_LANGUAGE%$'\r'}"

FIRST_TASK_INDEX=$((SHARD_INDEX * SHARD_SIZE))
LAST_TASK_INDEX=$((FIRST_TASK_INDEX + SHARD_SIZE - 1))

if (( LAST_TASK_INDEX >= TASK_LANGUAGE_COUNT )); then
    LAST_TASK_INDEX=$((TASK_LANGUAGE_COUNT - 1))
fi

FIRST_LINE=$((FIRST_TASK_INDEX + 1))
LAST_LINE=$((LAST_TASK_INDEX + 1))

mapfile -t TASK_LANGUAGES < <(
    sed -n "${FIRST_LINE},${LAST_LINE}p" "$TASK_MANIFEST"
)

for INDEX in "${!TASK_LANGUAGES[@]}"; do
    TASK_LANGUAGES[$INDEX]="${TASK_LANGUAGES[$INDEX]%$'\r'}"
done

if (( ${#TASK_LANGUAGES[@]} == 0 )); then
    echo "No task languages found for shard $SHARD_INDEX." >&2
    exit 2
fi

mkdir -p artifacts/logs

echo "Job Array ID / Job ID: ${SLURM_ARRAY_JOB_ID:-none} / $SLURM_JOB_ID"
echo "Array Task ID: $LOCAL_ARRAY_INDEX"
echo "Global evaluation index: $GLOBAL_EVAL_INDEX"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Task: $TASK"
echo "Transfer language: $TRANSFER_LANGUAGE"
echo "Evaluation shard: $((SHARD_INDEX + 1)) / $SHARD_COUNT"
echo "Task languages: ${TASK_LANGUAGES[*]}"

python -u "$PYTHON_SCRIPT" \
    --lang "$TRANSFER_LANGUAGE" \
    --eval-only \
    --task-langs "${TASK_LANGUAGES[@]}" \
    "$@"