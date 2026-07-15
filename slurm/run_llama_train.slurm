#!/bin/bash
#SBATCH --job-name=llama-train
#SBATCH --account=rrg-annielee
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=artifacts/logs/%x-%A_%a.out
#SBATCH --error=artifacts/logs/%x-%A_%a.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch --array=... $0 TASK TRANSFER_MANIFEST ARRAY_OFFSET [PYTHON_ARGS...]" >&2
    exit 2
fi

TASK="$1"
TRANSFER_MANIFEST="$2"
ARRAY_OFFSET="$3"
shift 3

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

if [[ ! -f "$TRANSFER_MANIFEST" ]]; then
    echo "Transfer-language manifest not found: $TRANSFER_MANIFEST" >&2
    exit 1
fi

if [[ ! "$ARRAY_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "ARRAY_OFFSET must be a non-negative integer; received: $ARRAY_OFFSET" >&2
    exit 2
fi

LOCAL_ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This file must be submitted as a job array.}"
GLOBAL_ARRAY_INDEX=$((ARRAY_OFFSET + LOCAL_ARRAY_INDEX))

TRANSFER_LANGUAGE="$(
    sed -n "$((GLOBAL_ARRAY_INDEX + 1))p" "$TRANSFER_MANIFEST"
)"
TRANSFER_LANGUAGE="${TRANSFER_LANGUAGE%$'\r'}"

if [[ -z "$TRANSFER_LANGUAGE" ]]; then
    echo "No language at global array index $GLOBAL_ARRAY_INDEX in $TRANSFER_MANIFEST" >&2
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

ADAPTER_DIR="models/$TASK/llama/$TRANSFER_LANGUAGE/adapter"

# Completed languages are skipped when the submission is rerun.
if [[ "${SKIP_TRAINED:-1}" == "1" &&
      -f "$ADAPTER_DIR/adapter_config.json" ]]; then
    echo "A completed adapter already exists at $ADAPTER_DIR; skipping training."
    exit 0
fi

# Resume from the newest Trainer checkpoint after a timeout or interrupted job.
RESUME_ARGS=()
RESUME_WAS_SUPPLIED=0

for ARGUMENT in "$@"; do
    case "$ARGUMENT" in
        --resume-from-checkpoint|--resume-from-checkpoint=*)
            RESUME_WAS_SUPPLIED=1
            ;;
    esac
done

if [[ "${AUTO_RESUME:-1}" == "1" &&
      "$RESUME_WAS_SUPPLIED" == "0" ]]; then
    CHECKPOINT_ROOT="models/$TASK/llama/$TRANSFER_LANGUAGE/checkpoints"

    if [[ -d "$CHECKPOINT_ROOT" ]]; then
        LATEST_CHECKPOINT="$(
            {
                find "$CHECKPOINT_ROOT" \
                    -mindepth 1 \
                    -maxdepth 1 \
                    -type d \
                    -name 'checkpoint-*' \
                    -print || true
            } |
                sort -V |
                tail -n 1
        )"

        if [[ -n "$LATEST_CHECKPOINT" ]]; then
            RESUME_ARGS=(
                --resume-from-checkpoint "$LATEST_CHECKPOINT"
            )
        fi
    fi
fi

mkdir -p artifacts/logs

echo "Job Array ID / Job ID: ${SLURM_ARRAY_JOB_ID:-none} / $SLURM_JOB_ID"
echo "Array Task ID: $LOCAL_ARRAY_INDEX"
echo "Global training index: $GLOBAL_ARRAY_INDEX"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Task: $TASK"
echo "Transfer language: $TRANSFER_LANGUAGE"

if (( ${#RESUME_ARGS[@]} > 0 )); then
    echo "Resuming from: ${RESUME_ARGS[1]}"
fi

# run_task evaluates after training. Limit this immediate evaluation to the
# transfer language. The evaluation Slurm arrays handle the full evaluation.
python -u "$PYTHON_SCRIPT" \
    --lang "$TRANSFER_LANGUAGE" \
    --task-langs "$TRANSFER_LANGUAGE" \
    "${RESUME_ARGS[@]}" \
    "$@"