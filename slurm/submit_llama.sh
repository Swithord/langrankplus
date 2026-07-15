#!/bin/bash

set -euo pipefail

REPOSITORY_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
        pwd
)"
cd "$REPOSITORY_ROOT"

SELECTION="${1:-all}"

if [[ $# -gt 0 ]]; then
    shift
fi

PYTHON_ARGS=("$@")

case "$SELECTION" in
    all)
        TASKS=(sib200 wikiann xnli)
        ;;
    sib200|wikiann|xnli)
        TASKS=("$SELECTION")
        ;;
    *)
        echo "Usage: $0 [all|sib200|wikiann|xnli] [PYTHON_ARGS...]" >&2
        exit 2
        ;;
esac

# STAGE=all submits training followed by dependent evaluation.
# STAGE=train submits training only.
# STAGE=eval submits evaluation only and assumes adapters already exist.
STAGE="${STAGE:-all}"

case "$STAGE" in
    all|train|eval)
        ;;
    *)
        echo "STAGE must be one of: all, train, eval" >&2
        exit 2
        ;;
esac

EVAL_SHARD_SIZE="${EVAL_SHARD_SIZE:-32}"

# Arrays are submitted in batches of 30, matching the configuration already
# known to work on the cluster. There is no Slurm %N concurrency throttle.
TRAIN_ARRAY_CHUNK_SIZE="${TRAIN_ARRAY_CHUNK_SIZE:-30}"
EVAL_ARRAY_CHUNK_SIZE="${EVAL_ARRAY_CHUNK_SIZE:-30}"

validate_positive_integer() {
    local NAME="$1"
    local VALUE="$2"

    if [[ ! "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
        echo "$NAME must be a positive integer; received: $VALUE" >&2
        exit 2
    fi
}

validate_positive_integer \
    "EVAL_SHARD_SIZE" \
    "$EVAL_SHARD_SIZE"

validate_positive_integer \
    "TRAIN_ARRAY_CHUNK_SIZE" \
    "$TRAIN_ARRAY_CHUNK_SIZE"

validate_positive_integer \
    "EVAL_ARRAY_CHUNK_SIZE" \
    "$EVAL_ARRAY_CHUNK_SIZE"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available. Run this script on a Slurm login node." >&2
    exit 1
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
ARTIFACT_DIR="$REPOSITORY_ROOT/slurm/artifacts/llama"
MANIFEST_DIR="$ARTIFACT_DIR/manifests/$RUN_ID"

# Slurm opens output files before the batch script starts, so this directory
# must exist before sbatch is called.
mkdir -p \
    "$REPOSITORY_ROOT/artifacts/logs" \
    "$MANIFEST_DIR"

make_manifest() {
    local DATA_DIR="$1"
    local MANIFEST="$2"

    if [[ ! -d "$DATA_DIR" ]]; then
        echo "Missing dataset directory: $DATA_DIR" >&2
        return 1
    fi

    while IFS= read -r -d '' LANGUAGE_DIR; do
        if [[ -f "$LANGUAGE_DIR/dataset_dict.json" ]]; then
            basename "$LANGUAGE_DIR"
        fi
    done < <(
        find "$DATA_DIR" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -print0
    ) | LC_ALL=C sort > "$MANIFEST"

    local LANGUAGE_COUNT
    LANGUAGE_COUNT="$(awk 'END {print NR}' "$MANIFEST")"

    if (( LANGUAGE_COUNT == 0 )); then
        echo "No DatasetDict language folders found in $DATA_DIR" >&2
        return 1
    fi
}

job_id_from_submission() {
    local SUBMISSION="$1"

    # Some Slurm installations return JOB_ID;CLUSTER with --parsable.
    printf '%s\n' "${SUBMISSION%%;*}"
}

# AFTEROK_JOB_ID can optionally make the first submitted batch depend on an
# existing Slurm job. When all tasks are selected, their pipelines are chained.
PIPELINE_DEPENDENCY="${AFTEROK_JOB_ID:-}"

for TASK in "${TASKS[@]}"; do
    DATA_DIR="$REPOSITORY_ROOT/data/$TASK"
    MANIFEST="$MANIFEST_DIR/${TASK}.txt"

    make_manifest "$DATA_DIR" "$MANIFEST"

    LANGUAGE_COUNT="$(awk 'END {print NR}' "$MANIFEST")"
    TRAIN_JOB_ID=""

    echo
    echo "Task: $TASK"
    echo "Languages: $LANGUAGE_COUNT"
    echo "Manifest: $MANIFEST"

    if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
        TRAIN_DEPENDENCY="$PIPELINE_DEPENDENCY"
        TRAIN_ARRAY_OFFSET=0
        TRAIN_BATCH_NUMBER=0

        while (( TRAIN_ARRAY_OFFSET < LANGUAGE_COUNT )); do
            TRAIN_REMAINING=$((LANGUAGE_COUNT - TRAIN_ARRAY_OFFSET))
            TRAIN_ARRAY_LENGTH="$TRAIN_ARRAY_CHUNK_SIZE"

            if (( TRAIN_REMAINING < TRAIN_ARRAY_LENGTH )); then
                TRAIN_ARRAY_LENGTH="$TRAIN_REMAINING"
            fi

            TRAIN_LOCAL_ARRAY_END=$((TRAIN_ARRAY_LENGTH - 1))
            TRAIN_BATCH_NUMBER=$((TRAIN_BATCH_NUMBER + 1))
            TRAIN_DEPENDENCY_ARGS=()

            if [[ -n "$TRAIN_DEPENDENCY" ]]; then
                TRAIN_DEPENDENCY_ARGS=(
                    --dependency="afterok:$TRAIN_DEPENDENCY"
                )
            fi

            TRAIN_SUBMISSION="$(
                sbatch \
                    --parsable \
                    --job-name="llama-${TASK}-train-${TRAIN_BATCH_NUMBER}" \
                    --array="0-${TRAIN_LOCAL_ARRAY_END}" \
                    "${TRAIN_DEPENDENCY_ARGS[@]}" \
                    slurm/run_llama_train.slurm \
                    "$TASK" \
                    "$MANIFEST" \
                    "$TRAIN_ARRAY_OFFSET" \
                    "${PYTHON_ARGS[@]}"
            )"

            TRAIN_JOB_ID="$(
                job_id_from_submission "$TRAIN_SUBMISSION"
            )"

            echo \
                "Training batch $TRAIN_BATCH_NUMBER submitted:" \
                "$TRAIN_JOB_ID" \
                "(offset=$TRAIN_ARRAY_OFFSET, jobs=$TRAIN_ARRAY_LENGTH)"

            # Each 30-language batch waits for the preceding batch. Every
            # element within the current batch may run concurrently.
            TRAIN_DEPENDENCY="$TRAIN_JOB_ID"
            TRAIN_ARRAY_OFFSET=$((TRAIN_ARRAY_OFFSET + TRAIN_ARRAY_LENGTH))
        done
    fi

    if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
        # Number of task-language shards evaluated for each transfer language.
        EVALUATION_SHARD_COUNT=$((
            (LANGUAGE_COUNT + EVAL_SHARD_SIZE - 1) /
            EVAL_SHARD_SIZE
        ))

        TOTAL_EVALUATION_JOBS=$((
            LANGUAGE_COUNT *
            EVALUATION_SHARD_COUNT
        ))

        if [[ -n "$TRAIN_JOB_ID" ]]; then
            # Evaluation starts after every training batch succeeds.
            EVAL_DEPENDENCY="$TRAIN_JOB_ID"
        else
            EVAL_DEPENDENCY="$PIPELINE_DEPENDENCY"
        fi

        ARRAY_OFFSET=0
        EVAL_BATCH_NUMBER=0

        while (( ARRAY_OFFSET < TOTAL_EVALUATION_JOBS )); do
            REMAINING=$((TOTAL_EVALUATION_JOBS - ARRAY_OFFSET))
            ARRAY_LENGTH="$EVAL_ARRAY_CHUNK_SIZE"

            if (( REMAINING < ARRAY_LENGTH )); then
                ARRAY_LENGTH="$REMAINING"
            fi

            LOCAL_ARRAY_END=$((ARRAY_LENGTH - 1))
            EVAL_BATCH_NUMBER=$((EVAL_BATCH_NUMBER + 1))
            EVAL_DEPENDENCY_ARGS=()

            if [[ -n "$EVAL_DEPENDENCY" ]]; then
                EVAL_DEPENDENCY_ARGS=(
                    --dependency="afterok:$EVAL_DEPENDENCY"
                )
            fi

            EVAL_SUBMISSION="$(
                sbatch \
                    --parsable \
                    --job-name="llama-${TASK}-eval-${EVAL_BATCH_NUMBER}" \
                    --array="0-${LOCAL_ARRAY_END}" \
                    "${EVAL_DEPENDENCY_ARGS[@]}" \
                    slurm/run_llama_eval_shard.slurm \
                    "$TASK" \
                    "$MANIFEST" \
                    "$MANIFEST" \
                    "$EVAL_SHARD_SIZE" \
                    "$ARRAY_OFFSET" \
                    "${PYTHON_ARGS[@]}"
            )"

            EVAL_JOB_ID="$(
                job_id_from_submission "$EVAL_SUBMISSION"
            )"

            echo \
                "Evaluation batch $EVAL_BATCH_NUMBER submitted:" \
                "$EVAL_JOB_ID" \
                "(offset=$ARRAY_OFFSET, jobs=$ARRAY_LENGTH)"

            # Each evaluation batch waits for the preceding batch. Every
            # element within the current batch may run concurrently.
            EVAL_DEPENDENCY="$EVAL_JOB_ID"
            ARRAY_OFFSET=$((ARRAY_OFFSET + ARRAY_LENGTH))
        done

        PIPELINE_DEPENDENCY="$EVAL_DEPENDENCY"
    else
        PIPELINE_DEPENDENCY="$TRAIN_JOB_ID"
    fi
done

echo
echo "Submission complete."
echo "Final pipeline job: $PIPELINE_DEPENDENCY"
echo "Results will be updated in:"

for TASK in "${TASKS[@]}"; do
    echo "  data/${TASK}_llama.csv"
done