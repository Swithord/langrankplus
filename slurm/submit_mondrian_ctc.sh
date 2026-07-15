#!/bin/bash

set -euo pipefail

REPOSITORY_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"
cd "$REPOSITORY_ROOT"

mkdir -p artifacts/logs artifacts/mondrian_ctc

EVAL_SUBMIT="$(sbatch --parsable slurm/run_mondrian_ctc.slurm)"
EVAL_JOB_ID="${EVAL_SUBMIT%%;*}"

TABLE_SUBMIT="$(
  sbatch --parsable \
    --dependency="afterok:${EVAL_JOB_ID}" \
    slurm/run_mondrian_ctc_table.slurm
)"
TABLE_JOB_ID="${TABLE_SUBMIT%%;*}"

echo "Mondrian CTC array job: ${EVAL_JOB_ID}"
echo "Dependent table job:    ${TABLE_JOB_ID}"
echo "Monitor with: squeue -j ${EVAL_JOB_ID},${TABLE_JOB_ID}"