#!/bin/bash
set -euo pipefail

SLURM_SCRIPT="${SLURM_SCRIPT:-slurm/run_eval.slurm}"

DATASETS=(
  "masakhaner_mt5"
  "masakhaner_xlm-r"
  "taxi1500_mt5"
  "taxi1500_xlm-r"
  "tydiqa_mt5"
  "tydiqa_xlm-r"
  "ud_pos_mt5"
  "ud_pos_xlm-r"
  "wikiann_mt5"
  "wikiann_xlm-r"
  "xnli_mt5"
  "xnli_xlm-r"
  "sib200_xlm-r"
)

mkdir -p artifacts

for name in "${DATASETS[@]}"; do
  csv="data/${name}.csv"
  outdir="artifacts/${name}"

  if [[ ! -f "$csv" ]]; then
    echo "Skipping ${name}: missing ${csv}"
    continue
  fi

  mkdir -p "$outdir"

  echo "Submitting ${name}"
  echo "  CSV:     ${csv}"
  echo "  OUT_DIR: ${outdir}"

  CSVS="$csv" \
  PERFORMANCE_COLS="f1_score" \
  OUT_DIR="$outdir" \
  INCLUDE_BUDGET_METRICS=1 \
  BUDGET_KS="3 5" \
  NEAR_ORACLE_STD_MULTIPLIER="0.5" \
  INCLUDE_IR_METRICS=1 \
  IR_CUTOFFS="1 3" \
  sbatch "$SLURM_SCRIPT"
done

echo "Submitted all available dataset/model evaluations."