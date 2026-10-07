#!/usr/bin/env bash
# Spam scenario: inject a bandwagon attack promoting one target item and split the
# training data into the forget set (fake users) and the retain set.
# STRATEGY selects the target: unpopular (bottom 20% by interactions), mid (30-70%),
# or popular (top 5%).
#
#   bash scripts/pipeline/poison.sh <clean_data_dir> <out_dir> [strategy] [seed]
set -euo pipefail
CLEAN="${1:?clean data_dir}"
OUT="${2:?output dir}"
STRATEGY="${3:-unpopular}"
SEED="${4:-2}"

python -m src.data.poisoning.bandwagon \
  --data_dir "${CLEAN}" --out_dir "${OUT}" \
  --method bandwagon --attack bandwagon \
  --target_strategy "${STRATEGY}" \
  --poisoning_ratio "${RATIO:-0.01}" --n_target_items "${N_TARGET:-1}" \
  --placement sprinkled --p_two_targets 0.119 \
  --seed "${SEED}" --rows_per_shard 5000

python -m src.data.unlearning.split_forget_retain \
  --data_dir "${OUT}" --forget_manifest "${OUT}/forget_manifest.json" --segregated-shards
