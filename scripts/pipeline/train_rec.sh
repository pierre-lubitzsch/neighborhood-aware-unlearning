#!/usr/bin/env bash
# Train the generative recommender. The same command gives the original model
# (clean data for the unwanted-item scenario, poisoned data for spam) and the
# retrained reference (clean data for spam, the *_retrain dir for unwanted items).
# MODEL is tiger (default) or letter.
#
#   bash scripts/pipeline/train_rec.sh <data_dir> <sids.pt> <run_dir> [seed]
set -euo pipefail
DATA_DIR="${1:?data_dir}"
SID="${2:?semantic id tensor}"
RUN_DIR="${3:?run dir}"
SEED="${4:-2}"
MODEL="${MODEL:-tiger}"

python -m src.train experiment="${MODEL}_train_flat" \
  data_dir="${DATA_DIR}" "semantic_id_path='${SID}'" num_hierarchies=4 seed="${SEED}" \
  hydra.run.dir="${RUN_DIR}" "${@:5}"
