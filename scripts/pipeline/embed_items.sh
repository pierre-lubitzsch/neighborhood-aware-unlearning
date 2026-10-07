#!/usr/bin/env bash
# Encode the item texts of a dataset with a pretrained text encoder (flan-t5-xl).
#
#   bash scripts/pipeline/embed_items.sh <data_dir> <out.pt>
set -euo pipefail
DATA_DIR="${1:?data_dir}"
OUT="${2:?output .pt}"
RUN_DIR="${RUN_DIR:-outputs/embeddings/$(basename "${DATA_DIR}")}"
mkdir -p "${RUN_DIR}/pickle"

python -m src.inference \
  experiment=sem_embeds_inference_flat \
  data_dir="${DATA_DIR}" \
  hydra.run.dir="${RUN_DIR}" \
  callbacks.pickle_writer.should_merge_files_on_main=false

python -m scripts.merge_predictions embeddings "${RUN_DIR}/pickle" "${OUT}"
