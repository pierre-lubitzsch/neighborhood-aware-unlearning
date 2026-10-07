#!/usr/bin/env bash
# Train a residual quantizer on item embeddings and assign semantic IDs.
# QUANTIZER is rkmeans (RQ-KMeans, default) or rqvae (RQ-VAE). The output tensor has
# shape [4, N]: three codebook levels plus one de-duplication digit.
#
#   bash scripts/pipeline/train_sid.sh <data_dir> <item_embeddings.pt> <out_sids.pt> [training overrides...]
set -euo pipefail
DATA_DIR="${1:?data_dir}"
EMB="${2:?item embeddings .pt}"
OUT="${3:?output .pt}"
QUANTIZER="${QUANTIZER:-rkmeans}"
LEVELS="${LEVELS:-3}"
WIDTH="${WIDTH:-256}"
RUN_DIR="${RUN_DIR:-outputs/sid/$(basename "${DATA_DIR}")_${QUANTIZER}}"
COMMON=("data_dir=${DATA_DIR}" "embedding_path='${EMB}'" embedding_dim=2048
        "num_hierarchies=${LEVELS}" "codebook_width=${WIDTH}")

python -m src.train experiment="${QUANTIZER}_train_flat" "${COMMON[@]}" \
  hydra.run.dir="${RUN_DIR}/train" \
  callbacks.model_checkpoint.dirpath="${RUN_DIR}/train/checkpoints" "${@:4}"
CKPT="$(ls -t "${RUN_DIR}"/train/checkpoints/*.ckpt | head -1)"

mkdir -p "${RUN_DIR}/infer/pickle"
python -m src.inference experiment="${QUANTIZER}_inference_flat" "${COMMON[@]}" \
  "ckpt_path='${CKPT}'" \
  hydra.run.dir="${RUN_DIR}/infer" \
  callbacks.pickle_writer.should_merge_files_on_main=false

python -m scripts.merge_predictions sids "${RUN_DIR}/infer/pickle" "${OUT}"
echo "codebook checkpoint: ${CKPT}"
