#!/usr/bin/env bash
# LETTER semantic IDs: collaborative-filtering item embeddings (SASRec), a LETTER
# tokenizer (RQ-VAE with collaborative and diversity regularization), and
# collision-free ID assignment. Output shape [4, N].
#
#   bash scripts/pipeline/letter_sid.sh <data_dir> <item_embeddings.pt> <out_sids.pt> [tokenizer overrides...]
set -euo pipefail
DATA_DIR="${1:?data_dir}"
EMB="${2:?item embeddings .pt}"
OUT="${3:?output .pt}"
RUN_DIR="${RUN_DIR:-outputs/sid/$(basename "${DATA_DIR}")_letter}"
CF="${RUN_DIR}/cf64.pt"
N_ITEMS="$(python -c "import torch; t=torch.load('${EMB}', map_location='cpu'); print((t['embeddings'] if isinstance(t, dict) else t).shape[0])")"
EPOCHS="${LETTER_EPOCHS:-10000}"
BATCH=2048
STEPS="$(( (N_ITEMS + BATCH - 1) / BATCH * EPOCHS ))"
mkdir -p "${RUN_DIR}"

python -m scripts.train_cf_embeddings --data-dir "${DATA_DIR}" --out "${CF}" \
  --dim 64 --epochs "${CF_EPOCHS:-200}" --seed 42 --num-items "${N_ITEMS}"

python -m src.train experiment=letter_rqvae_train_flat \
  data_dir="${DATA_DIR}" "embedding_path='${EMB}'" "cf_embedding_path='${CF}'" \
  embedding_dim=2048 num_hierarchies=4 codebook_width=256 \
  letter_alpha=0.01 letter_beta=0.0001 letter_sinkhorn_epsilon=0.003 \
  trainer.max_steps="${STEPS}" \
  hydra.run.dir="${RUN_DIR}/train" \
  callbacks.model_checkpoint.dirpath="${RUN_DIR}/train/checkpoints" "${@:4}"
CKPT="$(ls -t "${RUN_DIR}"/train/checkpoints/*.ckpt | head -1)"

python -m scripts.assign_sids_letter --codebook_ckpt "${CKPT}" --embedding_path "${EMB}" \
  --out "${OUT}" --sk_epsilon 0.003 --max_rounds 20 --max_collision_rate 0.01 --overwrite
echo "codebook checkpoint: ${CKPT}"
