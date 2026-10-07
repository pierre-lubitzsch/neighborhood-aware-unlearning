#!/usr/bin/env bash
# Evaluate a checkpoint on the test split: NDCG@10 / Recall@10 and, given a forget
# manifest, the exposure metrics (SH@10 for spam; SHF@10 / SHR@10 for the requesting
# and remaining users in the unwanted-item scenario, reported as UHF / UHR).
#
#   bash scripts/pipeline/evaluate.sh <ckpt> <clean_data_dir> <sids.pt> <out_dir> [forget_manifest.json] [seed]
set -euo pipefail
CKPT="${1:?checkpoint}"
DATA_DIR="${2:?clean data_dir}"
SID="${3:?semantic id tensor}"
OUT="${4:?output dir}"
MANIFEST="${5:-}"
SEED="${6:-2}"
MODEL="${MODEL:-tiger}"
EXTRA=()
[ -n "${MANIFEST}" ] && EXTRA+=("spam_forget_manifest='${MANIFEST}'")
[ -n "${FILTER_MASK:-}" ] && EXTRA+=("decode_filter_mask='${FILTER_MASK}'")

python -m scripts.eval_ckpt_on_test experiment="${MODEL}_train_flat" \
  data_dir="${DATA_DIR}" "semantic_id_path='${SID}'" "ckpt_path='${CKPT}'" \
  num_hierarchies=4 seed="${SEED}" train=False test=True \
  trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 \
  trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null \
  ${EXTRA[@]+"${EXTRA[@]}"} hydra.run.dir="${OUT}"
echo "metrics: ${OUT}/csv/version_0/metrics.csv"
