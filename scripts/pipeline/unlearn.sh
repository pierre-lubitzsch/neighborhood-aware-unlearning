#!/usr/bin/env bash
# Unlearn the forget set of <data_dir> from a trained recommender, then evaluate the
# unlearned model on the clean test split.
#
#   bash scripts/pipeline/unlearn.sh <method> <ckpt> <data_dir> <clean_data_dir> <sids.pt> <item_embeddings.pt> <run_dir> [seed] [extra overrides...]
#
# <data_dir> is a poisoned dir from poison.sh (spam) or a *_sens dir from
# select_unwanted.sh (unwanted items); both contain training_forget/,
# training_retain/ and forget_manifest.json.
#
# Methods (settings as in the paper):
#   nau             NAU for spam removal (lambda_f=0.1, lambda_s=0.01, lambda_n=0.01)
#   nau_unwanted    NAU for unwanted items (lambda_f=0.1, lambda_s=0, lambda_n=-1)
#   finetune        fine-tuning on the retain set
#   forget_only     gradient ascent on the forget set
#   forget_repair   gradient ascent on the forget set plus retain-set repair
#   scif | seif | kookmin | fanchuan | tracer | filter
# Any Hydra override can be appended, e.g. unlearning.lambda_n=-10.0.
# For tracer, pass unlearning.tracer_codebook_ckpt=<quantizer checkpoint from train_sid.sh>.
set -euo pipefail
METHOD="${1:?method}"
CKPT="${2:?checkpoint}"
DATA_DIR="${3:?data_dir with forget/retain split}"
EVAL_DATA_DIR="${4:?clean data_dir}"
SID="${5:?semantic id tensor}"
EMB="${6:?item embeddings .pt}"
RUN_DIR="${7:?run dir}"
SEED="${8:-2}"
EXTRA=("${@:9}")
MODEL="${MODEL:-tiger}"
MANIFEST="${DATA_DIR}/forget_manifest.json"

ALGO=unified
UNIFIED=(unlearning.sep_negatives=forget_target_only unlearning.n_epochs=4 unlearning.adaptive_codes=false)
NEIGHBORHOOD=(unlearning.sep_loss_type=generative unlearning.coherence_loss_type=mass
              unlearning.coherence_neighbor_method=embedding unlearning.coherence_rows=target_only
              unlearning.coherence_embedding_metric=cosine unlearning.neighborhood_count=8)
case "${METHOD}" in
  nau)           OVR=("${UNIFIED[@]}" "${NEIGHBORHOOD[@]}" unlearning.embedding_path="${EMB}" unlearning.lambda_f=0.1 unlearning.lambda_s=0.01 unlearning.lambda_n=0.01) ;;
  nau_unwanted)  OVR=("${UNIFIED[@]}" "${NEIGHBORHOOD[@]}" unlearning.embedding_path="${EMB}" unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=-1.0
                      unlearning.sep_gen_temperature=1.0 unlearning.batch_size_per_device=32) ;;
  finetune)      OVR=("${UNIFIED[@]}" unlearning.lambda_f=0.0 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.lambda_r=1.0) ;;
  forget_only)   OVR=("${UNIFIED[@]}" unlearning.lambda_f=1.0 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.lambda_r=0.0) ;;
  forget_repair) OVR=("${UNIFIED[@]}" unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.lambda_r=1.0) ;;
  scif)          ALGO=scif;     OVR=() ;;
  seif)          ALGO=seif;     OVR=(unlearning.seif_erase_std=0.06) ;;
  kookmin)       ALGO=kookmin;  OVR=(unlearning.kookmin_init_rate=0.001) ;;
  fanchuan)      ALGO=fanchuan; OVR=(unlearning.fanchuan_contrastive_temperature=0.07) ;;
  tracer)        ALGO=tracer;   OVR=(unlearning.embedding_path="${EMB}" unlearning.n_epochs=4 unlearning.tracer_lambda_forget=1.0
                                     unlearning.tracer_lambda_coherence=0.1 unlearning.tracer_phi_lr=0.001) ;;
  filter)        ALGO=filter;   OVR=() ;;
  *) echo "unknown method '${METHOD}'" >&2; exit 1 ;;
esac
if [ "${ALGO}" = unified ]; then EXPERIMENT="${MODEL}_unlearn_unified_sequential"; else EXPERIMENT="${MODEL}_unlearn_scif_sequential"; fi

# Unwanted-item deletions remove single (user, item) interactions inside real sessions.
if python -c "import json,sys; sys.exit(0 if json.load(open('${MANIFEST}')).get('deletion_spec') == 'item_pairs' else 1)"; then
  SCENARIO=(spam_forget_manifest="${MANIFEST}" unlearning.deletion_spec=item_pairs unlearning.include_context_rows=true
            unlearning.request_user_order=sorted unlearning.n_unlearning_chunks=5)
else
  SCENARIO=(unlearning.n_unlearning_chunks=10)
fi

python -m src.unlearn_sequential experiment="${EXPERIMENT}" \
  data_dir="${DATA_DIR}" "semantic_id_path='${SID}'" "ckpt_path='${CKPT}'" num_hierarchies=4 seed="${SEED}" \
  unlearning.algorithm="${ALGO}" unlearning.neighborhood_aware=false unlearning.neighborhood_aware_sample_rate=0.0 \
  unlearning.request_batch_size=1 \
  unlearning_run_tag="$(basename "${RUN_DIR}")" hydra.run.dir="${RUN_DIR}" \
  "${SCENARIO[@]}" ${OVR[@]+"${OVR[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}

[ "${ALGO}" = filter ] && export FILTER_MASK="${RUN_DIR}/filter_mask.json"
bash "$(dirname "$0")/evaluate.sh" "${RUN_DIR}/checkpoints/unlearned.ckpt" "${EVAL_DATA_DIR}" "${SID}" \
  "${RUN_DIR}/eval" "${MANIFEST}" "${SEED}"
