#!/usr/bin/env bash
# Unwanted-item scenario: pick users who request deletion of their interactions with
# an item category, then build
#   <out_prefix>_sens/     forget/retain split of those interactions (for unlearning)
#   <out_prefix>_retrain/  training data without them (for the retrained reference)
# The category is given either as a taxonomy node (CATEGORY and LEVEL) or as a list
# of title KEYWORDS.
#
#   CATEGORY="Hair Loss Products" LEVEL=2 bash scripts/pipeline/select_unwanted.sh <clean_data_dir> <out_prefix> [seed]
#   KEYWORDS="gun rifle pistol" bash scripts/pipeline/select_unwanted.sh <clean_data_dir> <out_prefix> [seed]
set -euo pipefail
CLEAN="${1:?clean data_dir}"
PREFIX="${2:?output prefix}"
SEED="${3:-2}"
RATIO="${FORGET_RATIO:-1e-4}"
SENS="${PREFIX}_sens"
RETRAIN="${PREFIX}_retrain"
MANIFEST="${PREFIX}_forget_manifest.json"

if [ -n "${KEYWORDS:-}" ]; then
  SELECT=(--keywords ${KEYWORDS})
else
  SELECT=(--category "${CATEGORY:?set CATEGORY and LEVEL, or KEYWORDS}" --category_level "${LEVEL:?LEVEL}")
fi
python -m scripts.build_sensitive_manifest --data_dir "${CLEAN}" "${SELECT[@]}" \
  --seed "${SEED}" --forget_ratio "${RATIO}" --deletion_spec item_pairs --out "${MANIFEST}"

# evaluation data and items are shared with the clean dataset
mkdir -p "${SENS}"
for sub in training items evaluation testing; do
  [ -e "${SENS}/${sub}" ] || ln -s "$(readlink -f "${CLEAN}/${sub}")" "${SENS}/${sub}"
done
cp "${MANIFEST}" "${SENS}/forget_manifest.json"
python -m src.data.unlearning.split_forget_retain --data_dir "${SENS}" --forget_manifest "${SENS}/forget_manifest.json"

python -m scripts.build_retrain_dataset --data_dir "${CLEAN}" --manifest "${MANIFEST}" --out_dir "${RETRAIN}"
