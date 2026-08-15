#!/bin/bash

set -euo pipefail

: "${RADIOBERT_PATH:?Please set RADIOBERT_PATH}"

python running.py \
  --name "${NAME:-exp001}" \
  --seed_t "${SEED:-2026}" \
  --hidden_dim "${HIDDEN_DIM:-256}" \
  --dropout "${DROPOUT:-0.4}" \
  --clin_layers "${CLIN_LAYERS:-2}" \
  --clin_hidden_dim "${CLIN_HIDDEN_DIM:-256}" \
  --lr_image "${LR_IMAGE:-5e-4}" \
  --lr_total "${LR_TOTAL:-1e-4}" \
  --lr_clin "${LR_CLIN:-1e-3}" \
  --lr_report "${LR_REPORT:-1e-4}" \
  --report_dropout "${REPORT_DROPOUT:-0.2}" \
  --radiobert_path "${RADIOBERT_PATH}" \
  --output_dir "${OUTPUT_DIR:-./checkpoints}"
