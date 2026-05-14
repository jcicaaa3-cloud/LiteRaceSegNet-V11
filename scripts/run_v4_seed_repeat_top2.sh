#!/usr/bin/env bash
set -euo pipefail

# Optional but recommended if one run looks unusually good or bad.
# Selection by validation mIoU is legitimate only if you report it as validation-selected.

CONFIGS=(
  "seg/config/pothole_binary_literace_v4_A_highres_damage_boost_s7.yaml"
  "seg/config/pothole_binary_literace_v4_A_highres_damage_boost_s123.yaml"
  "seg/config/pothole_binary_literace_v4_B_recall_aggressive_s7.yaml"
  "seg/config/pothole_binary_literace_v4_B_recall_aggressive_s123.yaml"
)

mkdir -p logs
for CFG in "${CONFIGS[@]}"; do
  NAME=$(basename "$CFG" .yaml)
  echo "========== RUN $NAME ==========" | tee "logs/${NAME}.start.txt"
  python seg/train_literace.py --config "$CFG" 2>&1 | tee "logs/${NAME}.log"
done
