#!/usr/bin/env bash
set -euo pipefail

# Run from repository root: /home/ubuntu/road-damage-segmentation-portfolio
# This script only changes training/configuration. It does not edit result CSV values.

mkdir -p logs seg/config
cp -v seg/config/pothole_binary_literace_v4_*.yaml ./seg/config/ 2>/dev/null || true

CONFIGS=(
  "seg/config/pothole_binary_literace_v4_A_highres_damage_boost.yaml"
  "seg/config/pothole_binary_literace_v4_B_recall_aggressive.yaml"
  "seg/config/pothole_binary_literace_v4_C_safe_no_oom.yaml"
  "seg/config/pothole_binary_literace_v4_D_precision_recover.yaml"
)

for CFG in "${CONFIGS[@]}"; do
  NAME=$(basename "$CFG" .yaml)
  echo "========== RUN $NAME ==========" | tee "logs/${NAME}.start.txt"
  python seg/train_literace.py --config "$CFG" 2>&1 | tee "logs/${NAME}.log"
done

echo "[DONE] v4 base sweep finished. Run scripts/summarize_v4_results.py next."
