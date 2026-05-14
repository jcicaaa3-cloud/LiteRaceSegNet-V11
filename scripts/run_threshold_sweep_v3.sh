#!/usr/bin/env bash
set -euo pipefail
python seg/tools/threshold_sweep_literace.py \
  --config seg/config/pothole_binary_literace_one_month_v3.yaml \
  --ckpt seg/runs/literace_one_month_v3/best.pth \
  --out_csv final_evidence/02_metrics_and_compare/literace_threshold_sweep_v3.csv \
  --thresholds 0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
  --min_areas 0,60,120,180,240
