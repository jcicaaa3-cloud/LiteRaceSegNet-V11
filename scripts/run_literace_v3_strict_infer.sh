#!/usr/bin/env bash
set -euo pipefail
python seg/infer_literace_research_strict.py \
  --input_dir assets/service_demo/input_batch \
  --config seg/config/pothole_binary_literace_one_month_v3.yaml \
  --ckpt seg/runs/literace_one_month_v3/best.pth \
  --outdir seg/runs/literace_one_month_v3_strict_infer \
  --threshold 0.65 \
  --min_area_pixels 180 \
  --save_prob
