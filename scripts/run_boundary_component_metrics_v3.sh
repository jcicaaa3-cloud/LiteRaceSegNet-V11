#!/usr/bin/env bash
set -euo pipefail
python seg/tools/boundary_component_metrics.py \
  --pred_dir seg/runs/literace_one_month_v3_strict_infer/04_postprocessed_masks \
  --gt_dir datasets/pothole_binary/processed/val/masks \
  --out_csv final_evidence/02_metrics_and_compare/literace_boundary_component_metrics_v3.csv \
  --boundary_radius 2
