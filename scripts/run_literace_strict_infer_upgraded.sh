#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-seg/config/pothole_binary_literace_upgrade_conservative.yaml}"
CKPT="${2:-seg/runs/literace_upgrade_conservative/best.pth}"
INPUT_DIR="${3:-datasets/pothole_binary/processed/val/images}"
OUTDIR="${4:-seg/runs/literace_upgrade_conservative/strict_val_pred}"
python seg/infer_literace_research_strict.py --config "$CONFIG" --ckpt "$CKPT" --input_dir "$INPUT_DIR" --outdir "$OUTDIR" --threshold 0.60 --min_area_pixels 120 --save_prob
PYTHONPATH=. python seg/tools/audit_prediction_masks.py --mask_dir "$OUTDIR/04_postprocessed_masks" --out_csv "$OUTDIR/mask_quality_audit.csv"
