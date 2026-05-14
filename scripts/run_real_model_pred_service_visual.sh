#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-seg/config/pothole_binary_literace_train.yaml}"
CKPT="${2:-seg/runs/literace_boundary_degradation/best.pth}"
INPUT_DIR="${3:-datasets/pothole_binary/processed/val/images}"
OUTDIR="${4:-seg/runs/literace_boundary_degradation/real_pred_val}"

python seg/infer_literace_to_service.py   --config "$CONFIG"   --ckpt "$CKPT"   --input_dir "$INPUT_DIR"   --outdir "$OUTDIR"
