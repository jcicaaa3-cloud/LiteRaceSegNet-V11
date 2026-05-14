#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/road-damage-segmentation-portfolio
if [ -d ".venv" ]; then source .venv/bin/activate; fi
DATA_ROOT="datasets/pothole_binary_v6_aug/processed"
if [ ! -d "$DATA_ROOT/train/images" ]; then DATA_ROOT="datasets/pothole_binary_aug/processed"; fi
python scripts/train_segformer_e3_baseline.py --data-root "$DATA_ROOT" --save-dir seg/runs/segformer_e3_v6_baseline_safe --model-name nvidia/mit-b0 --epochs 45 --batch-size 1 --height 320 --width 448 --lr 0.00004 --weight-decay 0.01 --patience 10 --num-workers 2 --seed 42
python scripts/collect_segformer_e3_evidence.py --data-root "$DATA_ROOT" --run-dir seg/runs/segformer_e3_v6_baseline_safe --checkpoint best.pth --out-zip LRS_SEGFORMER_E3_V6_SAFE_EVIDENCE.zip
ls -lah LRS_SEGFORMER_E3_V6_SAFE_EVIDENCE.zip
