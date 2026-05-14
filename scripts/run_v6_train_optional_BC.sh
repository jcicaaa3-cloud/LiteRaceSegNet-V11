#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/road-damage-segmentation-portfolio
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
python scripts/verify_pairs.py datasets/pothole_binary_aug_v6/processed
python seg/train_literace.py --config seg/config/pothole_binary_literace_v6_B_newval_recall.yaml
bash scripts/collect_v6_evidence.sh literace_v6_B_newval_recall_s42 || true
python seg/train_literace.py --config seg/config/pothole_binary_literace_v6_C_newval_precision.yaml
bash scripts/collect_v6_evidence.sh literace_v6_C_newval_precision_s42 || true
