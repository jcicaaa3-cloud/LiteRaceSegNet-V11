#!/usr/bin/env bash
set -euo pipefail
python seg/train_literace.py --config seg/config/pothole_binary_literace_one_month_v3.yaml
