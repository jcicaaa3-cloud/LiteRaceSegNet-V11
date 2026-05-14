#!/usr/bin/env bash
set -euo pipefail
python seg/train_literace.py --config seg/config/pothole_binary_literace_upgrade_conservative.yaml
