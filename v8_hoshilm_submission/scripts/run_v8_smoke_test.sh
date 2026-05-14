#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../hoshilm_kr"
python estimate_params.py --config configs/hoshilm_s.yaml
python train_hoshilm.py --config configs/hoshilm_s.yaml
