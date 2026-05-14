#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/road-damage-segmentation-portfolio
NEW_VAL_ZIP="${1:-~/LRS_NEW_VAL_10_PAIRS.zip}"
bash scripts/run_v6_prepare_and_train.sh "$NEW_VAL_ZIP"
bash scripts/run_v6_segformer_e3_baseline.sh
