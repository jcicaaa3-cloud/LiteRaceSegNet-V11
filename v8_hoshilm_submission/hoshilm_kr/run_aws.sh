#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python estimate_params.py --config configs/hoshilm_m.yaml
python train_hoshilm.py --config configs/hoshilm_m.yaml
python generate.py --ckpt runs/hoshilm_m/best.pt --prompt "LiteRaceSegNet 프로젝트는" --tokens 160
