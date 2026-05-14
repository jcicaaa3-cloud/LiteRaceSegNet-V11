#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python estimate_params.py --config configs/hoshilm_smoke_aws.yaml
python train_hoshilm.py --config configs/hoshilm_smoke_aws.yaml
python generate.py --ckpt runs/hoshilm_smoke_aws/best.pt --prompt "LiteRaceSegNet 프로젝트는" --tokens 80
