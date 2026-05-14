#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv_projectqa
source .venv_projectqa/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 build_project_qa_corpus.py
python3 estimate_params.py --config configs/hoshilm_project_qa.yaml
python3 train_hoshilm.py --config configs/hoshilm_project_qa.yaml
python3 generate.py --ckpt runs/hoshilm_project_qa/best.pt --prompt $'질문: 데이터는 몇 장인가?\n답변:' --tokens 220
