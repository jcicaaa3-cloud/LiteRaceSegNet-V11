#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv_projectqa
source .venv_projectqa/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 build_project_qa_corpus.py
# Default Project QA extension run: larger 12L/768d Project QA model (~86M params with bundled sp vocab).
python3 estimate_params.py --config configs/hoshilm_project_qa_xl.yaml
python3 train_hoshilm.py --config configs/hoshilm_project_qa_xl.yaml
python3 generate.py --ckpt runs/hoshilm_project_qa_xl/best.pt --prompt $'질문: 데이터는 몇 장인가?\n답변:' --tokens 260
