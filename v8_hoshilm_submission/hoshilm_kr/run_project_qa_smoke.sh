#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv_projectqa
source .venv_projectqa/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 build_project_qa_corpus.py
python3 train_hoshilm.py --config configs/hoshilm_project_qa_smoke.yaml
python3 project_qa_chat.py --ckpt runs/hoshilm_project_qa_smoke/best.pt --no-lm <<'INNER_EOF'
데이터는 몇 장인가?
mIoU는 어느 정도인가?
exit
INNER_EOF
