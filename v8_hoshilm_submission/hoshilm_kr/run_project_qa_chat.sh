#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv_projectqa ]; then python3 -m venv .venv_projectqa; fi
source .venv_projectqa/bin/activate
pip install -r requirements.txt >/dev/null
python3 build_project_qa_corpus.py
CKPT="runs/hoshilm_project_qa/best.pt"
if [ -f "$CKPT" ]; then python3 project_qa_chat.py --ckpt "$CKPT"; else python3 project_qa_chat.py --no-lm; fi
