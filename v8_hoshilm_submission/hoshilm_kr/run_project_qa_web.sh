#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv_projectqa ]; then python3 -m venv .venv_projectqa; fi
source .venv_projectqa/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null
python3 build_project_qa_corpus.py
CKPT="runs/hoshilm_project_qa/best.pt"
if [ ! -f "$CKPT" ]; then
  echo "[WARN] $CKPT not found. Web QA will run evidence-only until you train HoshiLM."
  python3 project_qa_api.py --host 0.0.0.0 --port 8000
else
  python3 project_qa_api.py --host 0.0.0.0 --port 8000 --ckpt "$CKPT"
fi
