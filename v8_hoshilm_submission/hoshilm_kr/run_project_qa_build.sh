#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 build_project_qa_corpus.py
python3 - <<'INNERPY'
import json
from pathlib import Path
facts=json.loads(Path('data/project_facts.json').read_text(encoding='utf-8'))
print(json.dumps({'dataset_counts': facts.get('dataset_counts'), 'best_miou_binary': facts.get('train_metrics',{}).get('best_miou_binary'), 'primary_config': facts.get('primary_config')}, ensure_ascii=False, indent=2))
INNERPY
