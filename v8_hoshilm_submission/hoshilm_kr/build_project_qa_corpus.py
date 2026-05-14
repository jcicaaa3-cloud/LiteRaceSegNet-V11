from __future__ import annotations
import csv
import json
import re
from pathlib import Path

TEXT_EXTS = {'.txt', '.md', '.json', '.yaml', '.yml', '.csv'}
SKIP_PARTS = {'__pycache__', '.git', 'runs', '.venv_projectqa'}
GENERATED_NAMES = {'project_facts.json', 'project_qa_pairs.jsonl', 'project_qa_corpus.txt', 'data_project_qa.txt'}

def read_text(path: Path, max_chars: int = 120_000) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='replace')[:max_chars]
    except Exception:
        return ''

def parse_key_value_counts(text: str) -> dict:
    facts = {}
    patterns = {
        'original_train_images': r'train images:\s*(\d+)',
        'original_train_masks': r'train masks\s*:\s*(\d+)',
        'original_val_images': r'val images\s*:\s*(\d+)',
        'original_val_masks': r'val masks\s*:\s*(\d+)',
        'aug_train_images': r'augmented dataset[\s\S]*?train images:\s*(\d+)',
        'aug_train_masks': r'augmented dataset[\s\S]*?train masks\s*:\s*(\d+)',
        'aug_val_images': r'augmented dataset[\s\S]*?val images\s*:\s*(\d+)',
        'aug_val_masks': r'augmented dataset[\s\S]*?val masks\s*:\s*(\d+)',
        'newval_in_train': r'newval in train should be 0\s*:\s*(\d+)',
        'newval_in_val': r'newval in val should be 10\s*:\s*(\d+)',
        'oldval_in_train': r'oldval_pothole in train should be 10:\s*(\d+)',
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, flags=re.I)
        if m:
            facts[k] = int(m.group(1))
    if '[OK] all pairs valid' in text:
        facts['pair_check'] = 'OK'
    return facts

def parse_train_log(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = []
    with path.open('r', encoding='utf-8', errors='replace', newline='') as f:
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                k = (k or '').lstrip('\ufeff')
                try:
                    clean[k] = float(v)
                except Exception:
                    clean[k] = v
            rows.append(clean)
    if not rows:
        return {}
    def best_by(col: str):
        if col not in rows[0]:
            return None
        return max(rows, key=lambda r: float(r.get(col, float('-inf'))))
    last = rows[-1]
    best_updates = [r for r in rows if int(float(r.get('best', 0))) == 1]
    return {
        'train_log_path': str(path),
        'epochs_recorded': int(last.get('epoch', len(rows))),
        'last_epoch': last,
        'best_miou_binary': best_by('miou_binary'),
        'best_iou_damage': best_by('iou_damage'),
        'best_pixel_acc': best_by('pixel_acc'),
        'final_best_update': best_updates[-1] if best_updates else None,
        'columns': list(rows[0].keys()),
    }

def parse_yaml_highlights(path: Path) -> dict:
    text = read_text(path)
    if not text:
        return {}
    facts = {'path': str(path)}
    patterns = {
        'save_dir': r'^save_dir:\s*(.+)$',
        'model_name': r'^\s*name:\s*(.+)$',
        'base_channels': r'^\s*base_channels:\s*(\d+)',
        'context_channels': r'^\s*context_channels:\s*(\d+)',
        'use_detail_branch': r'^\s*use_detail_branch:\s*(\w+)',
        'context_module': r'^\s*context_module:\s*(.+)',
        'use_boundary_gate': r'^\s*use_boundary_gate:\s*(\w+)',
        'fuse_boundary_logit': r'^\s*fuse_boundary_logit:\s*(\w+)',
        'image_size': r'^\s*image_size:\s*\[(.+)\]',
        'batch_size': r'^\s*batch_size:\s*(\d+)',
        'epochs': r'^\s*epochs:\s*(\d+)',
        'base_lr': r'^\s*base_lr:\s*([0-9.eE-]+)',
        'boundary_width': r'^\s*boundary_width:\s*(\d+)',
        'class_weights': r'^\s*class_weights:\s*\[(.+)\]',
        'dice_weight': r'^\s*dice_weight:\s*([0-9.eE-]+)',
        'boundary_weight': r'^\s*boundary_weight:\s*([0-9.eE-]+)',
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, flags=re.M)
        if m:
            facts[k] = m.group(1).strip()
    return facts

def fmt_float(x, nd=4):
    try:
        return f'{float(x):.{nd}f}'
    except Exception:
        return str(x)

def build_qa_pairs(facts: dict) -> list[dict]:
    pairs = []
    counts = facts.get('dataset_counts', {})
    metrics = facts.get('train_metrics', {})
    cfg = facts.get('primary_config', {})
    def add(q, a, source='project_facts.json'):
        pairs.append({'question': q, 'answer': a, 'source': source})
    if counts:
        add('데이터는 몇 장인가?', f"원본 데이터는 train {counts.get('original_train_images')} images/{counts.get('original_train_masks')} masks, val {counts.get('original_val_images')} images/{counts.get('original_val_masks')} masks입니다. 증강 후에는 train {counts.get('aug_train_images')} images/{counts.get('aug_train_masks')} masks, val {counts.get('aug_val_images')} images/{counts.get('aug_val_masks')} masks입니다. pair check는 {counts.get('pair_check', 'unknown')}입니다.")
        add('데이터 누수는 확인했나?', f"확인했습니다. newval in train={counts.get('newval_in_train')}, newval in val={counts.get('newval_in_val')}, oldval_pothole in train={counts.get('oldval_in_train')}로 기록되어 있습니다. 새 검증셋은 train에 들어가지 않았다는 근거로 제시할 수 있습니다.")
    if metrics:
        bm = metrics.get('best_miou_binary') or {}
        bd = metrics.get('best_iou_damage') or {}
        last = metrics.get('last_epoch') or {}
        add('mIoU는 어느 정도인가?', f"기록된 train_log 기준 best binary mIoU는 epoch {int(bm.get('epoch', -1))}에서 {fmt_float(bm.get('miou_binary'))}입니다. 같은 row의 damage IoU는 {fmt_float(bm.get('iou_damage'))}, background IoU는 {fmt_float(bm.get('iou_background'))}, pixel accuracy는 {fmt_float(bm.get('pixel_acc'))}입니다. 마지막 기록 epoch {int(last.get('epoch', -1))}의 binary mIoU는 {fmt_float(last.get('miou_binary'))}입니다.")
        add('damage IoU는 얼마인가?', f"damage IoU의 최고 기록은 epoch {int(bd.get('epoch', -1))}에서 {fmt_float(bd.get('iou_damage'))}입니다. 그 row의 binary mIoU는 {fmt_float(bd.get('miou_binary'))}입니다.")
        add('학습은 몇 epoch까지 기록됐나?', f"현재 evidence summary의 train_log.csv에는 epoch {metrics.get('epochs_recorded')}까지 기록되어 있습니다. 설정상 목표 epoch와 실제 기록 epoch는 다를 수 있으므로 발표에서는 '현재 제출 패키지 evidence 기준'이라고 말하는 것이 안전합니다.")
    if cfg:
        add('모델 구조는 어떻게 설명하나?', f"LiteRaceSegNet v6 A 설정은 model={cfg.get('model_name')}, base_channels={cfg.get('base_channels')}, context_channels={cfg.get('context_channels')}, detail branch={cfg.get('use_detail_branch')}, context module={cfg.get('context_module')}, boundary gate={cfg.get('use_boundary_gate')}, boundary logit fusion={cfg.get('fuse_boundary_logit')}로 요약할 수 있습니다.")
        add('학습 설정은 어떻게 되나?', f"주 설정은 image_size=[{cfg.get('image_size')}], batch_size={cfg.get('batch_size')}, epochs={cfg.get('epochs')}, base_lr={cfg.get('base_lr')}, boundary_width={cfg.get('boundary_width')}, class_weights=[{cfg.get('class_weights')}], dice_weight={cfg.get('dice_weight')}, boundary_weight={cfg.get('boundary_weight')}입니다.")
    add('이 HoshiLM은 무엇인가?', 'HoshiLM은 LiteRaceSegNet 이미지 모델과 별개의 소형 한국어 decoder-only Transformer 언어모델 실험입니다. v9 Project QA 확장 버전에서는 결과 파일을 읽어 만든 corpus와 facts를 기반으로 Project QA에 사용합니다. 대형 상용 LLM 수준 성능을 주장하지 않고, 재현 가능한 학습/검색/대화 파이프라인을 보여주는 목적입니다.')
    add('웹 UI 응답은 하드코딩 방식인가?', '아닙니다. web_project_qa/index.html은 질문을 Python API로 보내고, API가 project_facts.json 및 project_qa_corpus.txt를 검색해 답변합니다. 체크포인트가 있으면 HoshiLM 생성도 보조로 사용할 수 있습니다.')
    return pairs

def main():
    here = Path(__file__).resolve().parent
    submission = here.parent
    evidence = submission / 'lrs_v6_evidence_summary'
    docs = submission / 'docs'
    out_data = here / 'data'
    out_data.mkdir(parents=True, exist_ok=True)
    counts_text = ''
    for p in [evidence / 'V6_COUNTS_AND_LEAKAGE_CHECK.txt', evidence / 'dataset_verify_original.txt', evidence / 'dataset_verify_aug_v6.txt']:
        counts_text += f'\n--- {p.name} ---\n' + read_text(p)
    facts = {
        'project': 'LiteRaceSegNet v9 + HoshiLM Project QA',
        'purpose': 'Evidence-grounded QA over LiteRaceSegNet result files and HoshiLM training corpus.',
        'dataset_counts': parse_key_value_counts(counts_text),
        'train_metrics': parse_train_log(evidence / 'literace_v6_A_newval_balanced_s42' / 'train_log.csv'),
        'primary_config': parse_yaml_highlights(evidence / 'pothole_binary_literace_v6_A_newval_balanced.yaml'),
        'limits': ['HoshiLM is a toy/small decoder-only Transformer experiment, not a production LLM.', 'Project QA should cite evidence files and avoid claiming metrics not present in the archive.', 'HTML UI does not contain fixed answer tables; it calls the local Python API.'],
    }
    qa_pairs = build_qa_pairs(facts)
    (out_data / 'project_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out_data / 'project_qa_pairs.jsonl').open('w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    sections = ['# LiteRaceSegNet Project Facts\n' + json.dumps(facts, ensure_ascii=False, indent=2), '# Canonical QA Pairs']
    for p in qa_pairs:
        sections.append(f"질문: {p['question']}\n답변: {p['answer']}\n근거: {p['source']}")
    for base in [evidence, docs, submission]:
        if not base.exists():
            continue
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in TEXT_EXTS:
                continue
            if any(part in SKIP_PARTS for part in p.parts):
                continue
            if p.name in GENERATED_NAMES:
                continue
            try:
                rel = p.relative_to(submission)
            except ValueError:
                rel = p.name
            text = read_text(p, max_chars=30_000)
            if text.strip():
                sections.append(f"# SOURCE: {rel}\n{text}")
    corpus = '\n\n'.join(sections).strip() + '\n'
    (out_data / 'project_qa_corpus.txt').write_text(corpus, encoding='utf-8')
    old_data = out_data / 'data.txt'
    original = old_data.read_text(encoding='utf-8', errors='replace') if old_data.exists() else ''
    (out_data / 'data_project_qa.txt').write_text(corpus + '\n\n# Original HoshiLM Seed Data\n' + original, encoding='utf-8')
    print('Wrote:', out_data / 'project_facts.json')
    print('Wrote:', out_data / 'project_qa_pairs.jsonl')
    print('Wrote:', out_data / 'project_qa_corpus.txt')
    print('Wrote:', out_data / 'data_project_qa.txt')
    print(f'QA pairs: {len(qa_pairs)}, corpus chars: {len(corpus):,}')

if __name__ == '__main__':
    main()
