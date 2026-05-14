from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

def tokenize(text: str) -> list[str]:
    words = re.findall(r'[가-힣A-Za-z0-9_.+-]+', text.lower())
    chars = [c for c in text.lower() if not c.isspace()]
    trigrams = [''.join(chars[i:i+3]) for i in range(max(0, len(chars)-2))]
    return words + trigrams

def fmt(x: Any, nd: int = 4) -> str:
    try:
        if x is None:
            return 'unknown'
        return f'{float(x):.{nd}f}'
    except Exception:
        return str(x)

@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float

class ProjectQABot:
    def __init__(self, base_dir: str | Path | None = None, ckpt: str | Path | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent
        self.data_dir = self.base_dir / 'data'
        self.facts_path = self.data_dir / 'project_facts.json'
        self.corpus_path = self.data_dir / 'project_qa_corpus.txt'
        if not self.facts_path.exists() or not self.corpus_path.exists():
            import build_project_qa_corpus
            build_project_qa_corpus.main()
        self.facts = json.loads(self.facts_path.read_text(encoding='utf-8'))
        self.corpus = self.corpus_path.read_text(encoding='utf-8', errors='replace')
        self.chunks = self._split_corpus(self.corpus)
        self.ckpt = Path(ckpt) if ckpt else self._find_default_ckpt()
        self._lm = None
        self._tok = None
        self._device = None

    def _find_default_ckpt(self) -> Path | None:
        candidates = [
            self.base_dir/'runs/hoshilm_project_qa_xl/best.pt',
            self.base_dir/'runs/hoshilm_project_qa_xl/last.pt',
            self.base_dir/'runs/hoshilm_project_qa_xxl_200m/best.pt',
            self.base_dir/'runs/hoshilm_project_qa_xxl_200m/last.pt',
            self.base_dir/'runs/hoshilm_project_qa/best.pt',
            self.base_dir/'runs/hoshilm_project_qa/last.pt',
            self.base_dir/'runs/hoshilm_m/best.pt',
            self.base_dir/'runs/hoshilm_smoke_aws/best.pt',
        ]
        return next((p for p in candidates if p.exists()), None)

    def _split_corpus(self, corpus: str) -> list[dict[str, str]]:
        pieces = []
        for block in re.split(r'\n(?=# )', corpus):
            if not block.strip():
                continue
            title = block.split('\n', 1)[0].strip('# ').strip()
            m = re.match(r'# SOURCE:\s*(.+)', block)
            source = m.group(1).strip() if m else title or 'project_qa_corpus.txt'
            for i in range(0, len(block), 1800):
                pieces.append({'source': source, 'text': block[i:i+1800]})
        return pieces

    def retrieve(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        q_set = set(tokenize(question))
        scored = []
        for c in self.chunks:
            toks = tokenize(c['text'])
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            score = sum(1.0 + math.log1p(counts[t]) for t in q_set if t in counts)
            low = c['text'].lower(); qlow = question.lower()
            for term in ['miou','iou','epoch','train','val','boundary','leakage','segformer','dataset','html','hoshilm']:
                if term in qlow and term in low:
                    score += 2.0
            if score > 0:
                scored.append(RetrievedChunk(c['source'], c['text'], score))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]

    def answer_from_facts(self, question: str) -> str:
        q = question.lower()
        counts = self.facts.get('dataset_counts', {})
        metrics = self.facts.get('train_metrics', {})
        cfg = self.facts.get('primary_config', {})
        def has(*terms: str) -> bool:
            return any(t in q for t in terms)
        if has('데이터','몇 장','몇장','dataset','data count','이미지'):
            return f"원본 데이터는 train {counts.get('original_train_images')} images/{counts.get('original_train_masks')} masks, val {counts.get('original_val_images')} images/{counts.get('original_val_masks')} masks입니다. 증강 후에는 train {counts.get('aug_train_images')} images/{counts.get('aug_train_masks')} masks, val {counts.get('aug_val_images')} images/{counts.get('aug_val_masks')} masks입니다. pair 검증 상태는 {counts.get('pair_check','unknown')}입니다."
        if has('누수','leak','leakage','검증셋','newval'):
            return f"데이터 누수 체크 기록은 newval in train={counts.get('newval_in_train')}, newval in val={counts.get('newval_in_val')}, oldval_pothole in train={counts.get('oldval_in_train')}입니다. 현재 evidence 기준으로 새 검증셋은 train에 들어가지 않은 것으로 정리되어 있습니다."
        if has('miou','성능','metric','iou'):
            bm = metrics.get('best_miou_binary') or {}; bd = metrics.get('best_iou_damage') or {}; last = metrics.get('last_epoch') or {}
            return f"train_log.csv 기준 best binary mIoU는 epoch {int(bm.get('epoch', -1))}의 {fmt(bm.get('miou_binary'))}입니다. 그때 damage IoU={fmt(bm.get('iou_damage'))}, background IoU={fmt(bm.get('iou_background'))}, pixel accuracy={fmt(bm.get('pixel_acc'))}입니다. damage IoU 최고 기록은 epoch {int(bd.get('epoch', -1))}의 {fmt(bd.get('iou_damage'))}입니다. 마지막 기록 epoch {int(last.get('epoch', -1))}의 binary mIoU는 {fmt(last.get('miou_binary'))}입니다."
        if has('epoch','에포크','학습'):
            last = metrics.get('last_epoch') or {}
            return f"현재 포함된 train_log.csv는 epoch {metrics.get('epochs_recorded')}까지 기록되어 있습니다. 마지막 row 기준 train_loss={fmt(last.get('train_loss'))}, val_loss={fmt(last.get('val_loss'))}, binary mIoU={fmt(last.get('miou_binary'))}입니다."
        if has('구조','architecture','boundary','branch','모델'):
            return f"v6 A 설정 기준 LiteRaceSegNet은 {cfg.get('model_name')} 모델이며, base_channels={cfg.get('base_channels')}, context_channels={cfg.get('context_channels')}, detail branch={cfg.get('use_detail_branch')}, context module={cfg.get('context_module')}, boundary gate={cfg.get('use_boundary_gate')}, boundary logit fusion={cfg.get('fuse_boundary_logit')}로 정리됩니다. 발표에서는 경량 segmentation + boundary-aware 보조 흐름으로 설명하는 게 안전합니다."
        if has('설정','config','lr','batch','loss'):
            return f"주 학습 설정은 image_size=[{cfg.get('image_size')}], batch_size={cfg.get('batch_size')}, epochs={cfg.get('epochs')}, base_lr={cfg.get('base_lr')}, boundary_width={cfg.get('boundary_width')}, class_weights=[{cfg.get('class_weights')}], dice_weight={cfg.get('dice_weight')}, boundary_weight={cfg.get('boundary_weight')}입니다."
        if has('html','웹','고정','하드코딩','박아','fake','서비스'):
            return '현재 web_project_qa/index.html은 HTML 내부에 고정 답변을 두는 방식이 아닙니다. 브라우저가 /api/chat으로 질문을 보내면 Python API가 project_facts.json과 project_qa_corpus.txt를 검색해 답변합니다. HTML은 화면과 요청 전송만 담당합니다. 체크포인트가 있으면 HoshiLM 생성문도 보조로 붙일 수 있습니다.'
        if has('llm','hoshilm','챗','대화'):
            return 'HoshiLM은 소형 한국어 decoder-only Transformer 실험입니다. 이번 Project QA 확장 버전에서는 LiteRaceSegNet 결과 파일에서 만든 project QA corpus로 학습할 수 있고, 대화 UI는 그 결과 corpus와 facts를 근거로 답합니다. 상용 LLM급 성능 주장이 아니라 재현 가능한 미니 LLM/RAG형 실험입니다.'
        return ''

    def load_lm(self):
        if self._lm is not None or self.ckpt is None or not self.ckpt.exists():
            return self._lm, self._tok
        import torch
        from model import LMConfig, HoshiLM
        from tokenizer_utils import CharTokenizer, SentencePieceTokenizer
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(self.ckpt, map_location=self._device)
        train_cfg = ckpt.get('train_config', {})
        if train_cfg.get('tokenizer') == 'sentencepiece':
            self._tok = SentencePieceTokenizer(train_cfg['sp_model_path'])
        else:
            self._tok = CharTokenizer(vocab_path=train_cfg.get('char_vocab_path') or str(self.base_dir/'runs/hoshilm_project_qa/char_vocab.txt'))
        self._lm = HoshiLM(LMConfig(**ckpt['model_config'])).to(self._device)
        self._lm.load_state_dict(ckpt['model'])
        self._lm.eval()
        return self._lm, self._tok

    def generate_note(self, question: str, grounded_answer: str, max_tokens: int = 160) -> str:
        if self.ckpt is None or not self.ckpt.exists():
            return ''
        try:
            import torch
            lm, tok = self.load_lm()
            prompt = f"질문: {question}\n근거 답변: {grounded_answer}\n요약:"
            idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=self._device)
            out = lm.generate(idx, max_new_tokens=max_tokens, temperature=0.65, top_k=40)
            text = tok.decode(out[0].tolist())
            text = text.split('요약:', 1)[-1].strip() if '요약:' in text else text.strip()
            return text[:700]
        except Exception as e:
            return f'(HoshiLM generation skipped: {e})'

    def ask(self, question: str, use_lm: bool = True, k: int = 4) -> dict[str, Any]:
        grounded = self.answer_from_facts(question)
        chunks = self.retrieve(question, k=k)
        if not grounded:
            if chunks:
                excerpt = re.sub(r'\s+', ' ', chunks[0].text).strip()
                grounded = '검색된 근거 기준으로 답하면, ' + excerpt[:900]
            else:
                grounded = '현재 project_facts.json과 corpus에서 직접 근거를 찾지 못했습니다. 질문을 데이터 수, mIoU, 학습 설정, 구조, leakage, HTML/UI 쪽으로 좁히면 더 정확히 답할 수 있습니다.'
        model_note = self.generate_note(question, grounded) if use_lm else ''
        sources = []
        for c in chunks:
            if c.source not in sources:
                sources.append(c.source)
        return {'answer': grounded, 'model_note': model_note, 'sources': sources[:k], 'ckpt': str(self.ckpt) if self.ckpt else None, 'mode': 'evidence+hoshilm' if use_lm and self.ckpt else 'evidence-only'}
