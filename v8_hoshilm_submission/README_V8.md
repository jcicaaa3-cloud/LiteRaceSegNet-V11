# LiteRaceSegNet v8 Combined Package

This package combines the split LiteRaceSegNet v6 visual evidence archive with a new **HoshiLM-KR Lab** module.

**HoshiLM-KR** is the rebranded name for the uploaded mini language-model prototype. It avoids product-name confusion and presents the work as an original educational decoder-only Transformer LM experiment.

## What v8 adds

- LiteRaceSegNet evidence archive preserved under `lrs_evidence_base/`
- WWDC-style v7 presentation decks preserved under `presentation/`
- HoshiLM-KR module under `hoshilm_kr/`
- Static web demo under `web_demo/`
- Documentation under `docs/`

## Safe project framing

Do **not** present HoshiLM-KR as a large-commercial-LM-class system.

Use this wording:

> HoshiLM-KR is a toy-scale decoder-only Transformer language model implemented for reproducing the core decoder-only Transformer training pipeline: tokenization, causal self-attention, next-token prediction, checkpointing, and sampling.

Avoid this wording:

> We built large commercial LM.
> We built commercial chatbot.
> We built a production LLM.

## Key numbers

### LiteRaceSegNet evidence checkpoint

- Estimated trainable parameters: **124,509**
- Estimated parameter scale: **0.125M**
- FP32 parameter size: **0.475 MiB**
- Best validation binary mIoU from evidence: **0.7988**
- Damage IoU at best epoch: **0.7029**

### HoshiLM-KR config scale

Expected parameter scale depends on tokenizer vocabulary size.

| Config | Layers | Heads | Embedding | Block | Expected scale |
|---|---:|---:|---:|---:|---:|
| HoshiLM-S | 4 | 4 | 256 | 128 | ~5M with vocab≈8k |
| HoshiLM-M | 6 | 6 | 384 | 256 | ~14M with vocab≈8k |
| HoshiLM-L | 8 | 8 | 512 | 384 | ~30M with vocab≈8k |
| HoshiLM-XL experimental | 12 | 12 | 768 | 512 | ~95M with vocab≈12k |

The uploaded legacy prototype used a very small SentencePiece vocab of 660, so its true scale is smaller than a normal subword-vocab decoder-only Transformer configuration.

## Quick start: HoshiLM-KR

```bash
cd hoshilm_kr
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python estimate_params.py --config configs/hoshilm_m.yaml
python train_hoshilm.py --config configs/hoshilm_m.yaml
python generate.py --ckpt runs/hoshilm_m/best.pt --prompt "LiteRaceSegNet 프로젝트는" --tokens 160
```

On Windows PowerShell:

```powershell
cd hoshilm_kr
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python estimate_params.py --config configs/hoshilm_m.yaml
python train_hoshilm.py --config configs/hoshilm_m.yaml
python generate.py --ckpt runs/hoshilm_m\best.pt --prompt "LiteRaceSegNet 프로젝트는" --tokens 160
```

## Web demo

Open:

```text
web_demo/index.html
```

This is a static project showcase page. The previous mock assistant/chat section was removed from the submission-clean package to avoid confusion with a deployed chatbot.

