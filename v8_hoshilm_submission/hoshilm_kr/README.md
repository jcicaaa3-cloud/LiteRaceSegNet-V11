# HoshiLM-KR Lab

A toy-scale decoder-only Transformer language model implementation for understanding decoder-only Transformer language modeling.

## Scope

This module implements:

- Tokenization with either char-level tokenizer or SentencePiece
- Token embedding + positional embedding
- Causal self-attention
- Transformer decoder blocks
- Next-token prediction loss
- Train/validation split
- Checkpoint saving
- Top-k sampling

## Not a production LLM

HoshiLM-KR is not large commercial LM, not commercial chatbot, and not a production chatbot. It is a small reproducibility/learning project.

## Recommended run order

```bash
pip install -r requirements.txt
python estimate_params.py --config configs/hoshilm_m.yaml
python train_hoshilm.py --config configs/hoshilm_m.yaml
python generate.py --ckpt runs/hoshilm_m/best.pt --prompt "프로젝트는" --tokens 160
```

## Config choice

- `hoshilm_s.yaml`: safe quick run
- `hoshilm_m.yaml`: main v8 run
- `hoshilm_l.yaml`: GPU experiment after expanding data
- `hoshilm_xl_100m_experimental.yaml`: near-100M experiment, only meaningful with a much larger corpus
- `legacy_sp660.yaml`: reproduces the uploaded SentencePiece-660 setup

## Dataset warning

The included `data/data.txt` is a small dummy corpus. It is useful for smoke tests and pipeline verification, but it is not enough to train a useful language model. Increase corpus size before using the L/XL configs.


## v9 Project QA Extension: Project QA Web Chat

```bash
bash run_project_qa_build.sh
bash run_project_qa_train.sh
bash run_project_qa_web.sh
```

Open `http://AWS_PUBLIC_IP:8000` after allowing port 8000 in the EC2 security group. The HTML UI does not contain hard-coded answers. It calls the local Python API, which reads `data/project_facts.json` and `data/project_qa_corpus.txt`; if `runs/hoshilm_project_qa/best.pt` exists, HoshiLM generation can be used as an auxiliary note.
