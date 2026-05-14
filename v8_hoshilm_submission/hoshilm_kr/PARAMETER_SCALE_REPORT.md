# HoshiLM Parameter Scale Report

## Before

`configs/hoshilm_project_qa.yaml`

```text
n_layer: 6
n_head: 6
n_embd: 384
block_size: 384
tokenizer: char
estimated params: 11.027M
```

## Up-The-Ante Default

`configs/hoshilm_project_qa_xl.yaml`

```text
n_layer: 12
n_head: 12
n_embd: 768
block_size: 512
tokenizer: sentencepiece
estimated params: 85.956M
```

This is the new default used by:

```bash
bash run_project_qa_train.sh
```

## Experimental XXL

`configs/hoshilm_project_qa_xxl_200m_experimental.yaml`

```text
n_layer: 16
n_head: 16
n_embd: 1024
block_size: 768
tokenizer: sentencepiece
estimated params: 203.004M
```

This is optional and intentionally separated because the current Project QA corpus is small. It is useful as a capacity demonstration, but it can overfit quickly.

## Positioning

Do not claim that HoshiLM improves segmentation prediction. LiteRaceSegNet remains the core CNN segmentation model. HoshiLM Project QA is a reporting/support interface that reads project evidence and generates project explanations.
