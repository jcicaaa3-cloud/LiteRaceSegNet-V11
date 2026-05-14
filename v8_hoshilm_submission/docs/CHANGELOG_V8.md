# V8 Changelog

## v8: Vision + HoshiLM Lab

### Added

- `hoshilm_kr/`: clean HoshiLM-KR training module
- `model.py`: decoder-only Transformer implementation
- `train_hoshilm.py`: training loop with train/val split, checkpointing, logging, mixed precision support
- `generate.py`: checkpoint-based text generation
- `estimate_params.py`: model scale estimator
- `configs/`: S/M/L/XL/legacy configs
- `web_demo/`: static v8 project showcase
- `docs/figures/`: LiteRaceSegNet metric figures and parameter scale figure

### Preserved

- Existing LiteRaceSegNet visual evidence archive
- Existing v7 WWDC-style presentation deck
- Uploaded legacy HoshiLM scripts under `hoshilm_kr/legacy/`

### Corrected framing

- Legacy comment suggested a 100M-parameter setup, but the uploaded 6-layer/384-dim/vocab-660 prototype is closer to an ~11M-class model.
- V8 provides explicit configs for 5M/14M/30M/~95M scale experiments.

