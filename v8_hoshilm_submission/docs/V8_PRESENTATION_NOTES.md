# V8 Presentation Notes

## One-line summary

LiteRaceSegNet v8 combines a lightweight road-damage segmentation model with a toy-scale decoder-only Transformer language model lab, showing both computer vision and language model implementation experience.

## Recommended slide structure

1. v8 overview: Vision + HoshiLM Lab
2. LiteRaceSegNet recap: road damage segmentation
3. Evidence: dataset count, leakage check, validation metrics
4. LiteRaceSegNet scale: 0.125M params
5. Why add HoshiLM-KR?
6. HoshiLM-KR architecture: decoder-only Transformer
7. HoshiLM-KR training pipeline
8. Parameter scale: S/M/L/XL configs
9. What this proves / what it does not prove
10. Future work: bigger corpus and cleaner evaluation

## Safe Q&A

### Q. Did you build large commercial LM?

No. This is not a large-commercial-LM-class model. It is a toy-scale decoder-only Transformer implementation designed to reproduce the core decoder-only Transformer training process.

### Q. Why is this relevant to LiteRaceSegNet?

LiteRaceSegNet shows computer vision modeling. HoshiLM-KR shows language model architecture understanding. Together, v8 presents a broader AI implementation portfolio while keeping the original road-damage project intact.

