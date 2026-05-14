# Results dashboard template

Use this page after running the final evidence scripts. Keep unknown values as `TBD` until measured.

## Preliminary reference evidence

| Model | Params | FP32 size | Pixel Acc. | binary mIoU | Damage IoU | Notes |
|---|---:|---:|---:|---:|---:|---|
| LiteRaceSegNet reference | 0.1245M | 0.475 MiB | 0.9157 | 0.7988 | 0.7029 | preliminary small validation evidence |

## Final comparison table

| Model | Device | Params | Size | mIoU | Damage IoU | Boundary IoU | Latency | FPS | CUDA memory | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LiteRaceSegNet | CPU | TBD | TBD | TBD | TBD | TBD | TBD | TBD | - | not measured |
| SegFormer baseline | CPU | TBD | TBD | TBD | TBD | TBD | TBD | TBD | - | not measured |
| LiteRaceSegNet | CUDA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not measured |
| SegFormer baseline | CUDA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not measured |

## Ablation table

| Variant | mIoU | Damage IoU | Boundary IoU | Params | Latency | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Full model | TBD | TBD | TBD | TBD | TBD | reference |
| w/o detail branch | TBD | TBD | TBD | TBD | TBD | detail contribution |
| w/o LiteASPP | TBD | TBD | TBD | TBD | TBD | context contribution |
| w/o boundary gate | TBD | TBD | TBD | TBD | TBD | boundary modulation contribution |
| w/o boundary logit fusion | TBD | TBD | TBD | TBD | TBD | explicit boundary channel contribution |
| w/o aux loss | TBD | TBD | TBD | TBD | TBD | auxiliary supervision contribution |
