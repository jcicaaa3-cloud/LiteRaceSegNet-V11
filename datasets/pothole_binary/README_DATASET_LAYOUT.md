# Dataset layout

This repository does **not** include private or third-party road images, masks, pretrained weights, or checkpoints.

Put your permitted image/mask pairs here before training:

```text
datasets/pothole_binary/processed/
├─ train/
│  ├─ images/
│  └─ masks/
├─ val/
│  ├─ images/
│  └─ masks/
└─ test/
   ├─ images/
   └─ masks/
```

Mask values should match the config files in `seg/config/`. For binary training, use `0` for background and `1` for road damage unless your config explicitly defines another mapping.
