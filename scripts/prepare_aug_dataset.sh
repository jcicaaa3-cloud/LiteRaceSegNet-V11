#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-datasets/pothole_binary/processed}"
OUT="${2:-datasets/pothole_binary_aug/processed}"
python scripts/verify_pairs.py "$SRC"
python scripts/make_paired_aug_dataset_fast.py --src "$SRC" --out "$OUT" --width 512 --height 384 --image-ext jpg --jpeg-quality 92
python scripts/verify_pairs.py "$OUT"
echo "[NEXT] python seg/train_literace.py --config seg/config/pothole_binary_literace_v4_A_aug_highres_damage_boost.yaml"
