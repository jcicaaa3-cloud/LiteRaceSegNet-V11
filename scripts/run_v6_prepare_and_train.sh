#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/road-damage-segmentation-portfolio

NEW_VAL_ZIP="${1:-$HOME/LRS_NEW_VAL_10_PAIRS.zip}"
PYTHON_BIN="python"
if [ -f .venv/bin/activate ]; then
  echo "[ENV] activating .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON_BIN="python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] no python/python3 found"
  exit 1
fi

echo "[CHECK] python package sanity"
$PYTHON_BIN - <<'PY'
mods=['PIL','numpy','torch','yaml']
for m in mods:
    try:
        __import__(m)
        print('[OK]', m)
    except Exception as e:
        print('[MISS]', m, e)
        raise SystemExit(1)
import torch
print('[CUDA]', torch.cuda.is_available())
PY

echo "[LRS-V6] Prepare dataset with new held-out val"
echo "[NEW_VAL_ZIP] $NEW_VAL_ZIP"
$PYTHON_BIN scripts/prepare_v6_new_val.py --new-val-zip "$NEW_VAL_ZIP" --dataset-root datasets/pothole_binary/processed --expected-val-count 10

echo "[VERIFY] original processed dataset after V6 split"
$PYTHON_BIN scripts/verify_pairs.py datasets/pothole_binary/processed

echo "[AUG] rebuilding V6 augmentation into datasets/pothole_binary_aug_v6/processed"
$PYTHON_BIN scripts/make_paired_aug_dataset_v6.py \
  --src datasets/pothole_binary/processed \
  --out datasets/pothole_binary_aug_v6/processed \
  --width 512 \
  --height 384 \
  --image-ext jpg \
  --jpeg-quality 92

echo "[VERIFY] V6 augmented dataset"
$PYTHON_BIN scripts/verify_pairs.py datasets/pothole_binary_aug_v6/processed

echo "[START] V6-A new-val balanced run"
$PYTHON_BIN seg/train_literace.py --config seg/config/pothole_binary_literace_v6_A_newval_balanced.yaml

echo "[DONE] V6-A training finished. Collecting evidence."
bash scripts/collect_v6_evidence.sh literace_v6_A_newval_balanced_s42 || true

echo "[NEXT optional] recall run:"
echo "$PYTHON_BIN seg/train_literace.py --config seg/config/pothole_binary_literace_v6_B_newval_recall.yaml"
echo "[NEXT optional] precision run:"
echo "$PYTHON_BIN seg/train_literace.py --config seg/config/pothole_binary_literace_v6_C_newval_precision.yaml"
