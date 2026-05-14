#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/road-damage-segmentation-portfolio

echo "[LRS-V5] Corrected config update"
echo "[Reason] V4 log started near lr=0.001 although config had optimizer.lr=0.00040."
echo "[Reason] Existing train config style uses train.base_lr, train.class_weights, train.loss."

if [ ! -f seg/config/pothole_binary_literace_v5_A_corrected_aug_stable.yaml ]; then
  echo "[ERROR] v5 config not found. Run: unzip -o ~/LRS_V5_UPDATE_PATCH.zip -d /home/ubuntu/road-damage-segmentation-portfolio"
  exit 1
fi

if [ ! -d datasets/pothole_binary_aug/processed/train/images ]; then
  echo "[INFO] Augmented dataset not found. Rebuilding it."
  python scripts/verify_pairs.py datasets/pothole_binary/processed
  python scripts/make_paired_aug_dataset_fast.py \
    --src datasets/pothole_binary/processed \
    --out datasets/pothole_binary_aug/processed \
    --width 512 \
    --height 384 \
    --image-ext jpg \
    --jpeg-quality 92
fi

python scripts/verify_pairs.py datasets/pothole_binary_aug/processed

echo "[START] V5-A corrected stable run"
python seg/train_literace.py --config seg/config/pothole_binary_literace_v5_A_corrected_aug_stable.yaml

echo "[DONE] V5-A finished. Collecting evidence."
bash scripts/collect_v5_evidence.sh literace_v5_A_corrected_aug_stable_s42 || true

echo "[NEXT] If V5-A is worse or recall is weak:"
echo "python seg/train_literace.py --config seg/config/pothole_binary_literace_v5_B_corrected_aug_recall.yaml"
echo "[NEXT] If false positives are too large:"
echo "python seg/train_literace.py --config seg/config/pothole_binary_literace_v5_C_corrected_aug_precision.yaml"
echo "[NEXT] If CUDA OOM occurs:"
echo "python seg/train_literace.py --config seg/config/pothole_binary_literace_v5_D_safe_oom.yaml"
