#!/usr/bin/env bash
set -euo pipefail

# 기존 strict inference 실패 원인: assets/service_demo/input_batch 폴더가 비어 있었음.
# 서비스 데모용 입력 이미지를 val 이미지에서 복사해서 최소 실행 가능하게 만듦.
# 실제 repo/dataset 구조가 다르면 VAL_IMG_DIR만 수정.

VAL_IMG_DIR="datasets/pothole_binary/processed/val/images"
DEMO_DIR="assets/service_demo/input_batch"
mkdir -p "$DEMO_DIR"
find "$VAL_IMG_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) | head -10 | while read -r f; do
  cp -v "$f" "$DEMO_DIR/"
done

echo "Copied demo input images to $DEMO_DIR"
