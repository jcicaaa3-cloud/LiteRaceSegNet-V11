#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/road-damage-segmentation-portfolio
RUN_NAME="${1:-literace_v6_A_newval_balanced_s42}"
RUN_DIR="seg/runs/${RUN_NAME}"
DATA_ORIG="datasets/pothole_binary/processed"
DATA_AUG="datasets/pothole_binary_aug_v6/processed"
OUT_BASE="final_evidence_v6_newval_with_visuals"
OUT_DIR="${OUT_BASE}/${RUN_NAME}"
ZIP="LRS_V6_NEWVAL_EVIDENCE_WITH_VISUALS_${RUN_NAME}.zip"

if [ -d .venv ]; then source .venv/bin/activate; fi
if [ ! -d "$RUN_DIR" ]; then echo "[ERROR] run dir not found: $RUN_DIR"; exit 1; fi
mkdir -p "$OUT_DIR"

echo "[CHECK] counts and leakage"
{
  echo "=== original current dataset ==="
  printf "train images: "; find "$DATA_ORIG/train/images" -type f | wc -l
  printf "train masks : "; find "$DATA_ORIG/train/masks" -type f | wc -l
  printf "val images  : "; find "$DATA_ORIG/val/images" -type f | wc -l
  printf "val masks   : "; find "$DATA_ORIG/val/masks" -type f | wc -l
  echo "=== v6 leakage check ==="
  printf "oldval_pothole in train should be 10: "; find "$DATA_ORIG/train/images" -name "oldval_pothole_*" | wc -l
  printf "oldval_newval in train should be 0 : "; find "$DATA_ORIG/train/images" -name "oldval_newval_*" | wc -l
  printf "newval in train should be 0        : "; find "$DATA_ORIG/train/images" -name "newval_*" | wc -l
  printf "newval in val should be 10         : "; find "$DATA_ORIG/val/images" -name "newval_*" | wc -l
  echo "=== augmented dataset ==="
  printf "train images: "; find "$DATA_AUG/train/images" -type f | wc -l
  printf "train masks : "; find "$DATA_AUG/train/masks" -type f | wc -l
  printf "val images  : "; find "$DATA_AUG/val/images" -type f | wc -l
  printf "val masks   : "; find "$DATA_AUG/val/masks" -type f | wc -l
} | tee "$OUT_BASE/V6_COUNTS_AND_LEAKAGE_CHECK.txt"

python scripts/verify_pairs.py "$DATA_ORIG" > "$OUT_BASE/dataset_verify_original.txt" 2>&1 || true
python scripts/verify_pairs.py "$DATA_AUG" > "$OUT_BASE/dataset_verify_aug_v6.txt" 2>&1 || true

cp -f "$RUN_DIR"/train_log.csv "$OUT_DIR"/ 2>/dev/null || true
cp -f "$RUN_DIR"/best.pth "$OUT_DIR"/ 2>/dev/null || true
cp -f "$RUN_DIR"/last.pth "$OUT_DIR"/ 2>/dev/null || true
cp -f seg/config/pothole_binary_literace_v6_*.yaml "$OUT_BASE"/ 2>/dev/null || true
cp -f "$DATA_ORIG"/V6_DATASET_MANIFEST.json "$OUT_BASE"/ 2>/dev/null || true

# generate visuals from best checkpoint on current clean val set
python scripts/make_v6_visual_evidence.py \
  --run "$RUN_NAME" \
  --checkpoint best \
  --data-root "$DATA_ORIG" \
  --out "$RUN_DIR/v6_visual_evidence_best" \
  --max-images 10 \
  --norm imagenet

# last checkpoint visual is optional. If training was interrupted and last is odd, do not fail collection.
python scripts/make_v6_visual_evidence.py \
  --run "$RUN_NAME" \
  --checkpoint last \
  --data-root "$DATA_ORIG" \
  --out "$RUN_DIR/v6_visual_evidence_last" \
  --max-images 10 \
  --norm imagenet || true

# copy visual folders into evidence
for d in "$RUN_DIR"/v6_visual_evidence_*; do
  [ -d "$d" ] || continue
  rel="$(basename "$d")"
  mkdir -p "$OUT_DIR/$rel"
  cp -a "$d"/. "$OUT_DIR/$rel"/
done

python - <<'PY'
from pathlib import Path
import csv, json
base=Path('final_evidence_v6_newval_with_visuals')
for csvp in base.glob('*/train_log.csv'):
    rows=list(csv.DictReader(open(csvp, newline='', encoding='utf-8')))
    if not rows: continue
    def num(r,k):
        try: return float(r.get(k,''))
        except: return -1
    miou_key='miou_binary' if 'miou_binary' in rows[0] else ('mIoU' if 'mIoU' in rows[0] else None)
    dmg_key='iou_damage' if 'iou_damage' in rows[0] else ('damageIoU' if 'damageIoU' in rows[0] else None)
    best=max(enumerate(rows,start=1), key=lambda ir: num(ir[1], miou_key) if miou_key else ir[0])
    last=(len(rows), rows[-1])
    summary={'csv':str(csvp),'miou_key':miou_key,'damage_key':dmg_key,'best_row_number':best[0],'best':best[1],'last_row_number':last[0],'last':last[1]}
    (csvp.parent/'summary_best_last.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[SUMMARY]', 'best_row=', best[0], 'miou=', best[1].get(miou_key) if miou_key else None, 'damage=', best[1].get(dmg_key) if dmg_key else None)
PY

find "$OUT_BASE" -maxdepth 8 -type f | sort > "$OUT_BASE/FILE_LIST.txt"
rm -f "$ZIP"
zip -r "$ZIP" "$OUT_BASE" >/dev/null
ls -lah "$ZIP"
echo "[DONE] evidence with visuals: $ZIP"
