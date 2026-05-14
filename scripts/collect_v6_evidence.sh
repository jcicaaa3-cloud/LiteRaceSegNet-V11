#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/road-damage-segmentation-portfolio
RUN_NAME="${1:-literace_v6_A_newval_balanced_s42}"
RUN_DIR="seg/runs/${RUN_NAME}"
OUT_BASE="final_evidence_v6_newval"
OUT_DIR="${OUT_BASE}/${RUN_NAME}"
if [ ! -d "$RUN_DIR" ]; then echo "[ERROR] run dir not found: $RUN_DIR"; exit 1; fi
mkdir -p "$OUT_DIR"
cp -f "$RUN_DIR"/train_log.csv "$OUT_DIR"/ 2>/dev/null || true
cp -f "$RUN_DIR"/best.pth "$OUT_DIR"/ 2>/dev/null || true
cp -f "$RUN_DIR"/last.pth "$OUT_DIR"/ 2>/dev/null || true
cp -f datasets/pothole_binary/processed/V6_DATASET_MANIFEST.json "$OUT_BASE"/ 2>/dev/null || true
cp -f seg/config/pothole_binary_literace_v6_*.yaml "$OUT_BASE"/ 2>/dev/null || true
python scripts/verify_pairs.py datasets/pothole_binary/processed > "$OUT_BASE/dataset_verify_original.txt" 2>&1 || true
python scripts/verify_pairs.py datasets/pothole_binary_aug_v6/processed > "$OUT_BASE/dataset_verify_aug_v6.txt" 2>&1 || true
find "$RUN_DIR" -maxdepth 4 -type d \( -iname '*vis*' -o -iname '*pred*' -o -iname '*overlay*' -o -iname '*service*' -o -iname '*mask*' \) -print0 2>/dev/null | while IFS= read -r -d '' d; do
  rel="${d#$RUN_DIR/}"
  mkdir -p "$OUT_DIR/$rel"
  cp -r "$d"/* "$OUT_DIR/$rel/" 2>/dev/null || true
done
python - <<'PY'
from pathlib import Path
import csv, json, math
base=Path('final_evidence_v6_newval')
for csvp in base.glob('*/train_log.csv'):
    rows=[]
    with csvp.open(newline='', encoding='utf-8') as f:
        reader=csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        continue
    def num(r,k):
        try: return float(r.get(k,''))
        except Exception: return float('nan')
    # known columns from current trainer: miou_binary, iou_damage. Fallbacks included.
    miou_key = 'miou_binary' if 'miou_binary' in rows[0] else ('mIoU' if 'mIoU' in rows[0] else None)
    dmg_key = 'iou_damage' if 'iou_damage' in rows[0] else ('damageIoU' if 'damageIoU' in rows[0] else None)
    if miou_key:
        best=max(rows, key=lambda r: num(r,miou_key))
    else:
        best=rows[-1]
    last=rows[-1]
    summary={'csv':str(csvp),'miou_key':miou_key,'damage_key':dmg_key,'best':best,'last':last}
    (csvp.parent/'summary_best_last.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[SUMMARY]', csvp)
    print('best epoch=', best.get('epoch'), 'miou=', best.get(miou_key) if miou_key else None, 'damage_iou=', best.get(dmg_key) if dmg_key else None)
    print('last epoch=', last.get('epoch'), 'miou=', last.get(miou_key) if miou_key else None, 'damage_iou=', last.get(dmg_key) if dmg_key else None)
PY
find "$OUT_BASE" -maxdepth 4 -type f | sort > "$OUT_BASE/FILE_LIST.txt"
ZIP="LRS_V6_NEWVAL_EVIDENCE_${RUN_NAME}.zip"
rm -f "$ZIP"
zip -r "$ZIP" "$OUT_BASE" >/dev/null
echo "[DONE] evidence zip: $ZIP"
