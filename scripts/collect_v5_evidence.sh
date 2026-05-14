#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/road-damage-segmentation-portfolio
RUN_NAME="${1:-literace_v5_A_corrected_aug_stable_s42}"
RUN_DIR="seg/runs/${RUN_NAME}"
OUT_DIR="final_evidence_v5_update/${RUN_NAME}"

if [ ! -d "$RUN_DIR" ]; then
  echo "[ERROR] run dir not found: $RUN_DIR"
  exit 1
fi

mkdir -p "$OUT_DIR"
cp -f "$RUN_DIR"/train_log.csv "$OUT_DIR"/ 2>/dev/null || true
cp -f "$RUN_DIR"/best.pth "$OUT_DIR"/ 2>/dev/null || true
cp -f "$RUN_DIR"/last.pth "$OUT_DIR"/ 2>/dev/null || true
cp -f seg/config/pothole_binary_literace_v5_*.yaml final_evidence_v5_update/ 2>/dev/null || true

find "$RUN_DIR" -maxdepth 3 -type d \( -iname '*vis*' -o -iname '*pred*' -o -iname '*overlay*' -o -iname '*service*' \) -print0 2>/dev/null | while IFS= read -r -d '' d; do
  rel="${d#$RUN_DIR/}"
  mkdir -p "$OUT_DIR/$rel"
  cp -r "$d"/* "$OUT_DIR/$rel/" 2>/dev/null || true
done

python - <<'INNERPY'
from pathlib import Path
import pandas as pd, json
base = Path('final_evidence_v5_update')
for csv in base.glob('*/train_log.csv'):
    df = pd.read_csv(csv)
    score_col = 'miou_binary' if 'miou_binary' in df.columns else None
    if score_col:
        best = df.loc[df[score_col].idxmax()].to_dict()
        last = df.iloc[-1].to_dict()
        summary = {'csv': str(csv), 'best': best, 'last': last}
        (csv.parent/'summary_best_last.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[SUMMARY]', csv)
        print('best epoch=', best.get('epoch'), 'miou=', best.get('miou_binary'), 'damage_iou=', best.get('iou_damage'))
        print('last epoch=', last.get('epoch'), 'miou=', last.get('miou_binary'), 'damage_iou=', last.get('iou_damage'))
INNERPY

ZIP="LRS_V5_UPDATE_EVIDENCE_${RUN_NAME}.zip"
rm -f "$ZIP"
zip -r "$ZIP" final_evidence_v5_update >/dev/null

echo "[DONE] evidence zip: $ZIP"
