#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
runs = sorted((ROOT / "seg" / "runs").glob("literace_v4_*"))
rows = []
for run in runs:
    log = run / "train_log.csv"
    if not log.exists():
        continue
    df = pd.read_csv(log)
    if "miou_binary" not in df.columns:
        continue
    best = df.loc[df["miou_binary"].idxmax()]
    best_dmg = df.loc[df["iou_damage"].idxmax()] if "iou_damage" in df.columns else best
    last = df.iloc[-1]
    rows.append({
        "run": run.name,
        "epochs": int(df["epoch"].max()),
        "best_epoch_by_miou": int(best["epoch"]),
        "best_miou_binary": float(best["miou_binary"]),
        "damage_iou_at_best_miou": float(best.get("iou_damage", 0)),
        "pixel_acc_at_best_miou": float(best.get("pixel_acc", 0)),
        "best_damage_iou": float(best_dmg.get("iou_damage", 0)),
        "epoch_best_damage_iou": int(best_dmg["epoch"]),
        "last_miou_binary": float(last["miou_binary"]),
        "last_damage_iou": float(last.get("iou_damage", 0)),
        "lr_at_best_miou": float(best.get("lr", 0)),
    })

out_dir = ROOT / "final_evidence" / "07_v4_tuning_summary"
out_dir.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame(rows).sort_values("best_miou_binary", ascending=False) if rows else pd.DataFrame()
summary.to_csv(out_dir / "v4_training_summary.csv", index=False)
print(summary.to_string(index=False) if len(summary) else "No v4 train_log.csv files found.")

if len(summary):
    best = summary.iloc[0]
    (out_dir / "V4_BEST_RUN_README_KO.md").write_text(f"""# LiteRaceSegNet v4 튜닝 결과 요약

- validation mIoU 기준 선택 run: `{best['run']}`
- best mIoU: {best['best_miou_binary']:.6f}
- damage IoU: {best['damage_iou_at_best_miou']:.6f}
- best epoch: {int(best['best_epoch_by_miou'])}

주의: 이 값은 실제 재학습 로그에서 추출한 값만 기록한다. 수동으로 수치를 고치지 않는다.
""", encoding="utf-8")
