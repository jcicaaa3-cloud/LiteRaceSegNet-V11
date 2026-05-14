#!/usr/bin/env python3
"""Seed a manual failure taxonomy CSV from overlay/card image files. Manual labels are still required."""
import argparse
import csv
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual_dir", required=True, help="Directory containing overlay/card images.")
    ap.add_argument("--out_csv", default="seg/runs/v11_case_analysis/failure_taxonomy_manual.csv")
    args = ap.parse_args()

    files = sorted([p for p in Path(args.visual_dir).glob("*") if p.is_file() and p.suffix.lower() in EXTS])
    if not files:
        raise FileNotFoundError(args.visual_dir)

    rows = []
    for p in files:
        key = p.stem
        rows.append({
            "image_key": key,
            "input_image": "",
            "gt_mask": "",
            "pred_mask": "",
            "overlay_image": str(p),
            "primary_failure_type": "manual_required",
            "secondary_failure_type": "",
            "manual_confirmed": "no",
            "short_evidence_note": "",
            "report_use": "candidate",
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[OK] seeded {len(rows)} rows -> {out}")
    print("[NEXT] Manually label primary_failure_type as shadow_confusion / road_texture_confusion / missed_small_damage / over_prediction / good_case / unclear.")


if __name__ == "__main__":
    main()
