#!/usr/bin/env python3
"""Create objective easy/hard case groups from GT mask ratio and component count."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_bin(p):
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise ValueError(f"unreadable: {p}")
    return a > 0


def group_by_area(ratio, small_thr, large_thr):
    if ratio <= small_thr:
        return "small_or_hard_candidate"
    if ratio >= large_thr:
        return "large_or_easy_candidate"
    return "medium"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_csv", default="seg/runs/v11_case_analysis/easy_hard_cases.csv")
    ap.add_argument("--small_thr", type=float, default=None, help="If omitted, uses 33rd percentile of nonzero GT area ratios.")
    ap.add_argument("--large_thr", type=float, default=None, help="If omitted, uses 66th percentile of nonzero GT area ratios.")
    args = ap.parse_args()

    gt_files = sorted([p for p in Path(args.gt_dir).glob("*") if p.is_file() and p.suffix.lower() in EXTS])
    if not gt_files:
        raise FileNotFoundError(args.gt_dir)

    tmp = []
    for p in gt_files:
        m = read_bin(p)
        ratio = float(m.mean())
        ncomp, labels = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        areas = []
        for cid in range(1, ncomp):
            areas.append(int((labels == cid).sum()))
        tmp.append({
            "image_key": p.stem,
            "gt_file": str(p),
            "gt_area_ratio": ratio,
            "gt_positive_pixels": int(m.sum()),
            "gt_components": int(max(0, ncomp - 1)),
            "largest_component_pixels": max(areas) if areas else 0,
        })

    nonzero = np.array([r["gt_area_ratio"] for r in tmp if r["gt_area_ratio"] > 0], dtype=float)
    if len(nonzero) == 0:
        small_thr = args.small_thr if args.small_thr is not None else 0.0
        large_thr = args.large_thr if args.large_thr is not None else 0.0
    else:
        small_thr = float(args.small_thr) if args.small_thr is not None else float(np.percentile(nonzero, 33))
        large_thr = float(args.large_thr) if args.large_thr is not None else float(np.percentile(nonzero, 66))

    rows = []
    for r in tmp:
        group = group_by_area(r["gt_area_ratio"], small_thr, large_thr)
        # Extra flag: fragmented masks can be hard even when total area is not tiny.
        fragmentation_flag = "fragmented" if r["gt_components"] >= 3 else "not_fragmented"
        row = dict(r)
        row.update({
            "area_group": group,
            "fragmentation_flag": fragmentation_flag,
            "small_thr_used": round(small_thr, 8),
            "large_thr_used": round(large_thr, 8),
            "interpretation_guardrail": "area_group is objective-by-GT-mask, not subjective visual difficulty",
        })
        rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = {}
    for r in rows:
        summary[r["area_group"]] = summary.get(r["area_group"], 0) + 1
    print(f"[OK] wrote {out}")
    print("[THRESHOLDS]", {"small_thr": round(small_thr, 8), "large_thr": round(large_thr, 8)})
    print("[GROUP_COUNTS]", summary)


if __name__ == "__main__":
    main()
