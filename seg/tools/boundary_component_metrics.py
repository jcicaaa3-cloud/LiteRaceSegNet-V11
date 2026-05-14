"""Boundary and component-level audit for binary segmentation masks.

This tool complements mIoU. It answers questions such as:
- Does the mask preserve irregular pothole boundaries?
- Is the prediction fragmented into many small components?
- Is over-detection caused by broad road texture coverage?

It does not modify predictions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUFFIXES = [
    "_post_thr0.75_min240",
    "_post_thr0.70_min240",
    "_post_thr0.65_min180",
    "_post_thr0.60_min120",
    "_raw_argmax",
    "_pred",
    "_mask_color",
    "_mask",
    "_label",
    "_gt",
]


def stem_key(path: Path) -> str:
    s = path.stem
    changed = True
    while changed:
        changed = False
        for suffix in SUFFIXES:
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                changed = True
    return s


def index_files(folder: Path) -> Dict[str, Path]:
    return {stem_key(p): p for p in sorted(folder.glob("*")) if p.is_file() and p.suffix.lower() in EXTS}


def read_binary(path: Path, shape: Tuple[int, int] | None = None) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"unreadable image: {path}")
    if shape and arr.shape[:2] != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr > 0


def boundary(mask: np.ndarray, radius: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    k = max(3, int(radius) * 2 + 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m, kernel, iterations=1)
    return (dil != ero)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, radius: int) -> Tuple[float, float, float]:
    pb = boundary(pred, radius)
    gb = boundary(gt, radius)
    kernel = np.ones((max(3, radius * 2 + 1), max(3, radius * 2 + 1)), dtype=np.uint8)
    gb_dil = cv2.dilate(gb.astype(np.uint8), kernel, iterations=1).astype(bool)
    pb_dil = cv2.dilate(pb.astype(np.uint8), kernel, iterations=1).astype(bool)
    precision = np.logical_and(pb, gb_dil).sum() / max(1, int(pb.sum()))
    recall = np.logical_and(gb, pb_dil).sum() / max(1, int(gb.sum()))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return float(f1), float(precision), float(recall)


def component_stats(mask: np.ndarray) -> Tuple[int, int, int, float]:
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return 0, 0, 0, 0.0
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    return len(areas), min(areas), max(areas), float(np.mean(areas))


def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    eps = 1e-9
    return {
        "iou_damage": tp / (tp + fp + fn + eps),
        "dice_damage": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "pixel_acc": (tp + tn) / (tp + tn + fp + fn + eps),
        "damage_ratio_pred_percent": 100.0 * pred.mean(),
        "damage_ratio_gt_percent": 100.0 * gt.mean(),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate boundary F1 and component statistics for binary masks")
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--boundary_radius", type=int, default=2)
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    preds = index_files(pred_dir)
    gts = index_files(gt_dir)
    keys = sorted(set(preds) & set(gts))
    if not keys:
        raise FileNotFoundError("No matching prediction/GT pairs by stem")

    rows: List[Dict[str, str]] = []
    for key in keys:
        gt = read_binary(gts[key])
        pred = read_binary(preds[key], shape=gt.shape)
        px = pixel_metrics(pred, gt)
        bf1, bp, br = boundary_f1(pred, gt, args.boundary_radius)
        comp_n, comp_min, comp_max, comp_mean = component_stats(pred)
        row = {
            "image_key": key,
            "pred": str(preds[key]),
            "gt": str(gts[key]),
            **{k: f"{v:.6f}" if isinstance(v, float) else str(v) for k, v in px.items()},
            "boundary_f1": f"{bf1:.6f}",
            "boundary_precision": f"{bp:.6f}",
            "boundary_recall": f"{br:.6f}",
            "component_count": str(comp_n),
            "component_area_min": str(comp_min),
            "component_area_max": str(comp_max),
            "component_area_mean": f"{comp_mean:.2f}",
        }
        rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] boundary/component metrics pairs={len(rows)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
