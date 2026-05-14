#!/usr/bin/env python3
"""
V11 robust binary + boundary metrics.
Fixes the earlier failure mode: "No matching prediction/GT pairs by stem".

Use examples:
  python seg/tools/v11_boundary_metrics_fixed.py --pred_dir seg/outputs/literace_v3_strict/masks --gt_dir datasets/pothole_binary/processed/val/masks --out_csv seg/runs/v11_boundary/boundary_metrics.csv --allow_order_fallback
  python seg/tools/v11_boundary_metrics_fixed.py --pred_dir ... --gt_dir ... --debug_unmatched
"""
import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUFFIX_PATTERNS = [
    r"_post_thr[0-9.]+_min\d+$",
    r"_thr[0-9.]+$",
    r"_raw_argmax$",
    r"_pred$",
    r"_prediction$",
    r"_mask$",
    r"_label$",
    r"_gt$",
    r"_boundary$",
    r"_edge$",
    r"_overlay$",
    r"_card$",
]


def norm_key(p: Path) -> str:
    s = p.stem.lower()
    s = re.sub(r"[^a-z0-9가-힣]+", "_", s)
    changed = True
    while changed:
        changed = False
        for pat in SUFFIX_PATTERNS:
            ns = re.sub(pat, "", s)
            if ns != s:
                s = ns; changed = True
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def index_dir(d):
    files = sorted([p for p in Path(d).glob("*") if p.is_file() and p.suffix.lower() in EXTS])
    idx = {}
    for p in files:
        idx.setdefault(norm_key(p), []).append(p)
    return idx, files


def read_bin(path, shape=None):
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise ValueError(f"unreadable image: {path}")
    if shape is not None and a.shape[:2] != shape:
        a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return a > 0


def edge_mask(mask, width=3):
    m = (mask.astype(np.uint8) * 255)
    # Morphological boundary is more stable than Canny for binary masks.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, width), max(1, width)))
    dil = cv2.dilate(m, k)
    ero = cv2.erode(m, k)
    return (dil != ero)


def compute(pred, gt, tolerance=2):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum(); fp = np.logical_and(pred, ~gt).sum(); fn = np.logical_and(~pred, gt).sum(); tn = np.logical_and(~pred, ~gt).sum()
    eps = 1e-9
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)

    pe = edge_mask(pred, width=3)
    ge = edge_mask(gt, width=3)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    ge_d = cv2.dilate(ge.astype(np.uint8), k).astype(bool)
    pe_d = cv2.dilate(pe.astype(np.uint8), k).astype(bool)
    btp_p = np.logical_and(pe, ge_d).sum()
    btp_g = np.logical_and(ge, pe_d).sum()
    bprec = btp_p / (pe.sum() + eps)
    brec = btp_g / (ge.sum() + eps)
    bf1 = 2 * bprec * brec / (bprec + brec + eps)

    n_pred_comp, _ = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    n_gt_comp, _ = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    return {
        "iou_damage": iou, "dice_damage": dice, "precision": prec, "recall": rec,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "boundary_precision": bprec, "boundary_recall": brec, "boundary_f1": bf1,
        "pred_area_ratio": float(pred.mean()), "gt_area_ratio": float(gt.mean()),
        "pred_components": int(max(0, n_pred_comp - 1)), "gt_components": int(max(0, n_gt_comp - 1)),
    }


def fmt(v):
    return round(float(v), 6) if isinstance(v, (float, np.floating)) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tolerance", type=int, default=2)
    ap.add_argument("--allow_order_fallback", action="store_true", help="If stems do not match but file counts match, pair by sorted order.")
    ap.add_argument("--debug_unmatched", action="store_true")
    args = ap.parse_args()

    pred_idx, pred_files = index_dir(args.pred_dir)
    gt_idx, gt_files = index_dir(args.gt_dir)
    keys = sorted(set(pred_idx) & set(gt_idx))
    pairs = []
    match_mode = "stem"
    for k in keys:
        pairs.append((k, pred_idx[k][0], gt_idx[k][0]))

    if not pairs and args.allow_order_fallback and len(pred_files) == len(gt_files) and len(pred_files) > 0:
        match_mode = "sorted_order_fallback"
        pairs = [(f"order_{i:04d}", p, g) for i, (p, g) in enumerate(zip(pred_files, gt_files), 1)]

    if not pairs:
        if args.debug_unmatched:
            print("[DEBUG] pred keys sample:", sorted(pred_idx.keys())[:20])
            print("[DEBUG] gt keys sample:", sorted(gt_idx.keys())[:20])
            print("[DEBUG] pred files:", [p.name for p in pred_files[:20]])
            print("[DEBUG] gt files:", [p.name for p in gt_files[:20]])
        raise FileNotFoundError("No matching prediction/GT pairs. Try --allow_order_fallback or inspect --debug_unmatched output.")

    rows = []
    sums = {"tp":0, "fp":0, "fn":0, "tn":0}
    for key, pp, gp in pairs:
        gt = read_bin(gp)
        pred = read_bin(pp, gt.shape)
        m = compute(pred, gt, tolerance=args.tolerance)
        for k in sums:
            sums[k] += int(m[k])
        row = {"image_key": key, "pred_file": str(pp), "gt_file": str(gp), "match_mode": match_mode}
        row.update({k: fmt(v) for k, v in m.items()})
        rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # aggregate from pixel sums only for segmentation; boundary is averaged per image.
    eps = 1e-9
    agg_iou = sums["tp"] / (sums["tp"] + sums["fp"] + sums["fn"] + eps)
    agg_dice = 2 * sums["tp"] / (2 * sums["tp"] + sums["fp"] + sums["fn"] + eps)
    mean_bf1 = float(np.mean([r["boundary_f1"] for r in rows]))
    print(f"[OK] pairs={len(rows)} mode={match_mode} -> {out}")
    print("[AGG]", {"iou_damage": round(agg_iou, 6), "dice_damage": round(agg_dice, 6), "mean_boundary_f1": round(mean_bf1, 6)})


if __name__ == "__main__":
    main()
