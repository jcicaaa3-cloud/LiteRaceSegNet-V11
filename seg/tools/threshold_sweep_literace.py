"""Threshold/min-area sweep for LiteRaceSegNet validation evidence.

Purpose
-------
This tool does not train, relabel, or replace masks. It loads a trained
LiteRaceSegNet checkpoint, evaluates damage probability thresholds on the
validation split, and optionally removes tiny connected components. The output
CSV is meant to justify a conservative inference setting instead of manually
choosing visually pleasant masks.

Example
-------
python seg/tools/threshold_sweep_literace.py ^
  --config seg/config/pothole_binary_literace_one_month_v3.yaml ^
  --ckpt seg/runs/literace_one_month_v3/best.pth ^
  --out_csv final_evidence/02_metrics_and_compare/literace_threshold_sweep_v3.csv ^
  --thresholds 0.45,0.50,0.55,0.60,0.65,0.70,0.75 ^
  --min_areas 0,60,120,180,240
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SEG_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SEG_DIR.parent
if str(SEG_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_DIR))

from core.model_select import get_model
from core.save import load_state
from core.train_utils import get_device, load_yaml, set_seed
from train_literace import BinaryPotholeDataset


def resolve(path_text: str | Path) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def parse_floats(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def parse_ints(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty int list")
    return vals


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if min_area <= 1 or int(mask_u8.sum()) == 0:
        return mask_u8.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros_like(mask_u8, dtype=np.uint8)
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            keep[labels == lab] = 1
    return keep.astype(bool)


def empty_totals() -> Dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "pred_pos": 0, "gt_pos": 0, "valid_px": 0, "component_sum": 0}


def component_count(mask: np.ndarray) -> int:
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return 0
    n, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return max(0, int(n) - 1)


def update_totals(totals: Dict[str, int], pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> None:
    pred = pred.astype(bool)[valid]
    gt = gt.astype(bool)[valid]
    totals["tp"] += int(np.logical_and(pred, gt).sum())
    totals["tn"] += int(np.logical_and(~pred, ~gt).sum())
    totals["fp"] += int(np.logical_and(pred, ~gt).sum())
    totals["fn"] += int(np.logical_and(~pred, gt).sum())
    totals["pred_pos"] += int(pred.sum())
    totals["gt_pos"] += int(gt.sum())
    totals["valid_px"] += int(valid.sum())


def finish(t: Dict[str, int], image_count: int) -> Dict[str, float]:
    tp, tn, fp, fn = t["tp"], t["tn"], t["fp"], t["fn"]
    eps = 1e-9
    valid = max(1, t["valid_px"])
    iou_damage = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "miou_binary": (iou_damage + iou_bg) / 2,
        "iou_damage": iou_damage,
        "iou_background": iou_bg,
        "dice_damage": dice,
        "precision": precision,
        "recall": recall,
        "pixel_acc": pixel_acc,
        "pred_positive_ratio_percent": 100.0 * t["pred_pos"] / valid,
        "gt_positive_ratio_percent": 100.0 * t["gt_pos"] / valid,
        "false_positive_ratio_percent": 100.0 * fp / valid,
        "avg_components_per_image": t["component_sum"] / max(1, image_count),
    }


def load_probabilities(cfg: Dict, ckpt: Path, device: torch.device) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    dataset = BinaryPotholeDataset(cfg, split="val", augment=False)
    model = get_model(cfg).to(device)
    load_state(str(ckpt), model, map_location=device.type)
    model.eval()
    ignore_index = int(cfg.get("data", {}).get("ignore_index", 255))
    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            x = sample["pixel_values"].unsqueeze(0).to(device)
            labels = sample["labels"].numpy()
            out = model(x)["out"]
            out = F.interpolate(out, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.softmax(out, dim=1)[0, 1].detach().cpu().numpy()
            gt = labels == 1
            valid = labels != ignore_index
            image_name = Path(dataset.samples[idx][0]).name
            rows.append((prob, gt, valid, image_name))
    return rows


def pick_recommended(rows: Sequence[Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    # One choice for metrics/reporting, one choice for demo/inference conservativeness.
    def f(row: Dict[str, str], key: str) -> float:
        return float(row[key])

    metric_best = max(rows, key=lambda r: (f(r, "miou_binary"), f(r, "iou_damage"), f(r, "precision")))
    conservative_best = max(rows, key=lambda r: (f(r, "precision"), f(r, "iou_damage"), -f(r, "false_positive_ratio_percent")))
    return metric_best, conservative_best


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep LiteRaceSegNet thresholds/min-area filters on validation split")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    p.add_argument("--min_areas", default="0,60,120,180,240")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = p.parse_args()

    cfg = load_yaml(str(resolve(args.config)))
    cfg.setdefault("model", {})["name"] = "lite_race"
    if args.device:
        cfg["device"] = args.device
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg)
    ckpt = resolve(args.ckpt)
    probs = load_probabilities(cfg, ckpt, device)
    thresholds = parse_floats(args.thresholds)
    min_areas = parse_ints(args.min_areas)

    results: List[Dict[str, str]] = []
    for thr in thresholds:
        for min_area in min_areas:
            totals = empty_totals()
            for prob, gt, valid, _image_name in probs:
                pred = prob >= float(thr)
                pred = remove_small_components(pred, int(min_area))
                totals["component_sum"] += component_count(pred)
                update_totals(totals, pred, gt, valid)
            m = finish(totals, len(probs))
            row = {
                "threshold": f"{thr:.2f}",
                "min_area_pixels": str(min_area),
                "val_images": str(len(probs)),
                **{k: f"{v:.6f}" for k, v in m.items()},
                "tp": str(totals["tp"]),
                "fp": str(totals["fp"]),
                "fn": str(totals["fn"]),
                "tn": str(totals["tn"]),
            }
            results.append(row)

    out_csv = resolve(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    metric_best, conservative_best = pick_recommended(results)
    rec_path = out_csv.with_suffix(".recommendation.txt")
    rec_path.write_text(
        "LiteRaceSegNet threshold sweep recommendation\n"
        "============================================\n\n"
        "[Metric-first setting]\n"
        f"threshold={metric_best['threshold']}, min_area_pixels={metric_best['min_area_pixels']}, "
        f"mIoU={metric_best['miou_binary']}, damageIoU={metric_best['iou_damage']}, "
        f"precision={metric_best['precision']}, recall={metric_best['recall']}\n\n"
        "[Conservative inference setting]\n"
        f"threshold={conservative_best['threshold']}, min_area_pixels={conservative_best['min_area_pixels']}, "
        f"mIoU={conservative_best['miou_binary']}, damageIoU={conservative_best['iou_damage']}, "
        f"precision={conservative_best['precision']}, recall={conservative_best['recall']}\n\n"
        "Use the metric-first setting for quantitative comparison and the conservative setting for service/demo masks. "
        "Report them separately. Do not replace raw model evidence.\n",
        encoding="utf-8",
    )
    print(f"[OK] sweep rows={len(results)} -> {out_csv}")
    print(f"[OK] recommendation -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
