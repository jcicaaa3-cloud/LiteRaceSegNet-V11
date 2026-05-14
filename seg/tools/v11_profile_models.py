#!/usr/bin/env python3
"""Count parameters and benchmark inference latency for V11 config variants."""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "seg"
sys.path.insert(0, str(SEG))
from core.model_select import get_model  # noqa: E402


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def bench(model, device, image_size, warmup=10, iters=50):
    model.eval()
    h, w = int(image_size[0]), int(image_size[1])
    x = torch.randn(1, 3, h, w, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / max(1, iters)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_dir", default="seg/config/v11_ablation")
    ap.add_argument("--out_csv", default="seg/runs/v11_ablation/v11_model_profile.csv")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    cfg_dir = Path(args.config_dir)
    configs = sorted(cfg_dir.glob("v11_*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No v11_*.yaml configs found in {cfg_dir}")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    rows = []
    for cfg_path in configs:
        cfg = load_yaml(cfg_path)
        variant = cfg.get("v11_ablation", {}).get("name", cfg_path.stem)
        model = get_model(cfg).to(device)
        total, trainable = count_params(model)
        image_size = cfg.get("train", {}).get("image_size", [256, 384])
        try:
            latency_ms = bench(model, device, image_size, warmup=args.warmup, iters=args.iters)
            status = "ok"
        except Exception as exc:
            latency_ms = None
            status = f"bench_failed: {exc}"
        rows.append({
            "variant": variant,
            "config": str(cfg_path),
            "device": str(device),
            "image_h": int(image_size[0]),
            "image_w": int(image_size[1]),
            "params_total": total,
            "params_trainable": trainable,
            "params_million": round(total / 1_000_000, 6),
            "latency_ms_per_image": "" if latency_ms is None else round(latency_ms, 4),
            "status": status,
            "description": cfg.get("v11_ablation", {}).get("description", ""),
        })
        print(f"[PROFILE] {variant}: params={total:,}, latency_ms={latency_ms}")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote profile CSV: {out}")


if __name__ == "__main__":
    main()
