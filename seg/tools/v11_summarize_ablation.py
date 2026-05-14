#!/usr/bin/env python3
"""Summarize V11 ablation train logs + parameter/latency profile into CSV/Markdown."""
import argparse
import csv
from pathlib import Path


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def best_from_log(log_path):
    rows = read_csv(log_path)
    if not rows:
        return {}
    def f(row, key):
        try: return float(row.get(key, "nan"))
        except Exception: return float("nan")
    best = max(rows, key=lambda r: f(r, "miou_binary"))
    return {
        "best_epoch": best.get("epoch", ""),
        "best_miou_binary": best.get("miou_binary", ""),
        "best_iou_damage": best.get("iou_damage", ""),
        "best_iou_background": best.get("iou_background", ""),
        "best_pixel_acc": best.get("pixel_acc", ""),
        "best_train_loss": best.get("train_loss", ""),
        "best_val_loss": best.get("val_loss", ""),
        "num_logged_epochs": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="seg/runs/v11_ablation")
    ap.add_argument("--profile_csv", default="seg/runs/v11_ablation/v11_model_profile.csv")
    ap.add_argument("--out_dir", default="seg/runs/v11_ablation/_summary")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = {}
    p_csv = Path(args.profile_csv)
    if p_csv.exists():
        for r in read_csv(p_csv):
            profile[r["variant"]] = r

    rows = []
    for run in sorted(runs_dir.glob("v11_*")):
        if not run.is_dir():
            continue
        log = run / "train_log.csv"
        if not log.exists():
            rows.append({"variant": run.name, "status": "missing_train_log"})
            continue
        row = {"variant": run.name, "status": "ok", "run_dir": str(run)}
        row.update(best_from_log(log))
        if run.name in profile:
            row.update({
                "params_million": profile[run.name].get("params_million", ""),
                "params_total": profile[run.name].get("params_total", ""),
                "latency_ms_per_image": profile[run.name].get("latency_ms_per_image", ""),
                "profile_device": profile[run.name].get("device", ""),
                "description": profile[run.name].get("description", ""),
            })
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No v11 run directories found under {runs_dir}")

    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)

    out_csv = out_dir / "v11_ablation_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    out_md = out_dir / "v11_ablation_summary.md"
    keep = ["variant", "best_miou_binary", "best_iou_damage", "params_million", "latency_ms_per_image", "best_epoch", "status"]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# V11 Ablation Summary\n\n")
        f.write("| " + " | ".join(keep) + " |\n")
        f.write("|" + "|".join(["---"] * len(keep)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in keep) + " |\n")
        f.write("\n## Interpretation guardrail\n")
        f.write("- Treat this as an ablation table only after all variants have been trained under comparable data/epoch settings.\n")
        f.write("- If one variant is a quick-run or early-stopped run, mark it explicitly in the final report.\n")
        f.write("- Do not claim boundary superiority unless no-boundary variants are compared against the full reference under the same protocol.\n")

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
