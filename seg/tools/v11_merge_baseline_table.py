#!/usr/bin/env python3
"""Merge V11 ablation summary with manually supplied SegFormer/DeepLab baseline rows."""
import argparse
import csv
from pathlib import Path


def read_csv(p):
    if not Path(p).exists():
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_csv", default="seg/runs/v11_ablation/_summary/v11_ablation_summary.csv")
    ap.add_argument("--baseline_csv", default="v11_templates/V11_BASELINE_COMPARISON_TEMPLATE.csv")
    ap.add_argument("--out_csv", default="seg/runs/v11_baseline_comparison/v11_baseline_comparison.csv")
    ap.add_argument("--out_md", default="seg/runs/v11_baseline_comparison/v11_baseline_comparison.md")
    args = ap.parse_args()

    rows = []
    for r in read_csv(args.ablation_csv):
        if r.get("variant") == "v11_full_reference":
            rows.append({
                "model": "LiteRaceSegNet",
                "variant": "v11_full_reference",
                "source": "V11 ablation summary",
                "checkpoint_or_run": r.get("run_dir", ""),
                "best_miou_binary": r.get("best_miou_binary", ""),
                "iou_damage": r.get("best_iou_damage", ""),
                "params_million": r.get("params_million", ""),
                "latency_ms_per_image": r.get("latency_ms_per_image", ""),
                "device": r.get("profile_device", ""),
                "input_size": "",
                "notes": "Full reference model from V11 matrix",
            })
    # Include user-filled baseline rows if values are present.
    for r in read_csv(args.baseline_csv):
        rows.append(r)

    out_csv = Path(args.out_csv); out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model","variant","source","checkpoint_or_run","best_miou_binary","iou_damage","params_million","latency_ms_per_image","device","input_size","notes"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# V11 Baseline Comparison Table\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"]*len(fields)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k,"")) for k in fields) + " |\n")
        f.write("\n## Guardrail\n")
        f.write("Do not compare models as final evidence unless dataset split, image size, epoch budget, and evaluation script are clearly stated.\n")
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
