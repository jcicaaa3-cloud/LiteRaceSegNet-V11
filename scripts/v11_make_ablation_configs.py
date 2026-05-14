#!/usr/bin/env python3
"""
V11 ablation config generator for LiteRaceSegNet.
Run from repository root:
  python scripts/v11_make_ablation_configs.py --base seg/config/pothole_binary_literace_one_month_v3_balanced.yaml

This does not train anything. It only creates reproducible config files for the V11 research matrix.
"""
import argparse
import copy
from pathlib import Path
import yaml


def deep_set(d, path, value):
    cur = d
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="seg/config/pothole_binary_literace_one_month_v3_balanced.yaml")
    ap.add_argument("--out_dir", default="seg/config/v11_ablation")
    ap.add_argument("--epochs", type=int, default=None, help="Optional quick-run override. Omit for base config epochs.")
    args = ap.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    # Note: a literal "backbone removal" would make the model invalid.
    # V11 therefore uses a backbone-capacity ablation as a defensible proxy,
    # plus context-module and branch/fusion removals that are directly supported by LiteRaceSegNet.
    variants = {
        "v11_full_reference": {
            "description": "Full LiteRaceSegNet reference: detail branch + boundary gate + boundary-logit fusion + LiteASPP context.",
            "changes": [],
        },
        "v11_no_detail_branch": {
            "description": "Remove the high-resolution detail branch contribution.",
            "changes": [(('model','use_detail_branch'), False)],
        },
        "v11_no_boundary_gate": {
            "description": "Keep boundary prediction but disable boundary-guided gating.",
            "changes": [(('model','use_boundary_gate'), False)],
        },
        "v11_no_boundary_logit_fusion": {
            "description": "Predict boundary logit but do not concatenate it into the fusion module.",
            "changes": [(('model','fuse_boundary_logit'), False)],
        },
        "v11_no_liteaspp_context": {
            "description": "Replace LiteASPP context aggregation with DSConv context module.",
            "changes": [(('model','context_module'), 'dsconv')],
        },
        "v11_no_aux_loss": {
            "description": "Disable auxiliary segmentation head/loss.",
            "changes": [(('model','use_aux'), False), (('train','loss','aux_weight'), 0.0)],
        },
        "v11_slim_backbone_capacity": {
            "description": "Backbone-capacity ablation proxy: reduce base/context channels instead of claiming literal backbone removal.",
            "changes": [(('model','base_channels'), 16), (('model','context_channels'), 64)],
        },
    }

    manifest_rows = []
    for name, spec in variants.items():
        cfg = copy.deepcopy(base)
        cfg["save_dir"] = f"seg/runs/v11_ablation/{name}"
        cfg.setdefault("v11_ablation", {})["name"] = name
        cfg["v11_ablation"]["description"] = spec["description"]
        cfg["v11_ablation"]["base_config"] = str(base_path)
        if args.epochs is not None:
            cfg.setdefault("train", {})["epochs"] = int(args.epochs)
            cfg["v11_ablation"]["epochs_override"] = int(args.epochs)
        for path, value in spec["changes"]:
            deep_set(cfg, tuple(path), value)
        out_path = out_dir / f"{name}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        manifest_rows.append((name, out_path, spec["description"]))

    manifest = out_dir / "V11_ABLATION_CONFIGS_MANIFEST.tsv"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("variant\tconfig_path\tdescription\n")
        for name, path, desc in manifest_rows:
            f.write(f"{name}\t{path.as_posix()}\t{desc}\n")

    print(f"[OK] wrote {len(manifest_rows)} V11 ablation configs to {out_dir}")
    print(f"[OK] manifest: {manifest}")
    print("[NOTE] For a full study, train all generated configs and then run v11_summarize_ablation.py.")


if __name__ == "__main__":
    main()
