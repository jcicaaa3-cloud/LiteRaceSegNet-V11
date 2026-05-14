import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from core.model_select import get_model
from core.save import load_state
from core.train_utils import load_yaml, make_dir, get_device
from infer_seg import collect_images, prep
from infer_service_visual import process_one


def get_args():
    p = argparse.ArgumentParser(description="Run LiteRaceSegNet checkpoint on road images and build service visualizations from REAL model predictions.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Single input image path")
    group.add_argument("--input_dir", type=str, help="Input image directory")
    p.add_argument("--config", type=str, required=True, help="Training/inference config yaml")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path, e.g. best.pth")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--recursive", action="store_true", help="Search input_dir recursively")
    p.add_argument("--min_area_pixels", type=int, default=80, help="Remove tiny predicted components before service stats")
    p.add_argument("--fallback_to_mock_if_bad_mask", action="store_true", help="If prediction is nearly empty/full, fall back to conservative mock mask for presentation")
    p.add_argument("--no_card", action="store_true", help="Skip service card PNG")
    p.add_argument("--no_boundary", action="store_true", help="Skip boundary PNG")
    return p.parse_args()


def save_binary_mask(mask_arr: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), (mask_arr.astype(np.uint8) * 255))


@torch.no_grad()
def main():
    args = get_args()
    cfg = load_yaml(args.config)
    device = get_device(cfg)

    out_root = Path(args.outdir)
    pred_dir = out_root / "pred_masks"
    simple_dir = out_root / "simple_infer"
    service_dir = out_root / "service_vis_pred"
    make_dir(str(pred_dir))
    make_dir(str(simple_dir))
    make_dir(str(service_dir))

    net = get_model(cfg).to(device)
    load_state(args.ckpt, net, map_location=device.type)
    net.eval()

    image_size = cfg["train"]["image_size"]
    alpha = float(cfg.get("infer", {}).get("overlay_alpha", 0.45))
    palette = cfg.get("infer", {}).get("palette", [[30,30,30],[255,90,0]])
    color_fg = np.array(palette[1], dtype=np.uint8) if len(palette) > 1 else np.array([255,90,0], dtype=np.uint8)

    if args.image:
        image_list = [args.image]
    else:
        image_list = collect_images(args.input_dir)
        if args.recursive:
            image_list = [str(p) for p in sorted(Path(args.input_dir).rglob('*')) if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]
    if not image_list:
        raise FileNotFoundError(args.image or args.input_dir)

    for idx, path in enumerate(image_list, 1):
        org = cv2.imread(path, cv2.IMREAD_COLOR)
        if org is None:
            continue
        x, resized = prep(org, image_size)
        x = x.to(device)
        out = net(x)
        pred = torch.argmax(out['out'], dim=1)[0].cpu().numpy().astype(np.uint8)
        base = Path(path).stem

        pred_mask_path = pred_dir / f"{base}_pred.png"
        save_binary_mask(pred > 0, pred_mask_path)

        # Simple debug overlay from resized inference canvas
        mask_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        mask_color[pred > 0] = color_fg
        overlay = cv2.addWeighted(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), 1.0 - alpha, cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR), alpha, 0)
        cv2.imwrite(str(simple_dir / f"{base}_overlay.png"), overlay)
        cv2.imwrite(str(simple_dir / f"{base}_mask_color.png"), cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR))

        process_one(
            image_path=Path(path),
            out_dir=service_dir,
            mask_path=pred_mask_path,
            mock=False,
            make_card=not args.no_card,
            make_boundary=not args.no_boundary,
            fallback_to_mock_if_bad_mask=args.fallback_to_mock_if_bad_mask,
            min_area_pixels=args.min_area_pixels,
            mask_source_label="model_prediction",
        )
        print(f"[{idx}/{len(image_list)}] done: {path}")

    print(f"[OK] REAL model prediction pipeline done.\n- predicted masks: {pred_dir}\n- service visuals: {service_dir}\n- simple infer debug: {simple_dir}")


if __name__ == '__main__':
    main()
