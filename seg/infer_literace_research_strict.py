"""Strict LiteRaceSegNet inference for improvement experiments.

No mock mask. No fallback mask. No result replacement.
The script saves raw argmax, probability-threshold mask, and postprocessed mask
separately so that improvement experiments stay auditable.
"""
import argparse, csv, json, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch

SEG_DIR = Path(__file__).resolve().parent
if str(SEG_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_DIR))

from core.model_select import get_model
from core.save import load_state
from core.train_utils import load_yaml, make_dir, get_device
from infer_seg import collect_images, prep
from infer_service_visual import process_one

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def args_parser():
    p = argparse.ArgumentParser(description='LiteRaceSegNet strict real-model inference for upgrade experiments')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--image')
    g.add_argument('--input_dir')
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--threshold', type=float, default=0.60, help='damage class softmax threshold')
    p.add_argument('--min_area_pixels', type=int, default=120, help='remove tiny components after thresholding')
    p.add_argument('--save_prob', action='store_true')
    p.add_argument('--no_service_card', action='store_true')
    p.add_argument('--no_boundary', action='store_true')
    return p.parse_args()

def remove_small_components(mask, min_area):
    mask = (mask > 0).astype(np.uint8)
    raw_pixels = int(mask.sum())
    if min_area <= 1 or raw_pixels == 0:
        return mask.astype(bool), raw_pixels, 0, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    removed = 0
    kept_components = 0
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == lab] = 1
            kept_components += 1
        else:
            removed += area
    return keep.astype(bool), raw_pixels, removed, kept_components

def overlay_bgr(resized_rgb, mask, color_rgb, alpha):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask > 0] = np.asarray(color_rgb, dtype=np.uint8)
    return cv2.addWeighted(cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR), 1.0-alpha, cv2.cvtColor(color, cv2.COLOR_RGB2BGR), alpha, 0)

def component_count(mask):
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return 0
    n, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return max(0, int(n)-1)

def flag(ratio_percent, comps):
    if ratio_percent >= 90: return 'FAIL_NEAR_FULL_MASK'
    if ratio_percent >= 30: return 'REVIEW_OVERDETECTION_HIGH'
    if ratio_percent >= 15: return 'REVIEW_OVERDETECTION_MEDIUM'
    if ratio_percent <= 0.01: return 'REVIEW_EMPTY_OR_TOO_SMALL'
    if comps >= 10: return 'REVIEW_FRAGMENTED'
    return 'PASS_VISUAL_SANITY'

@torch.no_grad()
def main():
    args = args_parser()
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError('--threshold must be 0~1')
    cfg = load_yaml(args.config)
    device = get_device(cfg)
    out = Path(args.outdir)
    dirs = {
        'raw': out/'01_raw_argmax_masks',
        'prob': out/'02_damage_probability_u8',
        'thr': out/'03_threshold_masks',
        'post': out/'04_postprocessed_masks',
        'raw_overlay': out/'05_raw_argmax_overlays',
        'post_overlay': out/'06_postprocessed_overlays',
        'service': out/'07_service_visual_from_postprocessed',
    }
    for k,d in dirs.items():
        if k == 'prob' and not args.save_prob: continue
        make_dir(str(d))

    net = get_model(cfg).to(device)
    load_state(args.ckpt, net, map_location=device.type)
    net.eval()
    image_size = cfg['train']['image_size']
    infer_cfg = cfg.get('infer', {})
    alpha = float(infer_cfg.get('overlay_alpha', 0.45))
    palette = infer_cfg.get('palette', [[30,30,30],[255,90,0]])
    fg = palette[1] if len(palette) > 1 else [255,90,0]

    if args.image:
        images = [args.image]
    else:
        images = collect_images(args.input_dir)
        if args.recursive:
            images = [str(p) for p in sorted(Path(args.input_dir).rglob('*')) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not images:
        raise FileNotFoundError(args.image or args.input_dir)

    rows = []
    for i, path in enumerate(images, 1):
        org = cv2.imread(path, cv2.IMREAD_COLOR)
        if org is None:
            print('[SKIP]', path); continue
        x, resized_rgb = prep(org, image_size)
        logits = net(x.to(device))['out']
        prob = torch.softmax(logits, dim=1)[0,1].cpu().numpy()
        raw = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8) > 0
        thr = prob >= args.threshold
        post, thr_pixels, removed, kept_comps = remove_small_components(thr, args.min_area_pixels)
        base = Path(path).stem
        raw_path = dirs['raw']/f'{base}_raw_argmax.png'
        thr_path = dirs['thr']/f'{base}_thr{args.threshold:.2f}.png'
        post_path = dirs['post']/f'{base}_post_thr{args.threshold:.2f}_min{args.min_area_pixels}.png'
        cv2.imwrite(str(raw_path), (raw.astype(np.uint8)*255))
        cv2.imwrite(str(thr_path), (thr.astype(np.uint8)*255))
        cv2.imwrite(str(post_path), (post.astype(np.uint8)*255))
        if args.save_prob:
            cv2.imwrite(str(dirs['prob']/f'{base}_damage_prob_u8.png'), np.clip(prob*255,0,255).astype(np.uint8))
        cv2.imwrite(str(dirs['raw_overlay']/f'{base}_raw_overlay.png'), overlay_bgr(resized_rgb, raw, fg, alpha))
        cv2.imwrite(str(dirs['post_overlay']/f'{base}_post_overlay.png'), overlay_bgr(resized_rgb, post, fg, alpha))
        process_one(Path(path), dirs['service'], mask_path=post_path, mock=False, make_card=not args.no_service_card, make_boundary=not args.no_boundary, fallback_to_mock_if_bad_mask=False, min_area_pixels=0, mask_source_label='model_prediction_postprocessed_strict')
        raw_ratio = float(raw.mean()*100.0)
        thr_ratio = float(thr.mean()*100.0)
        post_ratio = float(post.mean()*100.0)
        comps = component_count(post)
        rows.append({
            'image': path,
            'threshold': args.threshold,
            'min_area_pixels': args.min_area_pixels,
            'raw_argmax_ratio_percent': round(raw_ratio,4),
            'threshold_ratio_percent': round(thr_ratio,4),
            'postprocessed_ratio_percent': round(post_ratio,4),
            'removed_pixels_after_threshold': removed,
            'component_count_post': comps,
            'quality_flag': flag(post_ratio, comps),
            'raw_mask': str(raw_path),
            'threshold_mask': str(thr_path),
            'postprocessed_mask': str(post_path),
        })
        print(f'[{i}/{len(images)}] {base}: raw={raw_ratio:.2f}% thr={thr_ratio:.2f}% post={post_ratio:.2f}% {rows[-1]["quality_flag"]}')

    with open(out/'strict_inference_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(out/'strict_inference_run_meta.json', 'w', encoding='utf-8') as f:
        json.dump({'config': args.config, 'checkpoint': args.ckpt, 'threshold': args.threshold, 'min_area_pixels': args.min_area_pixels, 'note': 'No mock/fallback mask was used. Raw/threshold/postprocessed outputs are saved separately.'}, f, ensure_ascii=False, indent=2)
    print('[OK] strict inference outputs:', out)

if __name__ == '__main__':
    main()
