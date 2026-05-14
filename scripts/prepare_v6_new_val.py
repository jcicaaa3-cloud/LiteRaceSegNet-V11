#!/usr/bin/env python3
"""
V6 dataset preparation.
- Keeps LiteRaceSegNet objective unchanged: accurate road-damage segmentation.
- Moves the CURRENT val set into train with an oldval_ prefix to avoid filename collision.
- Replaces val with a newly provided held-out validation ZIP.
- Writes a backup and manifest before touching the dataset.
"""
import argparse, json, shutil, sys, tempfile, zipfile
from datetime import datetime
from pathlib import Path
from PIL import Image

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp'}
MASK_EXTS = {'.png','.jpg','.jpeg','.bmp','.webp'}

def img_files(d):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]) if d.exists() else []

def mask_files(d):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTS]) if d.exists() else []

def find_new_val_root(tmp: Path) -> Path:
    candidates = []
    for base in [tmp, *[p for p in tmp.rglob('*') if p.is_dir()]]:
        if (base/'images').is_dir() and (base/'masks').is_dir():
            candidates.append(base)
    if not candidates:
        raise SystemExit('[ERROR] new val ZIP must contain images/ and masks/ folders. Examples: new_val/images + new_val/masks, or images + masks.')
    # Prefer explicit new_val folder, otherwise shortest path.
    candidates.sort(key=lambda p: (0 if p.name.lower()=='new_val' else 1, len(str(p))))
    return candidates[0]

def pair_map(images, masks):
    mb = {p.stem:p for p in masks}
    pairs=[]; missing=[]
    for img in images:
        m = mb.get(img.stem)
        if not m: missing.append(img.name)
        else: pairs.append((img,m))
    extra = [m.name for m in masks if m.stem not in {i.stem for i in images}]
    return pairs, missing, extra

def validate_pairs(pairs, expected, allow_empty=False):
    bad=[]; empty=[]
    for img, mask in pairs:
        with Image.open(img) as ii, Image.open(mask) as mm:
            if ii.size != mm.size:
                bad.append((img.name, ii.size, mm.size))
            mx = mm.convert('L').getextrema()[1]
            if mx == 0:
                empty.append(mask.name)
    if expected and len(pairs) != expected:
        raise SystemExit(f'[ERROR] expected {expected} new val pairs, got {len(pairs)}. Stop to prevent wrong validation split.')
    if bad:
        raise SystemExit('[ERROR] image/mask size mismatch in new val: ' + repr(bad[:8]))
    if empty and not allow_empty:
        raise SystemExit('[ERROR] empty masks detected in new val: ' + repr(empty[:8]) + ' / use --allow-empty-masks only if this is intentional.')

def copy_pair(src_img, src_mask, dst_img_dir, dst_mask_dir, stem_prefix=''):
    dst_img_dir.mkdir(parents=True, exist_ok=True); dst_mask_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_prefix + src_img.stem
    dst_img = dst_img_dir / (stem + src_img.suffix.lower())
    dst_mask = dst_mask_dir / (stem + '.png')
    if dst_img.exists() or dst_mask.exists():
        raise SystemExit(f'[ERROR] destination collision: {dst_img.name} or {dst_mask.name}. Dataset may already be V6-prepared. Restore backup or use a clean processed folder.')
    shutil.copy2(src_img, dst_img)
    # Normalize mask to binary PNG to avoid JPEG mask artifacts.
    with Image.open(src_mask) as mm:
        m = mm.convert('L')
        # binarize >0 to 255
        m = m.point(lambda x: 255 if x > 0 else 0)
        m.save(dst_mask)
    return dst_img.name, dst_mask.name

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--new-val-zip', required=True)
    ap.add_argument('--dataset-root', default='datasets/pothole_binary/processed')
    ap.add_argument('--backup-dir', default=None)
    ap.add_argument('--expected-val-count', type=int, default=10)
    ap.add_argument('--allow-empty-masks', action='store_true')
    a=ap.parse_args()

    root=Path(a.dataset_root)
    if not (root/'train/images').is_dir() or not (root/'train/masks').is_dir() or not (root/'val/images').is_dir() or not (root/'val/masks').is_dir():
        raise SystemExit(f'[ERROR] dataset root not ready: {root}')
    z=Path(a.new_val_zip).expanduser()
    if not z.exists():
        raise SystemExit(f'[ERROR] new val ZIP not found: {z}')

    train_imgs=img_files(root/'train/images'); train_masks=mask_files(root/'train/masks')
    old_val_imgs=img_files(root/'val/images'); old_val_masks=mask_files(root/'val/masks')
    old_pairs, old_missing, old_extra = pair_map(old_val_imgs, old_val_masks)
    if old_missing or old_extra:
        raise SystemExit(f'[ERROR] current val pair mismatch before V6. missing={old_missing[:5]} extra={old_extra[:5]}')
    if any(p.stem.startswith('oldval_') for p in train_imgs):
        raise SystemExit('[ERROR] oldval_ files already exist in train. This looks already V6-prepared. Stop to prevent duplicate old-val merge.')

    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)
        with zipfile.ZipFile(z) as zz:
            zz.extractall(tmp)
        nvroot=find_new_val_root(tmp)
        new_imgs=img_files(nvroot/'images'); new_masks=mask_files(nvroot/'masks')
        new_pairs, missing, extra = pair_map(new_imgs, new_masks)
        if missing or extra:
            raise SystemExit(f'[ERROR] new val pair mismatch. missing_masks_for={missing[:8]} extra_masks={extra[:8]}')
        validate_pairs(new_pairs, a.expected_val_count, allow_empty=a.allow_empty_masks)

        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        backup=Path(a.backup_dir) if a.backup_dir else Path('datasets')/f'_v6_backup_{ts}'/'pothole_binary_processed'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise SystemExit(f'[ERROR] backup dir already exists: {backup}')
        shutil.copytree(root, backup)
        print(f'[BACKUP] {root} -> {backup}')

        moved=[]
        for img,mask in old_pairs:
            moved.append(copy_pair(img, mask, root/'train/images', root/'train/masks', stem_prefix='oldval_'))
        shutil.rmtree(root/'val/images'); shutil.rmtree(root/'val/masks')
        (root/'val/images').mkdir(parents=True, exist_ok=True); (root/'val/masks').mkdir(parents=True, exist_ok=True)
        added=[]
        for img,mask in new_pairs:
            added.append(copy_pair(img, mask, root/'val/images', root/'val/masks', stem_prefix=''))

        manifest={
            'created_at': ts,
            'dataset_root': str(root),
            'backup': str(backup),
            'before': {'train_images': len(train_imgs), 'train_masks': len(train_masks), 'val_images': len(old_val_imgs), 'val_masks': len(old_val_masks)},
            'old_val_added_to_train_with_prefix': 'oldval_',
            'old_val_added_pairs': moved,
            'new_val_zip': str(z),
            'new_val_detected_root': str(nvroot),
            'new_val_pairs': added,
            'expected_val_count': a.expected_val_count
        }
        out=root/'V6_DATASET_MANIFEST.json'
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[DONE] V6 dataset prepared')
        print(f'[MANIFEST] {out}')
        print(f'[EXPECTED] train should increase by {len(old_pairs)}; val should be {len(new_pairs)}')

if __name__=='__main__': main()
