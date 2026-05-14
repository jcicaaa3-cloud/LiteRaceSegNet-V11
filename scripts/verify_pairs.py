#!/usr/bin/env python3
import sys
from pathlib import Path
try:
    from PIL import Image
except Exception as e:
    Image = None
IMG_EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}

def files(d):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]) if d.exists() else []

def check(root, split):
    img_dir=root/split/'images'; mask_dir=root/split/'masks'
    images=files(img_dir); masks=files(mask_dir)
    mb={p.stem:p for p in masks}; ib={p.stem:p for p in images}
    missing_masks=[]; missing_images=[]; mismatches=[]; empty_masks=[]
    for img in images:
        m=mb.get(img.stem)
        if not m:
            missing_masks.append(img.name); continue
        if Image:
            with Image.open(img) as ii, Image.open(m) as mm:
                if ii.size != mm.size:
                    mismatches.append((img.name, ii.size, mm.size))
                try:
                    extrema = mm.convert('L').getextrema()
                    if extrema[1] == 0:
                        empty_masks.append(m.name)
                except Exception:
                    pass
    for m in masks:
        if m.stem not in ib: missing_images.append(m.name)
    print(f'{split} images={len(images)} masks={len(masks)} missing_masks={len(missing_masks)} missing_images={len(missing_images)} size_mismatch={len(mismatches)} empty_masks={len(empty_masks)}')
    if missing_masks[:8]: print('  missing_masks', missing_masks[:8])
    if missing_images[:8]: print('  missing_images', missing_images[:8])
    if mismatches[:8]: print('  mismatches', mismatches[:8])
    if empty_masks[:8]: print('  empty_masks', empty_masks[:8])
    return len(missing_masks)+len(missing_images)+len(mismatches)

def main():
    if len(sys.argv)!=2:
        print('usage: python scripts/verify_pairs.py datasets/pothole_binary/processed'); sys.exit(2)
    root=Path(sys.argv[1])
    bad=check(root,'train')+check(root,'val')
    print('[OK] all pairs valid' if bad==0 else '[FAIL] pair verification failed')
    sys.exit(1 if bad else 0)
if __name__=='__main__': main()
