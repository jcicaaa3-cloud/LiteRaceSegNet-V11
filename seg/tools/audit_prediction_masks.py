import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np

EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

def mask_stats(path):
    arr=cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None: raise ValueError(f'unreadable: {path}')
    m=arr>0
    ratio=float(m.mean()*100.0)
    n=0
    if int(m.sum())>0:
        n,_,_,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
        n=max(0,int(n)-1)
    if ratio>=90: flag='FAIL_NEAR_FULL_MASK'
    elif ratio>=30: flag='REVIEW_OVERDETECTION_HIGH'
    elif ratio>=15: flag='REVIEW_OVERDETECTION_MEDIUM'
    elif ratio<=0.01: flag='REVIEW_EMPTY_OR_TOO_SMALL'
    elif n>=10: flag='REVIEW_FRAGMENTED'
    else: flag='PASS_VISUAL_SANITY'
    return ratio,n,flag

def main():
    ap=argparse.ArgumentParser(description='Audit prediction masks by damage area ratio and component count')
    ap.add_argument('--mask_dir', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--recursive', action='store_true')
    args=ap.parse_args()
    root=Path(args.mask_dir)
    pattern='**/*' if args.recursive else '*'
    files=[p for p in sorted(root.glob(pattern)) if p.is_file() and p.suffix.lower() in EXTS]
    rows=[]
    for p in files:
        ratio, comps, flag=mask_stats(p)
        rows.append({'mask':str(p),'damage_ratio_percent':round(ratio,4),'component_count':comps,'quality_flag':flag})
    Path(args.out_csv).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out_csv,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['mask','damage_ratio_percent','component_count','quality_flag'])
        w.writeheader(); w.writerows(rows)
    print(f'[OK] audited {len(rows)} masks -> {args.out_csv}')

if __name__=='__main__': main()
