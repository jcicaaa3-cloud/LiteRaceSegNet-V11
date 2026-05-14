import argparse, csv
from pathlib import Path
import cv2
import numpy as np

EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

def stem_key(p):
    s=p.stem
    for suf in ['_pred','_raw_argmax','_mask','_label','_gt','_thr0.60','_post_thr0.60_min120']:
        if s.endswith(suf): s=s[:-len(suf)]
    return s

def index_files(d):
    return {stem_key(p):p for p in Path(d).glob('*') if p.is_file() and p.suffix.lower() in EXTS}

def read_bin(p, shape=None):
    a=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    if a is None: raise ValueError(f'unreadable: {p}')
    if shape and a.shape[:2] != shape:
        a=cv2.resize(a,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
    return a>0

def metrics(pred, gt):
    pred=pred.astype(bool); gt=gt.astype(bool)
    tp=np.logical_and(pred,gt).sum(); fp=np.logical_and(pred,~gt).sum(); fn=np.logical_and(~pred,gt).sum(); tn=np.logical_and(~pred,~gt).sum()
    eps=1e-9
    iou=tp/(tp+fp+fn+eps); dice=2*tp/(2*tp+fp+fn+eps); prec=tp/(tp+fp+eps); rec=tp/(tp+fn+eps)
    acc=(tp+tn)/(tp+tn+fp+fn+eps)
    return dict(iou_damage=iou,dice_damage=dice,precision=prec,recall=rec,pixel_acc=acc,tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))

def main():
    ap=argparse.ArgumentParser(description='Evaluate binary prediction masks against GT masks')
    ap.add_argument('--pred_dir',required=True); ap.add_argument('--gt_dir',required=True); ap.add_argument('--out_csv',required=True)
    args=ap.parse_args()
    preds=index_files(args.pred_dir); gts=index_files(args.gt_dir)
    keys=sorted(set(preds)&set(gts))
    if not keys: raise FileNotFoundError('No matching prediction/GT pairs by stem')
    rows=[]
    sums={'tp':0,'fp':0,'fn':0,'tn':0}
    for k in keys:
        gt=read_bin(gts[k]); pred=read_bin(preds[k], gt.shape)
        m=metrics(pred,gt); sums.update({x:sums[x]+m[x] for x in sums})
        row={'image_key':k,'pred':str(preds[k]),'gt':str(gts[k])}; row.update({kk:round(v,6) if isinstance(v,float) else v for kk,v in m.items()}); rows.append(row)
    agg=metrics(np.array([1]*sums['tp']+[1]*sums['fp']+[0]*sums['fn']+[0]*sums['tn'],dtype=bool), np.array([1]*sums['tp']+[0]*sums['fp']+[1]*sums['fn']+[0]*sums['tn'],dtype=bool))
    Path(args.out_csv).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out_csv,'w',newline='',encoding='utf-8') as f:
        fields=list(rows[0].keys()); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('[OK]',len(rows),'pairs ->',args.out_csv)
    print('aggregate:', {k:round(v,6) if isinstance(v,float) else v for k,v in agg.items()})

if __name__=='__main__': main()
