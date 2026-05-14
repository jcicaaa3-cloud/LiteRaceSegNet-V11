#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect SegFormer-E3 evidence: masks, overlays, cards, metrics, and zip."""
import argparse, csv, json, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
IMG_EXTS=[".jpg",".jpeg",".png",".bmp",".webp"]
MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def find_pairs(root, split):
    root=Path(root); masks={p.stem:p for p in (root/split/'masks').iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}; return [(p,masks[p.stem]) for p in sorted((root/split/'images').iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem in masks]
def prep(img,h,w):
    r=img.convert('RGB').resize((w,h),Image.Resampling.BILINEAR); a=np.asarray(r).astype(np.float32)/255.; t=torch.from_numpy(a).permute(2,0,1); return (t-MEAN)/STD
def bmask(mask,h,w): return (np.asarray(mask.convert('L').resize((w,h),Image.Resampling.NEAREST))>0).astype(np.uint8)
def overlay(base,mask,color,alpha=.45):
    base=base.convert('RGB'); c=Image.new('RGB',base.size,color); m=Image.fromarray((mask>0).astype(np.uint8)*255,'L').resize(base.size,Image.Resampling.NEAREST); return Image.composite(Image.blend(base,c,alpha),base,m)
def metrics(pred,gt):
    p=pred.astype(bool); g=gt.astype(bool); tp=int((p&g).sum()); tn=int((~p&~g).sum()); fp=int((p&~g).sum()); fn=int((~p&g).sum())
    bg=tn/max(1,tn+fp+fn); di=tp/max(1,tp+fp+fn); return dict(tp=tp,tn=tn,fp=fp,fn=fn,mIoU=(bg+di)/2,background_IoU=bg,damage_IoU=di,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),dice=2*tp/max(1,2*tp+fp+fn))
def card(img,gtov,prov,title,row):
    w,h=img.size; pad=20; header=70; labelh=35; out=Image.new('RGB',(w*3+pad*4,h+header+labelh+pad),'white'); d=ImageDraw.Draw(out)
    d.text((pad,15),title,fill=(0,0,0)); d.text((pad,40),f"mIoU={row['mIoU']:.4f} damageIoU={row['damage_IoU']:.4f} precision={row['precision']:.4f} recall={row['recall']:.4f}",fill=(0,0,0))
    for x,label,im in zip([pad,pad*2+w,pad*3+w*2],['Image','GT overlay','SegFormer-E3 pred'],[img,gtov,prov]): d.text((x,header),label,fill=(0,0,0)); out.paste(im,(x,header+labelh))
    return out
def zip_dir(src,out_zip):
    out_zip=Path(out_zip); out_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(out_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for p in Path(src).rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(Path(src).parent))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='datasets/pothole_binary_v6_aug/processed'); ap.add_argument('--run-dir',default='seg/runs/segformer_e3_v6_baseline'); ap.add_argument('--checkpoint',default='best.pth'); ap.add_argument('--out-zip',default='LRS_SEGFORMER_E3_V6_EVIDENCE.zip'); args=ap.parse_args()
    run=Path(args.run_dir); cfg=json.loads((run/'segformer_e3_config.json').read_text(encoding='utf-8')); h=int(cfg.get('height',384)); w=int(cfg.get('width',512)); model_name=cfg.get('model_name','nvidia/mit-b0'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(run/args.checkpoint,map_location=device); model=SegformerForSemanticSegmentation.from_pretrained(model_name,num_labels=2,id2label={0:'background',1:'damage'},label2id={'background':0,'damage':1},ignore_mismatched_sizes=True); model.load_state_dict(ck['model_state']); model.to(device).eval()
    pred_dir=run/'segformer_e3_val_predictions'
    for sub in ['pred_mask','gt_overlay','pred_overlay','cards']: (pred_dir/sub).mkdir(parents=True,exist_ok=True)
    rows=[]; total=dict(tp=0,tn=0,fp=0,fn=0)
    with torch.no_grad():
        for imgp,maskp in find_pairs(args.data_root,'val'):
            img=Image.open(imgp).convert('RGB'); img_res=img.resize((w,h),Image.Resampling.BILINEAR); gt=bmask(Image.open(maskp),h,w); x=prep(img,h,w).unsqueeze(0).to(device); logits=F.interpolate(model(pixel_values=x).logits,size=(h,w),mode='bilinear',align_corners=False); pred=logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            row=metrics(pred,gt); row['file']=imgp.name; rows.append(row); [total.__setitem__(k,total[k]+row[k]) for k in total]
            Image.fromarray(pred*255,'L').save(pred_dir/'pred_mask'/(imgp.stem+'_segformer_e3_pred_mask.png')); gtov=overlay(img_res,gt,(0,180,0)); prov=overlay(img_res,pred,(255,70,0)); gtov.save(pred_dir/'gt_overlay'/(imgp.stem+'_gt_overlay.png')); prov.save(pred_dir/'pred_overlay'/(imgp.stem+'_segformer_e3_pred_overlay.png')); card(img_res,gtov,prov,imgp.name,row).save(pred_dir/'cards'/(imgp.stem+'_segformer_e3_card.jpg'),quality=92)
    tp,tn,fp,fn=total['tp'],total['tn'],total['fp'],total['fn']; bg=tn/max(1,tn+fp+fn); di=tp/max(1,tp+fp+fn); summary=dict(total,mIoU=(bg+di)/2,background_IoU=bg,damage_IoU=di,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),dice=2*tp/max(1,2*tp+fp+fn),count=len(rows),checkpoint=str(run/args.checkpoint))
    field=['file','mIoU','background_IoU','damage_IoU','precision','recall','dice','tp','tn','fp','fn']
    with (run/'segformer_e3_val_per_image_metrics.csv').open('w',newline='',encoding='utf-8') as f: wr=csv.DictWriter(f,fieldnames=field); wr.writeheader(); [wr.writerow({k:r.get(k,'') for k in field}) for r in rows]
    (run/'segformer_e3_val_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); zip_dir(run,args.out_zip); print('[DONE][SegFormer-E3 evidence]',args.out_zip); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
