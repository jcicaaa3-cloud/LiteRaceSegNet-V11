#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SegFormer-E3 baseline trainer for LiteRaceSegNet V6 experiments."""
import argparse, csv, json, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
try:
    from transformers import SegformerForSemanticSegmentation
except Exception as e:
    raise SystemExit("[ERROR] transformers is not installed. Run: pip install transformers safetensors\n" + str(e))
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=True

def find_pairs(root:Path, split:str)->List[Tuple[Path,Path]]:
    img_dir=root/split/'images'; mask_dir=root/split/'masks'
    if not img_dir.exists() or not mask_dir.exists(): raise FileNotFoundError(f'Missing {split}/images or {split}/masks under {root}')
    masks={p.stem:p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    pairs=[]; missing=[]
    for img in sorted(img_dir.iterdir()):
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            if img.stem in masks: pairs.append((img,masks[img.stem]))
            else: missing.append(img.name)
    if missing: raise RuntimeError(f'Missing masks for {len(missing)} images. Sample: {missing[:5]}')
    return pairs

class SegPairDataset(Dataset):
    def __init__(self, root, split, height, width):
        self.root=Path(root); self.split=split; self.height=height; self.width=width; self.pairs=find_pairs(self.root, split)
        if not self.pairs: raise RuntimeError(f'No pairs found split={split} under {self.root}')
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img=Image.open(img_path).convert('RGB').resize((self.width,self.height), Image.Resampling.BILINEAR)
        mask=Image.open(mask_path).convert('L').resize((self.width,self.height), Image.Resampling.NEAREST)
        arr=np.asarray(img).astype(np.float32)/255.0
        x=torch.from_numpy(arr).permute(2,0,1); x=(x-MEAN)/STD
        y=torch.from_numpy((np.asarray(mask)>0).astype(np.int64))
        return x,y,img_path.name

def confusion(pred, target):
    p=pred.detach().cpu().numpy().astype(np.uint8); t=target.detach().cpu().numpy().astype(np.uint8)
    return dict(tp=int(((p==1)&(t==1)).sum()), tn=int(((p==0)&(t==0)).sum()), fp=int(((p==1)&(t==0)).sum()), fn=int(((p==0)&(t==1)).sum()))

def met(tp,tn,fp,fn):
    bg=tn/max(1,tn+fp+fn); di=tp/max(1,tp+fp+fn); mi=(bg+di)/2; pr=tp/max(1,tp+fp); re=tp/max(1,tp+fn); dice=2*tp/max(1,2*tp+fp+fn); acc=(tp+tn)/max(1,tp+tn+fp+fn)
    return dict(mIoU=mi, background_IoU=bg, damage_IoU=di, precision=pr, recall=re, dice=dice, pixel_acc=acc)

@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval(); total_loss=0.; total_n=0; tot=dict(tp=0,tn=0,fp=0,fn=0)
    for x,y,_ in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(amp and device.type=='cuda')):
            out=model(pixel_values=x, labels=y); logits=F.interpolate(out.logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
        pred=logits.argmax(1); c=confusion(pred,y)
        for k in tot: tot[k]+=c[k]
        total_loss += float(out.loss.item())*x.size(0); total_n += x.size(0)
    m=met(**tot); m['val_loss']=total_loss/max(1,total_n); return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', default='datasets/pothole_binary_v6_aug/processed'); ap.add_argument('--save-dir', default='seg/runs/segformer_e3_v6_baseline')
    ap.add_argument('--model-name', default='nvidia/mit-b0'); ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--height', type=int, default=384); ap.add_argument('--width', type=int, default=512); ap.add_argument('--lr', type=float, default=6e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01); ap.add_argument('--patience', type=int, default=12); ap.add_argument('--num-workers', type=int, default=4); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--no-amp', action='store_true')
    args=ap.parse_args(); seed_all(args.seed); save_dir=Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); amp=not args.no_amp
    train_ds=SegPairDataset(args.data_root,'train',args.height,args.width); val_ds=SegPairDataset(args.data_root,'val',args.height,args.width)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=(device.type=='cuda'))
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=(device.type=='cuda'))
    print(f'[SegFormer-E3] device={device} model={args.model_name} data_root={args.data_root}')
    print(f'[SegFormer-E3] train_pairs={len(train_ds)} val_pairs={len(val_ds)} image_size={args.height}x{args.width}')
    model=SegformerForSemanticSegmentation.from_pretrained(args.model_name,num_labels=2,id2label={0:'background',1:'damage'},label2id={'background':0,'damage':1},ignore_mismatched_sizes=True).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.PolynomialLR(opt,total_iters=max(1,args.epochs*len(train_loader)),power=0.9)
    scaler=torch.cuda.amp.GradScaler(enabled=(amp and device.type=='cuda'))
    cfg=vars(args).copy(); cfg.update(device=str(device), train_pairs=len(train_ds), val_pairs=len(val_ds)); (save_dir/'segformer_e3_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    fields=['epoch','train_loss','val_loss','mIoU','background_IoU','damage_IoU','precision','recall','dice','pixel_acc','lr','best_mIoU']
    with (save_dir/'train_log.csv').open('w',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    best=-1.; best_epoch=-1; no_imp=0
    for epoch in range(1,args.epochs+1):
        model.train(); run=0.; seen=0
        pbar=tqdm(train_loader,desc=f'[SegFormer-E3] epoch {epoch:03d}/{args.epochs:03d}',dynamic_ncols=True)
        for x,y,_ in pbar:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type=='cuda')): loss=model(pixel_values=x, labels=y).loss
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); sched.step()
            run += float(loss.item())*x.size(0); seen += x.size(0); pbar.set_postfix(loss=f'{run/max(1,seen):.4f}', lr=f'{opt.param_groups[0]["lr"]:.2e}')
        train_loss=run/max(1,seen); val=evaluate(model,val_loader,device,amp=amp); improved=val['mIoU']>best
        if improved:
            best=val['mIoU']; best_epoch=epoch; no_imp=0; torch.save(dict(epoch=epoch,model_state=model.state_dict(),optimizer_state=opt.state_dict(),metrics=val,config=cfg), save_dir/'best.pth')
        else: no_imp += 1
        torch.save(dict(epoch=epoch,model_state=model.state_dict(),optimizer_state=opt.state_dict(),metrics=val,config=cfg), save_dir/'last.pth')
        row=dict(epoch=epoch,train_loss=train_loss,val_loss=val['val_loss'],mIoU=val['mIoU'],background_IoU=val['background_IoU'],damage_IoU=val['damage_IoU'],precision=val['precision'],recall=val['recall'],dice=val['dice'],pixel_acc=val['pixel_acc'],lr=opt.param_groups[0]['lr'],best_mIoU=best)
        with (save_dir/'train_log.csv').open('a',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=fields).writerow(row)
        print(f"[VAL][SegFormer-E3] epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val['val_loss']:.4f} mIoU={val['mIoU']:.4f} damageIoU={val['damage_IoU']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} dice={val['dice']:.4f} best={best:.4f}@{best_epoch}")
        if no_imp>=args.patience: print(f'[EARLY STOP][SegFormer-E3] No mIoU improvement for {no_imp} epoch(s).'); break
    (save_dir/'summary.json').write_text(json.dumps(dict(best_mIoU=best,best_epoch=best_epoch,save_dir=str(save_dir)),indent=2),encoding='utf-8'); print('[DONE][SegFormer-E3]', best, best_epoch)
if __name__=='__main__': main()
