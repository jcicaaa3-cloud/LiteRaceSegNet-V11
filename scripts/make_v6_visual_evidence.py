#!/usr/bin/env python3
"""
LiteRaceSegNet V6 visual evidence hotfix.
Creates validation prediction masks, overlays, service cards, per-image IoU CSV, and a contact sheet.
Designed to run inside /home/ubuntu/road-damage-segmentation-portfolio.
"""
from __future__ import annotations

import argparse, csv, importlib, inspect, json, os, sys, math, traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:
    yaml = None

IMG_EXTS = {'.png','.jpg','.jpeg','.bmp','.webp'}

# ----------------------------- fallback model -----------------------------
class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1, groups=1, bias=False):
        if p is None:
            p = ((k - 1) // 2) * d
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=groups, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class SepBlock(nn.Module):
    # depthwise -> pointwise, used by stem.1 and fuse.1 in the current checkpoint naming
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3, stride, 1, groups=in_ch),
            ConvBNAct(in_ch, out_ch, 1, 1, 0),
        )
        self.use_res = stride == 1 and in_ch == out_ch
    def forward(self, x):
        y = self.net(x)
        return y + x if self.use_res else y

class DSBlock(nn.Module):
    # pointwise expand -> depthwise -> pointwise project -> BN, used by detail/down blocks
    def __init__(self, in_ch, out_ch, hidden=None, stride=1):
        super().__init__()
        hidden = hidden or out_ch
        self.net = nn.Sequential(
            ConvBNAct(in_ch, hidden, 1, 1, 0),
            ConvBNAct(hidden, hidden, 3, stride, 1, groups=hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_res = stride == 1 and in_ch == out_ch
    def forward(self, x):
        y = self.net(x)
        if self.use_res:
            y = y + x
        return F.relu(y, inplace=True)

class ASPPBranch(nn.Module):
    def __init__(self, ch, out_ch, rate):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(ch, ch, 3, 1, p=rate, d=rate, groups=ch),
            ConvBNAct(ch, out_ch, 1, 1, 0),
        )
    def forward(self, x):
        return self.net(x)

class LiteASPP(nn.Module):
    def __init__(self, ch=96, out=96, rates=(1,2,4)):
        super().__init__()
        self.branches = nn.ModuleList([ASPPBranch(ch, 24, r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, 24, 1, bias=True), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(96, out, 1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True))
    def forward(self, x):
        outs = [b(x) for b in self.branches]
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode='bilinear', align_corners=False)
        outs.append(pooled)
        return self.project(torch.cat(outs, dim=1))

class EdgeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(32, 32, 3, 1, 1, groups=32),
            ConvBNAct(32, 16, 1, 1, 0),
        )
    def forward(self, x):
        return self.net(x)

class BoundaryGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail_proj = ConvBNAct(24, 16, 1, 1, 0)
        self.context_proj = ConvBNAct(96, 16, 1, 1, 0)
        self.edge = nn.Sequential(EdgeBlock(), nn.Conv2d(16, 1, 1, bias=True))
    def forward(self, detail, ctx):
        edge = self.edge(torch.cat([self.detail_proj(detail), self.context_proj(ctx)], dim=1))
        return torch.sigmoid(edge)

class FallbackLiteRaceSegNet(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(3, 24, 3, 1, 1), SepBlock(24, 24, 1))
        self.detail = nn.Sequential(DSBlock(24, 24, 48, 1), DSBlock(24, 24, 48, 1))
        self.down1 = nn.Sequential(DSBlock(24, 48, 48, 2), DSBlock(48, 48, 96, 1))
        self.down2 = nn.Sequential(DSBlock(48, 96, 96, 2), DSBlock(96, 96, 192, 1))
        self.context = LiteASPP(96, 96, rates=(1,2,4))
        self.gate = BoundaryGate()
        self.fuse = nn.Sequential(ConvBNAct(121, 96, 1, 1, 0), SepBlock(96, 96, 1))
        self.head = nn.Conv2d(96, num_classes, 1)
        self.aux_head = nn.Conv2d(48, num_classes, 1)
    def forward(self, x):
        size = x.shape[-2:]
        s = self.stem(x)
        detail = self.detail(s)
        d1 = self.down1(s)
        d2 = self.down2(d1)
        ctx = self.context(d2)
        ctx_up = F.interpolate(ctx, size=detail.shape[-2:], mode='bilinear', align_corners=False)
        edge = self.gate(detail, ctx_up)
        fused = self.fuse(torch.cat([ctx_up, detail, edge], dim=1))
        out = self.head(fused)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        aux = self.aux_head(d1)
        aux = F.interpolate(aux, size=size, mode='bilinear', align_corners=False)
        return out, aux

# ----------------------------- helpers -----------------------------
def load_ckpt(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(str(path), map_location='cpu')
    if not isinstance(ckpt, dict):
        raise RuntimeError(f'Unsupported checkpoint type: {type(ckpt)}')
    return ckpt

def get_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for k in ('model','model_state','state_dict','net'):
        if isinstance(ckpt.get(k), dict):
            return ckpt[k]
    # maybe checkpoint itself is a state dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore
    raise RuntimeError(f'Cannot locate model state dict. keys={list(ckpt.keys())}')

def load_yaml_config(config_path: Optional[Path], ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if config_path and config_path.exists() and yaml is not None:
        with config_path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    if isinstance(ckpt.get('config'), dict):
        return ckpt['config']
    return {}

def try_build_project_model(cfg: Dict[str, Any]) -> Optional[nn.Module]:
    sys.path.insert(0, str(Path.cwd()))
    candidates = [
        'seg.train_literace', 'seg.models.literace', 'seg.models.literace_segnet',
        'seg.models.literacesegnet', 'models.literace', 'models.literace_segnet',
    ]
    model_cfg = cfg.get('model', {}) if isinstance(cfg.get('model'), dict) else {}
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        # builder functions first
        for fn_name in ('build_model','create_model','get_model','make_model'):
            fn = getattr(mod, fn_name, None)
            if not callable(fn):
                continue
            for arg in (cfg, model_cfg, None):
                try:
                    if arg is None:
                        m = fn()
                    else:
                        m = fn(arg)
                    if isinstance(m, nn.Module):
                        print(f'[MODEL] built by {modname}.{fn_name}')
                        return m
                except Exception:
                    pass
        # class names
        for cls_name in ('LiteRaceSegNet','LiteRace','LiteRaceNet','LiteRaceSeg','RoadDamageLiteRaceSegNet'):
            cls = getattr(mod, cls_name, None)
            if inspect.isclass(cls) and issubclass(cls, nn.Module):
                attempts = []
                kw = {k:v for k,v in model_cfg.items() if k not in ('name',)}
                attempts.append(kw)
                attempts.append({'num_classes': model_cfg.get('num_classes', 2)})
                attempts.append({})
                for kwargs in attempts:
                    try:
                        m = cls(**kwargs)
                        print(f'[MODEL] built by {modname}.{cls_name} kwargs={list(kwargs.keys())}')
                        return m
                    except Exception:
                        pass
    return None

def build_model(cfg: Dict[str, Any], state: Dict[str, torch.Tensor]) -> nn.Module:
    model = try_build_project_model(cfg)
    if model is None:
        print('[MODEL] project model builder not found. using bundled fallback LiteRaceSegNet.')
        num_classes = int((cfg.get('model') or {}).get('num_classes', 2)) if isinstance(cfg.get('model'), dict) else 2
        model = FallbackLiteRaceSegNet(num_classes=num_classes)
    try:
        model.load_state_dict(state, strict=True)
        print('[MODEL] checkpoint loaded strict=True')
    except Exception as e:
        print('[WARN] strict load failed:', e)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print('[MODEL] checkpoint loaded strict=False')
        print('[MODEL] missing:', missing[:20], '...' if len(missing)>20 else '')
        print('[MODEL] unexpected:', unexpected[:20], '...' if len(unexpected)>20 else '')
        if len(unexpected) > 20 or len(missing) > 20:
            print('[WARN] Many missing/unexpected keys. Visual result may be invalid. Check model definition.')
    return model

def list_pairs(root: Path) -> List[Tuple[Path, Path, str]]:
    img_dir = root/'val/images'
    mask_dir = root/'val/masks'
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f'val images/masks not found under {root}')
    pairs=[]
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS: continue
        stem=img.stem
        found=None
        for ext in ('.png','.jpg','.jpeg','.bmp','.webp'):
            p=mask_dir/(stem+ext)
            if p.exists():
                found=p; break
        if found is not None:
            pairs.append((img, found, stem))
    return pairs

def read_image(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')

def read_mask(path: Path) -> Image.Image:
    im = Image.open(path).convert('L')
    arr = np.array(im)
    # any positive value is damage. This matches binary masks.
    arr = (arr > 0).astype(np.uint8)
    return Image.fromarray(arr*255)

def preprocess(img: Image.Image, size_hw: Tuple[int,int], mode: str) -> torch.Tensor:
    H,W=size_hw
    im = img.resize((W,H), Image.BILINEAR)
    arr = np.asarray(im).astype(np.float32)/255.0
    if mode == 'imagenet':
        mean=np.array([0.485,0.456,0.406], dtype=np.float32)
        std=np.array([0.229,0.224,0.225], dtype=np.float32)
        arr=(arr-mean)/std
    # raw = only [0,1]
    t=torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    return t

def _extract_logits(out: Any) -> torch.Tensor:
    """Normalize common segmentation model outputs to a logits tensor.

    Some project/torchvision-style models return OrderedDict({'out': tensor, ...})
    instead of a raw tensor. The first V6 visual hotfix passed that dict directly
    into F.interpolate, which caused: OrderedDict object has no attribute dim.
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        preferred = ('out', 'logits', 'pred', 'prediction', 'mask', 'masks', 'seg', 'main', 'final')
        for k in preferred:
            v = out.get(k)
            if torch.is_tensor(v):
                return v
            if isinstance(v, (tuple, list, dict)):
                try:
                    return _extract_logits(v)
                except Exception:
                    pass
        for v in out.values():
            if torch.is_tensor(v):
                return v
            if isinstance(v, (tuple, list, dict)):
                try:
                    return _extract_logits(v)
                except Exception:
                    pass
    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v):
                return v
            if isinstance(v, (tuple, list, dict)):
                try:
                    return _extract_logits(v)
                except Exception:
                    pass
    raise TypeError(f'Cannot extract logits tensor from model output type={type(out)}')

def predict_mask(model, img: Image.Image, device, size_hw, norm_mode: str) -> np.ndarray:
    orig_w, orig_h = img.size
    x = preprocess(img, size_hw, norm_mode).to(device)
    with torch.no_grad():
        raw_out = model(x)
        out = _extract_logits(raw_out)
        if out.dim() == 3:
            out = out.unsqueeze(0)
        if out.dim() != 4:
            raise RuntimeError(f'Expected logits tensor with 4 dims [N,C,H,W], got shape={tuple(out.shape)}')
        out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        if out.shape[1] == 1:
            prob = torch.sigmoid(out[:,0])
            pred = (prob > 0.5).long()[0]
        else:
            pred = out.argmax(dim=1)[0]
    return pred.cpu().numpy().astype(np.uint8)

def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred=(pred>0).astype(np.uint8); gt=(gt>0).astype(np.uint8)
    inter_d = float(np.logical_and(pred==1, gt==1).sum())
    union_d = float(np.logical_or(pred==1, gt==1).sum())
    inter_b = float(np.logical_and(pred==0, gt==0).sum())
    union_b = float(np.logical_or(pred==0, gt==0).sum())
    iou_d = inter_d / union_d if union_d else 1.0
    iou_b = inter_b / union_b if union_b else 1.0
    miou = (iou_d + iou_b)/2.0
    pix = float((pred==gt).sum()) / float(gt.size)
    pred_area = float(pred.sum()) / float(gt.size)
    gt_area = float(gt.sum()) / float(gt.size)
    return {'pixel_acc':pix, 'miou_binary':miou, 'iou_damage':iou_d, 'iou_background':iou_b, 'pred_damage_ratio':pred_area, 'gt_damage_ratio':gt_area}

def overlay_mask(img: Image.Image, mask: np.ndarray, color=(255, 90, 0), alpha=0.45) -> Image.Image:
    base = img.convert('RGBA')
    m = (mask>0)
    overlay = Image.new('RGBA', base.size, (0,0,0,0))
    arr=np.array(overlay)
    arr[m] = [color[0], color[1], color[2], int(255*alpha)]
    overlay=Image.fromarray(arr)
    return Image.alpha_composite(base, overlay).convert('RGB')

def mask_color(mask: np.ndarray, color=(255,90,0)) -> Image.Image:
    arr=np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    arr[mask>0]=color
    return Image.fromarray(arr)

def make_card(img: Image.Image, gt: np.ndarray, pred: np.ndarray, stem: str, met: Dict[str,float]) -> Image.Image:
    W,H=img.size
    thumb_h=260
    scale=thumb_h/H
    tw=max(1,int(W*scale))
    original=img.resize((tw,thumb_h), Image.BILINEAR)
    gt_ov=overlay_mask(img, gt, color=(0,180,90), alpha=0.45).resize((tw,thumb_h), Image.BILINEAR)
    pred_ov=overlay_mask(img, pred, color=(255,90,0), alpha=0.45).resize((tw,thumb_h), Image.BILINEAR)
    info_w=360
    card=Image.new('RGB', (tw*3+info_w, thumb_h+52), (245,245,245))
    card.paste(original,(0,30)); card.paste(gt_ov,(tw,30)); card.paste(pred_ov,(tw*2,30))
    draw=ImageDraw.Draw(card)
    try:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
        small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    except Exception:
        font=None; small=None
    draw.text((8,6),'original',fill=(0,0,0),font=small)
    draw.text((tw+8,6),'GT overlay',fill=(0,100,40),font=small)
    draw.text((tw*2+8,6),'Prediction overlay',fill=(180,60,0),font=small)
    x=tw*3+18; y=20
    lines=[
        f'V6 visual evidence', f'file: {stem}',
        f'mIoU: {met["miou_binary"]:.4f}', f'damage IoU: {met["iou_damage"]:.4f}',
        f'background IoU: {met["iou_background"]:.4f}', f'pixel acc: {met["pixel_acc"]:.4f}',
        f'pred damage: {met["pred_damage_ratio"]*100:.2f}%', f'GT damage: {met["gt_damage_ratio"]*100:.2f}%',
        'green=GT, orange=pred'
    ]
    for line in lines:
        draw.text((x,y),line,fill=(20,20,20),font=font if y==20 else small)
        y += 26 if y==20 else 22
    draw.text((8,thumb_h+34), stem, fill=(0,0,0), font=small)
    return card

def make_sheet(cards: List[Path], out: Path, cols=2):
    if not cards: return
    ims=[Image.open(p).convert('RGB') for p in cards]
    w=max(im.width for im in ims); h=max(im.height for im in ims)
    rows=math.ceil(len(ims)/cols)
    sheet=Image.new('RGB',(w*cols,h*rows),(235,235,235))
    for i,im in enumerate(ims):
        x=(i%cols)*w; y=(i//cols)*h
        sheet.paste(im,(x,y))
    sheet.save(out, quality=92)

def resolve_norm_mode(args, cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Resolve preprocessing without looking at validation masks.

    Important: the previous hotfix used --norm auto, which tried raw vs imagenet
    on validation GT masks and selected the higher damage IoU. That is not GT-copying,
    but it is validation-label tuning and should not be used for clean evidence.
    This resolver only uses the user argument and configuration text.
    """
    if args.norm in ('raw', 'imagenet'):
        return args.norm, {'source': 'explicit_arg'}

    # config mode: infer from config keys/text only. Never inspect val masks.
    cfg_text = json.dumps(cfg, ensure_ascii=False).lower() if isinstance(cfg, dict) else ''
    imagenet_markers = ['imagenet', '0.485', '0.456', '0.406', '0.229', '0.224', '0.225']
    raw_markers = ['raw', 'to_tensor_only', 'normalize: false', 'normalization: false']
    if any(m in cfg_text for m in imagenet_markers):
        return 'imagenet', {'source': 'config_marker'}
    if any(m in cfg_text for m in raw_markers):
        return 'raw', {'source': 'config_marker'}
    return 'raw', {'source': 'default_raw_no_val_gt_used'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run', default='literace_v6_A_newval_balanced_s42')
    ap.add_argument('--checkpoint', default='best', choices=['best','last'])
    ap.add_argument('--data-root', default='datasets/pothole_binary/processed', help='Use non-augmented current val set for clean visual evidence.')
    ap.add_argument('--config', default='seg/config/pothole_binary_literace_v6_A_newval_balanced.yaml')
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-images', type=int, default=10)
    ap.add_argument('--norm', default='config', choices=['config','raw','imagenet'], help='Preprocess mode. config/raw/imagenet. Does not inspect validation GT.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    root=Path.cwd()
    run_dir=root/'seg/runs'/args.run
    ckpt_path=run_dir/(args.checkpoint+'.pth')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
    out_dir=Path(args.out) if args.out else run_dir/(f'v6_visual_evidence_{args.checkpoint}')
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ('pred_masks','pred_color_masks','pred_overlays','gt_overlays','service_cards'):
        (out_dir/sub).mkdir(exist_ok=True)

    ckpt=load_ckpt(ckpt_path)
    cfg=load_yaml_config(Path(args.config), ckpt)
    state=get_state_dict(ckpt)
    size = (cfg.get('train') or {}).get('image_size', [384,512])
    size_hw=(int(size[0]), int(size[1]))
    device=torch.device(args.device)
    model=build_model(cfg,state).to(device).eval()

    pairs=list_pairs(Path(args.data_root))[:args.max_images]
    if not pairs:
        raise RuntimeError(f'No val pairs found at {args.data_root}')
    norm_mode, norm_scores=resolve_norm_mode(args,cfg)
    print(f'[NORM] selected={norm_mode} info={norm_scores}')

    rows=[]; card_paths=[]
    for imgp, maskp, stem in pairs:
        img=read_image(imgp)
        gt=(np.array(read_mask(maskp))>0).astype(np.uint8)
        pred=predict_mask(model,img,device,size_hw,norm_mode)
        met=metrics(pred,gt)
        row={'file':imgp.name, 'mask':maskp.name, **{k:f'{v:.8f}' for k,v in met.items()}}
        rows.append(row)
        Image.fromarray((pred>0).astype(np.uint8)*255).save(out_dir/'pred_masks'/f'{stem}_pred.png')
        mask_color(pred).save(out_dir/'pred_color_masks'/f'{stem}_pred_color.png')
        overlay_mask(img,pred,color=(255,90,0),alpha=0.45).save(out_dir/'pred_overlays'/f'{stem}_pred_overlay.png', quality=92)
        overlay_mask(img,gt,color=(0,180,90),alpha=0.45).save(out_dir/'gt_overlays'/f'{stem}_gt_overlay.png', quality=92)
        card=make_card(img,gt,pred,stem,met)
        card_path=out_dir/'service_cards'/f'{stem}_service_card.png'
        card.save(card_path, quality=92)
        card_paths.append(card_path)
        print('[VIS]', stem, 'miou=', row['miou_binary'], 'damage=', row['iou_damage'])

    # CSV and summary
    csv_path=out_dir/'v6_visual_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames=list(rows[0].keys())
        wr=csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader(); wr.writerows(rows)
    avg={}
    for k in ['pixel_acc','miou_binary','iou_damage','iou_background','pred_damage_ratio','gt_damage_ratio']:
        avg[k]=float(np.mean([float(r[k]) for r in rows]))
    summary={'run':args.run,'checkpoint':str(ckpt_path),'checkpoint_epoch':ckpt.get('epoch'), 'checkpoint_best':ckpt.get('best'), 'data_root':args.data_root, 'norm_mode':norm_mode, 'norm_resolution_info':norm_scores, 'image_size_hw':size_hw, 'average':avg, 'note':'Visual metrics are regenerated from the saved checkpoint on current validation images. GT masks are used only for metrics/overlays, not for prediction or preprocessing selection. Official training metrics remain train_log.csv.'}
    (out_dir/'v6_visual_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out_dir/'VISUAL_EVIDENCE_NOTE.txt').write_text(
        'This folder was generated by make_v6_visual_evidence.py.\n'
        'Use service_cards and pred_overlays for qualitative inspection.\n'
        'Official numeric result should be taken from train_log.csv / summary_best_last.json; this script is a visualization sanity check.\n'
        f'checkpoint={ckpt_path}\ncheckpoint_epoch={ckpt.get("epoch")}\ncheckpoint_best={ckpt.get("best")}\nselected_norm={norm_mode}\nnorm_resolution={norm_scores}\nGT masks are not used to choose preprocessing.\n', encoding='utf-8')
    make_sheet(card_paths, out_dir/'v6_service_cards_sheet.jpg', cols=2)
    print('[DONE] visual evidence:', out_dir)
    print('[DONE] csv:', csv_path)

if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
