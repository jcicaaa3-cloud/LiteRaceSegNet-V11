#!/usr/bin/env python3
import argparse, shutil, hashlib
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
IMG_EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}

def find_pairs(split_dir):
    img_dir=split_dir/'images'; mask_dir=split_dir/'masks'; pairs=[]
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS: continue
        m=None
        for ext in ['.png','.jpg','.jpeg','.bmp','.webp']:
            c=mask_dir/(img.stem+ext)
            if c.exists(): m=c; break
        if m: pairs.append((img,m))
    return pairs

def open_i(p): return Image.open(p).convert('RGB')
def open_m(p): return Image.open(p).convert('L')
def bin_m(m): return Image.fromarray(((np.array(m)>0).astype('uint8')*255), 'L')
def resize_pair(i,m,w,h): return i.resize((w,h), Image.Resampling.BILINEAR), bin_m(m.resize((w,h), Image.Resampling.NEAREST))
def save_pair(i,m,oi,om,name,ext,q):
    oi.mkdir(parents=True,exist_ok=True); om.mkdir(parents=True,exist_ok=True)
    if ext=='jpg': i.save(oi/(name+'.jpg'), quality=q, optimize=True)
    else: i.save(oi/(name+'.png'))
    bin_m(m).save(om/(name+'.png'))
def bbox(m):
    a=np.array(m); y,x=np.where(a>0)
    if len(x)==0: return None
    return int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)
def expand(b,w,h,scale):
    x1,y1,x2,y2=b; cx=(x1+x2)/2; cy=(y1+y2)/2
    bw=max(24,(x2-x1)*scale); bh=max(24,(y2-y1)*scale)
    cw=max(bw,bh*1.2); ch=max(bh,bw*.8)
    nx1=max(0,int(cx-cw/2)); ny1=max(0,int(cy-ch/2)); nx2=min(w,int(cx+cw/2)); ny2=min(h,int(cy+ch/2))
    if nx2<=nx1+8 or ny2<=ny1+8: return None
    return nx1,ny1,nx2,ny2
def crop_resize(i,m,box,w,h): return resize_pair(i.crop(box),m.crop(box),w,h)
def quads(w,h): return [(0,0,w//2,h//2),(w//2,0,w,h//2),(0,h//2,w//2,h),(w//2,h//2,w,h)]
def soft_shadow(img, stem):
    w,h=img.size
    overlay=Image.new('RGBA',(w,h),(0,0,0,0))
    draw=ImageDraw.Draw(overlay)
    seed=int(hashlib.md5(stem.encode()).hexdigest()[:8],16)
    x0=(seed % max(1,w//2)); x1=min(w, x0 + w//2 + (seed % max(1,w//4)))
    y0=max(0, (seed//7) % max(1,h//3) - h//8); y1=min(h, y0 + h//2)
    draw.polygon([(x0,y0),(x1,y0+h//8),(x1,y1),(max(0,x0-w//5),y1-h//8)], fill=(0,0,0,45))
    return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
def light_noise(img, stem):
    arr=np.array(img).astype(np.int16)
    seed=int(hashlib.md5((stem+'noise').encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    noise=rng.normal(0,5,arr.shape).astype(np.int16)
    return Image.fromarray(np.clip(arr+noise,0,255).astype(np.uint8),'RGB')

def process_train(src,out,w,h,ext,q,skip_empty):
    pairs=find_pairs(src/'train'); count=0; oi=out/'train/images'; om=out/'train/masks'
    for imgp,maskp in pairs:
        stem=imgp.stem; i,m=resize_pair(open_i(imgp),open_m(maskp),w,h); vars=[]
        vars += [('orig',i,m),('hflip',i.transpose(Image.Transpose.FLIP_LEFT_RIGHT),m.transpose(Image.Transpose.FLIP_LEFT_RIGHT))]
        vars += [('dark085',ImageEnhance.Brightness(i).enhance(.85),m),('bright115',ImageEnhance.Brightness(i).enhance(1.15),m),('contrast110',ImageEnhance.Contrast(i).enhance(1.10),m)]
        vars += [('sharp115',ImageEnhance.Sharpness(i).enhance(1.15),m),('blur_soft',i.filter(ImageFilter.GaussianBlur(radius=0.6)),m)]
        sh=soft_shadow(i,stem); vars += [('shadow_soft',sh,m),('shadow_soft_hflip',sh.transpose(Image.Transpose.FLIP_LEFT_RIGHT),m.transpose(Image.Transpose.FLIP_LEFT_RIGHT))]
        vars += [('noise_light',light_noise(i,stem),m)]
        for k,box in enumerate(quads(w,h),1):
            ci,cm=crop_resize(i,m,box,w,h)
            if (not skip_empty) or np.array(cm).max()>0: vars.append((f'quad{k}',ci,cm))
        b=bbox(m)
        if b:
            for lab,sc in [('damage_tight',2.2),('damage_wide',3.5),('damage_context',5.0)]:
                box=expand(b,w,h,sc)
                if box:
                    ci,cm=crop_resize(i,m,box,w,h)
                    vars.append((lab,ci,cm)); vars.append((lab+'_hflip',ci.transpose(Image.Transpose.FLIP_LEFT_RIGHT),cm.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
        for suf,vi,vm in vars:
            save_pair(vi,vm,oi,om,stem+'_'+suf,ext,q); count+=1
    return len(pairs),count

def process_val(src,out,w,h,ext,q):
    pairs=find_pairs(src/'val'); oi=out/'val/images'; om=out/'val/masks'; count=0
    for imgp,maskp in pairs:
        i,m=resize_pair(open_i(imgp),open_m(maskp),w,h); save_pair(i,m,oi,om,imgp.stem,ext,q); count+=1
    return len(pairs),count

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--src',required=True); ap.add_argument('--out',required=True); ap.add_argument('--width',type=int,default=512); ap.add_argument('--height',type=int,default=384); ap.add_argument('--image-ext',choices=['jpg','png'],default='jpg'); ap.add_argument('--jpeg-quality',type=int,default=92); ap.add_argument('--skip-empty-quadrants',action='store_true')
    a=ap.parse_args(); src=Path(a.src); out=Path(a.out)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    ti,to=process_train(src,out,a.width,a.height,a.image_ext,a.jpeg_quality,a.skip_empty_quadrants)
    vi,vo=process_val(src,out,a.width,a.height,a.image_ext,a.jpeg_quality)
    print('[DONE] V6 paired augmentation')
    print(f'train input pairs: {ti}')
    print(f'train output pairs: {to}')
    print(f'val input pairs: {vi}')
    print(f'val output pairs: {vo}')
    print(f'output: {out}')
if __name__=='__main__': main()
