import argparse, math
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
EXTS={'.png','.jpg','.jpeg','.webp','.bmp'}

def main():
    ap=argparse.ArgumentParser(description='Make contact sheet from result images')
    ap.add_argument('--input_dir',required=True); ap.add_argument('--out',required=True); ap.add_argument('--pattern',default='*.png'); ap.add_argument('--thumb_w',type=int,default=360); ap.add_argument('--cols',type=int,default=2)
    args=ap.parse_args()
    files=sorted(Path(args.input_dir).glob(args.pattern))
    files=[p for p in files if p.suffix.lower() in EXTS]
    if not files: raise FileNotFoundError('no images matched')
    thumbs=[]
    font=ImageFont.load_default()
    for p in files:
        img=Image.open(p).convert('RGB'); ratio=args.thumb_w/img.width; h=max(1,int(img.height*ratio)); img=img.resize((args.thumb_w,h))
        canvas=Image.new('RGB',(args.thumb_w,h+24),'white'); canvas.paste(img,(0,0)); ImageDraw.Draw(canvas).text((4,h+6),p.name,fill=(0,0,0),font=font); thumbs.append(canvas)
    rows=math.ceil(len(thumbs)/args.cols); row_h=max(t.height for t in thumbs); sheet=Image.new('RGB',(args.cols*args.thumb_w,rows*row_h),'white')
    for i,t in enumerate(thumbs): sheet.paste(t,((i%args.cols)*args.thumb_w,(i//args.cols)*row_h))
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); sheet.save(args.out)
    print('[OK]',args.out)
if __name__=='__main__': main()
