#!/usr/bin/env python3
"""Render all pages; identify changed body rasters and include adjacent pages for review."""
import argparse,hashlib,json
from pathlib import Path
import pymupdf as fitz
from PIL import Image,ImageDraw

def signature(page):
    clip=fitz.Rect(0,0,page.rect.width,page.rect.height-35)
    return hashlib.sha256(page.get_pixmap(clip=clip,matrix=fitz.Matrix(1,1),alpha=False).samples).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('pdf',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--compare',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);old={}
    if a.compare:
        with fitz.open(a.compare) as d:old={signature(page):i+1 for i,page in enumerate(d)}
    report={'pdf_sha256':hashlib.sha256(a.pdf.read_bytes()).hexdigest(),'unchanged':[],'changed':[]}
    with fitz.open(a.pdf) as d:
        report['pages']=len(d)
        for i,page in enumerate(d):
            page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(a.output/f'page-{i+1:03d}.png')
            prior=old.get(signature(page))
            if prior:report['unchanged'].append({'new':i+1,'old':prior})
            else:report['changed'].append(i+1)
        selected=sorted({n for page in report['changed'] for n in [page-1,page,page+1] if 1<=n<=len(d)})
    report['review_pages']=selected
    for start in range(0,len(selected),6):
        sheet=Image.new('RGB',(1800,1740),'#d4dce0');draw=ImageDraw.Draw(sheet)
        for j,n in enumerate(selected[start:start+6]):
            im=Image.open(a.output/f'page-{n:03d}.png');im.thumbnail((590,830));x=(j%3)*600;y=(j//3)*870
            sheet.paste(im,(x,y+30));draw.text((x+10,y+8),f'Page {n}',fill='black')
        sheet.save(a.output/f'sheet-{start//6+1:02d}.jpg',quality=90)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{report["pages"]} pages rendered; {len(report["changed"])} changed bodies, {len(selected)} pages including neighbors to review')
if __name__=='__main__':main()
