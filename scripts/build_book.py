#!/usr/bin/env python3
"""Build the actual Markdown list; no prose is injected by the converter."""
from __future__ import annotations
import argparse
import base64
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import pymupdf as fitz
from booklib import ROOT,PANDOC,load,combined,visit,image_node,run,pandoc


def render_assets(source, stage):
    epub=copy.deepcopy(source); pdf=copy.deepcopy(source)
    descriptions=json.loads((ROOT/'publishing/math-descriptions.json').read_text())
    captions=[]
    def find_captions(value):
        if isinstance(value,dict):
            for child in value.values():find_captions(child)
        elif isinstance(value,list):
            for i,node in enumerate(value):
                if isinstance(node,dict) and node.get('t')=='CodeBlock' and 'mermaid' in node['c'][0][1]:
                    following=value[i+1] if i+1<len(value) else {}
                    if following.get('t')!='Para':raise ValueError('Diagram needs adjacent prose caption')
                    text=''.join(n['c'] if n['t']=='Str' else ' ' if n['t'] in ('Space','SoftBreak') else n['c'][1] if n['t'] in ('Code','Math') else '' for n in visit(following))
                    captions.append(text.split('。')[0]+'。读图说明见紧随图后的正文。')
                find_captions(node)
    find_captions(source)
    for table in (n for n in visit(pdf) if n['t']=='Table'):
        c=table['c'];rows=c[3][1]+[r for body in c[4] for r in body[2]+body[3]]+c[5][1]
        widths=[5.0]*len(c[2])
        for row in rows:
            for index,cell in enumerate(row[1]):
                chars=''.join(n['c'] if n['t']=='Str' else n['c'][1] if n['t']=='Code' else 'MM' if n['t']=='Math' else '' for n in visit(cell[4]))
                length=sum(.55 if ord(ch)<128 else 1 for ch in chars)
                widths[index]=max(widths[index],min(length,32))
        weights=[math.sqrt(x) for x in widths];total=sum(weights)
        for i,spec in enumerate(c[2]):spec[1]={'t':'ColWidth','c':weights[i]/total}
    media=stage/'media';media.mkdir(exist_ok=True)
    math_items={}
    for node in visit(source):
        if node['t']=='Math':
            display=node['c'][0]['t']=='DisplayMath'; tex=node['c'][1]
            key=hashlib.sha256((str(display)+tex).encode()).hexdigest()[:24]
            math_items[key]=dict(id=key,tex=tex,display=display)
    request=stage/'math-request.json';request.write_text(json.dumps(list(math_items.values())))
    math_result=json.loads(run(['node',ROOT/'scripts/render_math.cjs',request,media],text=True,capture_output=True).stdout)
    math_sizes={item['id']:item for item in math_result}
    for node in visit(epub):
        if node['t']=='Math':
            display=node['c'][0]['t']=='DisplayMath';tex=node['c'][1]
            key=hashlib.sha256((str(display)+tex).encode()).hexdigest()[:24]
            size=math_sizes[key]
            mode='math-display' if display else 'math-inline'
            style=f'width:{size["width"]};height:{size["height"]};vertical-align:{size["vertical_align"]};'
            if display:
                description=descriptions.get(key,'排版测试公式，计算含义见相邻说明。')
            else:
                description='数学记号：'+tex
                symbols={'alpha':'α','beta':'β','gamma':'γ','theta':'θ','lambda':'λ','sigma':'σ','epsilon':'ε','pi':'π','omega':'ω','eta':'η','times':'乘','in':'属于'}
                description=re.sub(r'\\('+'|'.join(symbols)+r')(?![A-Za-z])',lambda m:symbols[m.group(1)],description)
                for name in ['mathcal','mathrm','mathbb','mathbf','mathsf','operatorname']:
                    description=description.replace('\\'+name,'')
                description=description.replace('{','(').replace('}',')')
            replacement=image_node(media/f'{key}.svg',description,[mode],
                [['style',style],['data-tex-b64',base64.b64encode(tex.encode()).decode()],['data-display',str(display).lower()]])
            node.clear();node.update(replacement)
    diagrams={}
    renderer_hash=hashlib.sha256((ROOT/'publishing/mermaid.json').read_bytes()+(ROOT/'publishing/puppeteer.json').read_bytes()).hexdigest()
    for node in visit(source):
        if node['t']=='CodeBlock' and 'mermaid' in node['c'][0][1]:
            code=node['c'][1];key=hashlib.sha256(code.encode()).hexdigest()[:24]
            mmd=media/f'{key}.mmd';mmd.write_text(code)
            pdfpath=media/f'{key}.pdf'
            cache_tag=media/f'{key}.renderer'
            if not pdfpath.exists() or not cache_tag.exists() or cache_tag.read_text()!=renderer_hash:
                run([ROOT/'node_modules/.bin/mmdc','-i',mmd,'-o',pdfpath,'--pdfFit',
                    '-c',ROOT/'publishing/mermaid.json','-p',ROOT/'publishing/puppeteer.json','-b','white'],
                    stdout=open(stage/f'{key}-mermaid.log','w'),stderr=None)
                cache_tag.write_text(renderer_hash)
            with fitz.open(pdfpath) as diagram:
                if len(diagram)!=1:raise ValueError('Diagram must fit one page')
                svg=diagram[0].get_svg_image(text_as_path=True)
            tree=ET.fromstring(svg);tree.set('data-mermaid-b64',base64.b64encode(code.encode()).decode())
            ET.ElementTree(tree).write(media/f'{key}.svg',encoding='unicode')
            diagrams[key]=code
    for ast,ext in [(epub,'svg'),(pdf,'pdf')]:
        counter=0
        for node in visit(ast):
            if node['t']=='CodeBlock' and 'mermaid' in node['c'][0][1]:
                counter+=1;key=hashlib.sha256(node['c'][1].encode()).hexdigest()[:24]
                replacement={'t':'Para','c':[image_node(media/f'{key}.{ext}',captions[counter-1],['diagram'])]}
                node.clear();node.update(replacement)
    def group_captions(value):
        if isinstance(value,dict):
            for item in value.values():group_captions(item)
        elif isinstance(value,list):
            for item in value:group_captions(item)
            i=0
            while i+1<len(value):
                block=value[i];next_block=value[i+1]
                if (isinstance(block,dict) and block.get('t')=='Para' and len(block['c'])==1
                    and block['c'][0].get('t')=='Image' and 'diagram' in block['c'][0]['c'][0][1]
                    and isinstance(next_block,dict) and next_block.get('t')=='Para'):
                    value[i:i+2]=[{'t':'RawBlock','c':['latex','\\begin{minipage}{\\linewidth}']},block,next_block,
                        {'t':'RawBlock','c':['latex','\\end{minipage}\\par\\medskip']}]
                    i+=4
                else:i+=1
            i=0
            while i<len(value):
                table=value[i]
                if isinstance(table,dict) and table.get('t')=='CodeBlock':
                    lines=table['c'][1].splitlines()
                    wrapped=sum(max(1,math.ceil(sum(.6 if ord(c)<128 else 1.1 for c in line)/32)) for line in lines)
                    if wrapped<=14:
                        value.insert(i,{'t':'RawBlock','c':['latex',f'\\Needspace{{{20+wrapped*14}pt}}']});i+=1
                if isinstance(table,dict) and table.get('t')=='Table':
                    c=table['c'];rows=c[3][1]+[r for body in c[4] for r in body[2]+body[3]]+c[5][1]
                    if len(rows)<=9:
                        colwidth=328/max(len(c[2]),1)-14
                        height=24
                        for row in rows:
                            widths=[]
                            for cell in row[1]:
                                chars=''.join(n['c'] if n['t']=='Str' else n['c'][1] if n['t']=='Code' else 'MM' if n['t']=='Math' else '' for n in visit(cell[4]))
                                widths.append(sum(5.5 if ord(ch)<128 else 11 for ch in chars))
                            height+=max(1,math.ceil(max(widths,default=0)/max(colwidth,1)))*15
                        if height<430:
                            value.insert(i,{'t':'RawBlock','c':['latex',f'\\Needspace{{{height}pt}}']});i+=1
                i+=1
    group_captions(pdf)
    escaped={'\\':r'\textbackslash{}','{':r'\{','}':r'\}','$':r'\$','&':r'\&',
        '%':r'\%','#':r'\#','_':r'\_','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}
    for node in visit(pdf):
        if node['t']=='Code':
            text=node['c'][1];pieces=[];span=0
            for i,char in enumerate(text):
                pieces.append(escaped.get(char,char));span+=1
                boundary=char in '_./=,()|:-' or (i+1<len(text) and char.islower() and text[i+1].isupper())
                if boundary or span>=8:pieces.append(r'\allowbreak{}');span=0
            replacement={'t':'RawInline','c':['latex',r'\texttt{'+''.join(pieces)+'}']}
            node.clear();node.update(replacement)
    hashes={f'{key}.svg':hashlib.sha256((media/f'{key}.svg').read_bytes()).hexdigest() for key in set(math_items)|set(diagrams)}
    return epub,pdf,dict(math=list(math_items.values()),diagrams=diagrams,svg_sha256=hashes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--edition',choices=['preview','release'],default='preview')
    p.add_argument('--chapters',nargs='+');p.add_argument('--smoke',action='store_true')
    p.add_argument('--format',choices=['epub','pdf','all'],default='all');args=p.parse_args()
    if args.edition=='release' and args.format!='all':raise ValueError('Release candidate requires both formats')
    from check_book import check
    check(args.edition,args.chapters,args.smoke)
    config,docs=load(args.edition,args.chapters,args.smoke)
    if args.edition=='release':
        for key in ['title','author','lang','identifier','rights','version']:
            if not config.get(key):raise ValueError(f'Missing release metadata: {key}')
        for path in ['assets/cover.png','assets/cover.pdf','book/appendices/licenses.md']:
            if not (ROOT/path).is_file():raise ValueError(f'Missing release resource: {path}')
    source=combined(config,docs)
    stage=ROOT/'build'/args.edition;stage.mkdir(parents=True,exist_ok=True)
    (stage/'source.json').write_text(json.dumps(source,ensure_ascii=False,indent=2))
    (stage/'inputs.json').write_text(json.dumps([{k:d[k] for k in ['id','path','sha256']} for d in docs],ensure_ascii=False,indent=2))
    epub,pdf,assets=render_assets(source,stage)
    if args.edition=='release':
        descriptions=json.loads((ROOT/'publishing/math-descriptions.json').read_text())
        if any(x['display'] and x['id'] not in descriptions for x in assets['math']):raise ValueError('Missing reviewed display-math description')
    from content_contract import verify
    (stage/'content-contract.json').write_text(json.dumps(verify(source,epub),indent=2))
    (stage/'assets.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2))
    title=config['title']+('（预览·未完成）' if args.edition=='preview' else '')
    metadata={k:config[k] for k in ['subtitle','author','lang','date','identifier','rights','version'] if k in config}
    metadata.update(title=title,description='收录：'+', '.join(d['id'] for d in docs))
    (stage/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False))
    meta_args=['--metadata-file',stage/'metadata.json']
    epub_json=stage/'epub.json';epub_json.write_text(json.dumps(epub,ensure_ascii=False))
    (stage/'expected.xhtml').write_text(pandoc(epub,'html5',['--standalone',*map(str,meta_args)]))
    stem='minimind-source-guide'+('-preview' if args.edition=='preview' else '')
    outputs=[]
    if args.format in ('epub','all'):
        output=stage/f'{stem}.epub'
        cover_args=['--epub-cover-image',ROOT/'assets/cover.png'] if args.edition=='release' else []
        run([PANDOC,'-f','json',epub_json,'-t','epub3','--toc','--toc-depth=2','--epub-embed-font',ROOT/'assets/fonts/MiniMindBookCJK.otf',
            '--css',ROOT/'publishing/epub/style.css',*meta_args,*cover_args,'-o',output])
        run([ROOT/'.venv/bin/python',ROOT/'scripts/check_epub.py',output,'--stage',stage])
        run(['java','-jar',ROOT/'.tools/epubcheck-5.3.0/epubcheck.jar',output])
        outputs.append(output)
    if args.format in ('pdf','all'):
        header=stage/'header.tex'
        header.write_text((ROOT/'publishing/pdf/header.tex').read_text().replace('\\setCJKmonofont{Noto Sans CJK SC}',
            f'\\setCJKmonofont[Path={ROOT}/assets/fonts/]{{NotoSansCJKsc-Regular.otf}}'))
        tex=pandoc(pdf,'latex',['--standalone','--top-level-division=chapter','--toc',
            '--include-in-header',str(header),*map(str,meta_args),
            '-V','documentclass=book','-V','classoption=oneside','-V','papersize=a5',
            '-V','geometry:margin=16mm','-V','fontsize=11pt','-V','CJKmainfont=NotoSansCJKsc-Regular.otf',
            '-V',f'CJKoptions=Path={ROOT}/assets/fonts/','-V','monofont=SourceCodePro-Regular.otf',
            '-V',f'monofontoptions=Path={ROOT}/assets/fonts/','-V','linestretch=1.2'])
        texpath=stage/f'{stem}.tex';texpath.write_text(tex)
        env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(ROOT/'.cache/tectonic')
        run([ROOT/'.tools/tectonic','--keep-logs','--keep-intermediates',texpath],env=env)
        if args.edition=='release':
            target=stage/f'{stem}.pdf'
            with fitz.open(target) as doc,fitz.open(ROOT/'assets/cover.pdf') as cover:
                before=doc.get_toc();doc.insert_pdf(cover,start_at=0)
                after=doc.get_toc()
                if after!=[[level,title,page+1] for level,title,page in before]:
                    doc.set_toc([[level,title,page+1] for level,title,page in before])
                doc.save(stage/f'{stem}-covered.pdf')
            (stage/f'{stem}-covered.pdf').replace(target)
        run([ROOT/'.venv/bin/python',ROOT/'scripts/check_pdf.py',stage/f'{stem}.pdf','--stage',stage])
        outputs.append(stage/f'{stem}.pdf')
    if args.edition=='release' and args.format!='all':raise ValueError('Release requires both formats')
    if args.edition=='preview':
        outdir=ROOT/'dist';outdir.mkdir(exist_ok=True)
        for f in outputs:shutil.copyfile(f,outdir/f.name)
        print('Built preview:',*[str(outdir/f.name) for f in outputs],sep='\n')
    else:
        print('Release candidates only; publication still requires final visual and reader checks:',*map(str,outputs),sep='\n')


if __name__=='__main__':main()
