#!/usr/bin/env python3
"""Check PDF text coverage and page bounds; visual QA remains mandatory."""
import argparse
import json
from pathlib import Path
import re
import unicodedata
import pymupdf


def heading_text(nodes):
    return ''.join(n.get('c','') if n['t']=='Str' else ' ' if n['t'] in ('Space','SoftBreak')
                   else heading_text(n['c']) if n['t'] in ('Emph','Strong') else '' for n in nodes)


def normalized(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text)).replace('\u00ad','').replace('↪','')


def prose_chunks(inlines):
    parts=['']
    def add(nodes):
        for node in nodes:
            t=node['t'];c=node.get('c')
            if t=='Str':parts[-1]+=c
            elif t in ('Space','SoftBreak','LineBreak'):parts[-1]+=' '
            elif t=='Code':parts[-1]+=c[1]
            elif t in ('Math','Note','Image'):parts.append('')
            elif t in ('Emph','Strong','Strikeout','Superscript','Subscript','SmallCaps'):add(c)
            elif t in ('Link','Span'):add(c[1])
            elif t=='Quoted':
                parts[-1]+='“' if c[0]['t']=='DoubleQuote' else '‘';add(c[1]);parts[-1]+='”' if c[0]['t']=='DoubleQuote' else '’'
    add(inlines)
    return [normalized(p) for p in parts if len(normalized(p))>=8]


def source_probes(value):
    if isinstance(value,dict):
        t=value.get('t')
        if t in ('Para','Plain'):yield from prose_chunks(value['c'])
        elif t=='Header':yield from prose_chunks(value['c'][2])
        elif t=='CodeBlock' and 'mermaid' not in value['c'][0][1]:
            yield from (normalized(s) for s in value['c'][1].splitlines() if len(normalized(s))>=8)
        for child in value.values():yield from source_probes(child)
    elif isinstance(value,list):
        for child in value:yield from source_probes(child)


def check(file,stage):
    if not __debug__:raise RuntimeError('Run PDF validation without -O or PYTHONOPTIMIZE; assertions must remain enabled')
    source=json.loads((stage/'source.json').read_text())
    doc=pymupdf.open(file);all_text=[];bounds=[]
    expected=[normalized(heading_text(n['c'][2])) for n in source['blocks'] if n.get('t')=='Header' and n['c'][0]==1]
    chapters=[(normalized(title),page) for level,title,page in doc.get_toc() if level==1]
    assert [title for title,page in chapters]==expected,'PDF chapter bookmark order differs from Markdown'
    pages=[page for title,page in chapters]
    assert pages==sorted(set(pages)),'PDF chapter destinations are duplicated or out of order'
    for title,page in chapters:
        assert 1<=page<=len(doc) and title in normalized(doc[page-1].get_text()),f'PDF bookmark points to wrong chapter: {title}'
    for i,page in enumerate(doc):
        # The template places only running header/footer outside this region.
        # Preserve descenders on the last body line. The page number is below
        # this 30pt exclusion; a 42pt crop can cut a code underscore in half.
        body=page.get_text(clip=pymupdf.Rect(0,42,page.rect.width,page.rect.height-30))
        all_text.append(body)
        assert '\ufffd' not in body,f'Missing/replaced glyph on page {i+1}'
        for b in page.get_text('blocks'):
            if b[0]<-1 or b[1]<-1 or b[2]>page.rect.width+1 or b[3]>page.rect.height+1:
                bounds.append(dict(page=i+1,rect=list(b[:4])))
    assert not bounds,f'Page overflow: {bounds}'
    flat=normalized(''.join(all_text));probes=list(source_probes(source['blocks']))
    missing=[s for s in probes if s not in flat]
    report=dict(pages=len(doc),bookmarks=len(doc.get_toc()),text_probes=len(probes),missing=missing,bounds=bounds,
        scope='Paragraph fragments, code lines, table text and page bounds. Formula meaning and visual layout need separate review.')
    (stage/'pdf-text-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    assert not missing,f'PDF text coverage failed: {missing[:5]}'
    assert doc.get_toc(),'Missing PDF bookmarks'
    print(f'PDF text checks passed: {len(doc)} pages, {len(probes)} source text probes')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pdf',type=Path);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();check(a.pdf,a.stage)
