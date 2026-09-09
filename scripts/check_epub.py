#!/usr/bin/env python3
"""Read final XHTML and SVG resources, independently of the build transforms."""
import argparse
import base64
import copy
from collections import Counter
import json
import hashlib
from pathlib import Path,PurePosixPath
import re
import posixpath
from urllib.parse import unquote,urlsplit
import unicodedata
import zipfile
from lxml import etree,html


def normalized(s):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))


def parsed(data):
    tree=html.fromstring(data)
    for e in tree.iter():
        if isinstance(e.tag,str):e.tag=e.tag.rsplit('}',1)[-1]
    return tree


def signature(element):
    # Text order, exact code, table cells, and formulas are checked separately.
    links=[]
    for e in element.xpath('.//a[@href]'):
        # Pandoc resets hidden syntax-highlight line IDs in each EPUB section.
        # These are generated UI anchors, not links authored in Markdown.
        if e.xpath('ancestor::pre') and not ''.join(e.itertext()):continue
        url=e.get('href','');parsed_url=urlsplit(url)
        links.append((normalized(''.join(e.itertext())),unquote(parsed_url.fragment) if not parsed_url.scheme else url))
    return {
        'text':normalized(''.join(element.itertext())),
        'code':[''.join(e.itertext()) for e in element.xpath('.//pre')],
        'cells':[normalized(''.join(e.itertext())) for e in element.xpath('.//th|.//td')],
        'math':[(e.get('data-tex-b64'),e.get('data-display')) for e in element.xpath('.//img[@data-tex-b64]')],
        'images':[e.get('alt','') for e in element.xpath('.//img')],
        'links':links}


def note_text(element):
    element=copy.deepcopy(element)
    for back in element.xpath('.//a[contains(@class,"footnote-back")]'):
        back.drop_tree()
    return normalized(''.join(element.itertext()))


def check(epub_path,stage):
    if not __debug__:raise RuntimeError('Run EPUB validation without -O or PYTHONOPTIMIZE; assertions must remain enabled')
    expected=parsed((stage/'expected.xhtml').read_bytes())
    inputs=json.loads((stage/'inputs.json').read_text())
    assets=json.loads((stage/'assets.json').read_text())
    with zipfile.ZipFile(epub_path) as z:
        names=z.namelist()
        assert names[0]=='mimetype' and z.read('mimetype')==b'application/epub+zip'
        assert z.getinfo('mimetype').compress_type==zipfile.ZIP_STORED
        container=etree.fromstring(z.read('META-INF/container.xml'))
        opf_path=container.xpath('//*[local-name()="rootfile"]/@full-path')[0]
        opf=etree.fromstring(z.read(opf_path));opfdir=PurePosixPath(opf_path).parent
        items={e.get('id'):str(opfdir/e.get('href')) for e in opf.xpath('//*[local-name()="item"]')}
        spine=[items[e.get('idref')] for e in opf.xpath('//*[local-name()="itemref"]')]
        actual={};order=[];trees={}
        for path in names:
            if path.endswith('.xhtml'):trees[path]=parsed(z.read(path))
        for path,tree in trees.items():
            for element in tree.xpath('//*[@src]|//*[@href]'):
                ref=element.get('src') or element.get('href')
                url=urlsplit(ref)
                if url.scheme or url.netloc:continue
                target=posixpath.normpath(posixpath.join(posixpath.dirname(path),unquote(url.path))) if url.path else path
                assert target in names,f'Missing linked resource: {path}: {ref}'
                if url.fragment and target in trees:
                    assert trees[target].xpath('//*[@id=$id]',id=unquote(url.fragment)),f'Broken anchor: {ref}'
        for path in spine:
            tree=trees[path]
            for d in inputs:
                elements=tree.xpath('//*[@id=$id]',id=d['id'])
                for e in elements:
                    assert d['id'] not in actual, 'duplicate book document'
                    actual[d['id']]=signature(e);order.append(d['id'])
        assert order==[d['id'] for d in inputs],f'Document order mismatch {order}'
        for d in inputs:
            original=expected.xpath('//*[@id=$id]',id=d['id'])
            assert len(original)==1
            assert actual[d['id']]==signature(original[0]),f'Content mismatch: {d["id"]}'
        for h in expected.xpath('//h1[@id]|//h2[@id]|//h3[@id]'):
            found=[e for t in trees.values() for e in t.xpath('//*[@id=$id]',id=h.get('id'))]
            assert len(found)==1,'Heading missing or duplicate'
            target=found[0]
            if target.tag not in ('h1','h2','h3'):
                headers=target.xpath('./h1|./h2|./h3')
                assert len(headers)==1,'Heading container malformed'
                target=headers[0]
            assert normalized(''.join(target.itertext()))==normalized(''.join(h.itertext())), 'Heading changed'
        expected_notes=[note_text(e) for e in expected.xpath('//*[@id]') if re.fullmatch(r'fn\d+',e.get('id',''))]
        actual_notes=[note_text(e) for path in spine for e in trees[path].xpath('//*[@id]') if re.fullmatch(r'fn\d+',e.get('id',''))]
        assert expected_notes==actual_notes,'Footnote content mismatch'
        maths=[];diagrams=[]
        math_ids={(a['tex'],'block' if a['display'] else 'inline'):a['id'] for a in assets['math']}
        for path in names:
            if path.endswith('.svg'):
                svg=etree.fromstring(z.read(path))
                if svg.get('data-tex-b64'):
                    entry=(base64.b64decode(svg.get('data-tex-b64')).decode(),svg.get('data-display'));maths.append(entry)
                    if 'svg_sha256' in assets:assert hashlib.sha256(z.read(path)).hexdigest()==assets['svg_sha256'][math_ids[entry]+'.svg'],'Formula SVG bytes changed'
                if svg.get('data-mermaid-b64'):
                    code=base64.b64decode(svg.get('data-mermaid-b64')).decode();diagrams.append(code)
                    if 'svg_sha256' in assets:assert hashlib.sha256(z.read(path)).hexdigest()==assets['svg_sha256'][hashlib.sha256(code.encode()).hexdigest()[:24]+'.svg'],'Diagram SVG bytes changed'
        assert Counter(maths)==Counter((a['tex'],'block' if a['display'] else 'inline') for a in assets['math'])
        assert Counter(diagrams)==Counter(assets['diagrams'].values())
    print(f'EPUB content verified: {len(inputs)} documents, {len(maths)} formula resources, {len(diagrams)} diagrams')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('epub',type=Path);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();check(a.epub,a.stage)
