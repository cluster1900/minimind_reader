#!/usr/bin/env python3
"""Validate selected manuscript inputs, source anchors and native mathematics."""
import argparse
import ast
from collections import Counter
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit
from booklib import ROOT,load,combined,visit,strip_meta
from source_snapshot import SourceSnapshot


def require(condition,message):
    if not condition:raise ValueError(message)


def check_metadata(cfg,docs,edition):
    known={e['id'] for e in cfg['chapters']}
    require(len(known)==len(cfg['chapters']),'Duplicate chapter IDs in publication manifest')
    dependencies={}
    for d in docs:
        m=d['meta']
        if d['id'].split('-')[0] not in known:continue
        needed={'title','chapter_id','status','audience','prerequisites','sources','verified_at',
                'validation','validation_records','scope'}
        require(needed<=m.keys(),f'Missing chapter metadata: {d["path"]}: {needed-m.keys()}')
        cid=m['chapter_id']
        require(m['status'] in {'outline','draft','verified','stale'},f'Invalid status: {cid}')
        require(m['audience'] in {'core','advanced'},f'Invalid audience: {cid}')
        require(isinstance(m['prerequisites'],list) and set(m['prerequisites'])<=known,
                f'Unknown prerequisite: {cid}')
        dependencies[cid]=m['prerequisites']
        require(m['sources'] and m['scope'],f'Missing source/scope: {cid}')
        for source in m['sources']:
            require(re.fullmatch(r'jingyaogong/minimind(?:-v|-o)?',source.get('repository','')) and
                    re.fullmatch('[0-9a-f]{40}',source.get('commit','')),f'Invalid source metadata: {cid}')
            require(source.get('dirty') is False,f'Chapter needs a fixed clean source snapshot: {cid}')
        require(set(m['validation'])=={'static','cpu','gpu'},f'Missing validation dimensions: {cid}')
        valid={'pending','passed','failed','not_run','not_required'}
        require(set(m['validation'].values())<=valid,f'Invalid validation status: {cid}')
        for record in m['validation_records']:
            require((ROOT/record).is_file(),f'Missing validation record: {cid}: {record}')
        if m['status']=='verified':
            require(m['verified_at'] and m['validation_records'],f'Unsubstantiated verified status: {cid}')
            require(m['validation']['static']=='passed' and
                    not set(m['validation'].values())&{'pending','failed'},f'Incomplete verification: {cid}')
        if edition=='release':require(m['status']=='verified',f'Unverified release chapter: {cid}')
    # Include declared dependencies in unselected existing chapters, so a preview
    # cannot hide a cycle simply by omitting the other side of it.
    for e in cfg['chapters']:
        if e['id'] not in dependencies and (ROOT/e['files'][0]).is_file():
            meta,_=strip_meta((ROOT/e['files'][0]).read_text())
            dependencies[e['id']]=meta.get('prerequisites',[])
    def walk(cid,chain):
        require(cid not in chain,'Cyclic learning dependency: '+' -> '.join([*chain,cid]))
        for other in dependencies.get(cid,[]):walk(other,[*chain,cid])
    for cid in dependencies:walk(cid,[])


def scan_math(body):
    def blank(m):return ''.join('\n' if c=='\n' else ' ' for c in m.group())
    text=re.sub(r'(?ms)^```[^\n]*\n.*?^```\s*$',blank,body)
    text=re.sub(r'(`+)(.*?)\1',blank,text)
    result=[];i=0
    while i<len(text):
        if text[i]!='$' or (i>0 and text[i-1]=='\\'):i+=1;continue
        delimiter='$$' if text.startswith('$$',i) else '$';start=i;i+=len(delimiter)
        end=text.find(delimiter,i)
        if end<0:raise ValueError('Unclosed math delimiter')
        value=text[i:end]
        if not value.strip():raise ValueError('Empty math')
        if delimiter=='$' and ('\n' in value or value!=value.strip()):raise ValueError('Invalid inline math')
        if delimiter=='$$':
            if not value.startswith('\n') or not value.endswith('\n'):raise ValueError('Display math needs independent lines')
        result.append((delimiter=='$$',' '.join(value.split())))
        i=end+len(delimiter)
    return result


def check(edition,chapters,smoke=False,require_review=True):
    cfg,docs=load(edition,chapters,smoke)
    src=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))
    check_metadata(cfg,docs,edition)
    snapshot=SourceSnapshot(src)
    for d in docs:
        body=d['body'];nodes=list(visit(d['ast']))
        scanned=Counter(scan_math(body))
        parsed=Counter((n['c'][0]['t']=='DisplayMath',' '.join(n['c'][1].split())) for n in nodes if n['t']=='Math')
        require(scanned==parsed,f'Math parser disagreement: {d["path"]}')
        for diagram in re.finditer(r'```mermaid\n.*?\n```',body,re.S):
            following=body[diagram.end():].lstrip()
            require(re.match(r'图\s*\d',following),f'Missing caption and reading explanation: {d["path"]}')
        declared={(s['repository'].split('/')[-1],s['commit']) for s in d['meta'].get('sources',[])}
        for node in nodes:
            if node['t']!='Link':continue
            url=urlsplit(node['c'][2][0]);path=unquote(url.path)
            if url.netloc!='github.com':continue
            match=re.fullmatch(r'/jingyaogong/(minimind(?:-v|-o)?)/blob/([^/]+)/(.+)',path)
            if not match:continue
            repo,commit,file=match.groups()
            if declared:require((repo,commit) in declared,f'Citation/frontmatter mismatch: {d["path"]}: {path}')
            snapshot.read(repo,commit,file)
    mapping_path=ROOT/'meta/chapter-source-map.json'
    if mapping_path.exists():
        mapping=json.loads(mapping_path.read_text())
        selected={d['meta'].get('chapter_id'):d for d in docs if d['meta'].get('chapter_id')}
        require(set(selected)<={e['chapter_id'] for e in mapping},'Chapter has no source map')
        for entry in mapping:
            if entry['chapter_id'] not in selected:continue
            declared=selected[entry['chapter_id']]['meta']['sources']
            require(any(s['repository']=='jingyaogong/'+entry['repository'] and s['commit']==entry['commit']
                        for s in declared),f'Source map/frontmatter mismatch: {entry}')
            code=snapshot.read(entry['repository'],entry['commit'],entry['path'])
            tree=ast.parse(code)
            symbols=set()
            def collect(nodes,prefix=''):
                for n in nodes:
                    if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                        name=prefix+n.name;symbols.add(name);collect(n.body,name+'.')
            collect(tree.body)
            require(set(entry['symbols'])<=symbols,f'Unknown source symbols: {entry}')
    if edition=='release' and require_review:
        from check_evidence import check_evidence
        check_evidence(docs,snapshot.manifest())
    combined(cfg,docs,allow_missing_internal=(edition=='preview')) # resolves duplicate heading IDs and every internal link
    print(f'Manuscript checks passed for {len(docs)} input documents')
    return {'documents':docs,'source_files':snapshot.manifest()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--edition',default='preview',choices=['preview','release'])
    p.add_argument('--chapters',nargs='+');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();check(a.edition,a.chapters,a.smoke)
