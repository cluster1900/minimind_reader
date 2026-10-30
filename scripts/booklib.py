"""Shared input parsing; output verification lives separately in check_epub.py."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import yaml
import pypandoc

ROOT = Path(__file__).resolve().parents[1]
PANDOC = pypandoc.get_pandoc_path()
FORMAT = 'gfm+tex_math_dollars+attributes+raw_html'


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def pandoc(document, target, extra=()):
    return run([PANDOC, '-f', 'json', '-t', target, *extra],
        input=json.dumps(document), text=True, capture_output=True).stdout


def visit(value):
    if isinstance(value, dict):
        if 't' in value: yield value
        for child in value.values(): yield from visit(child)
    elif isinstance(value, list):
        for child in value: yield from visit(child)


def attr(identifier='', classes=(), pairs=()):
    return [identifier, list(classes), list(pairs)]


def image_node(path, description, classes=(), pairs=()):
    return {'t':'Image','c':[attr(classes=classes,pairs=pairs),[{'t':'Str','c':description}],[str(path),'']]}


def strip_meta(text):
    if text.startswith('---\n'):
        pieces=text.split('\n---\n',1)
        if len(pieces)!=2: raise ValueError('Unclosed YAML frontmatter')
        return yaml.safe_load(pieces[0][4:]),pieces[1]
    return {},text


def load(edition, chosen=None, smoke=False):
    config=yaml.safe_load((ROOT/'publishing/book.yaml').read_text())
    entries=config['chapters']
    if edition=='preview':
        if not chosen: raise ValueError('Preview requires explicit --chapters')
        known={e['id'] for e in entries}
        if set(chosen)-known: raise ValueError('Unknown chapter selection')
        entries=[e for e in entries if e['id'] in chosen]
    elif chosen or smoke: raise ValueError('Release cannot omit chapters or include fixtures')
    entries=config['frontmatter']+entries+config['backmatter']
    if smoke: entries=entries+[{'id':'layout-test','files':['publishing/samples/layout.md']}]
    docs=[]; seen=set(); names=set()
    for entry in entries:
        for index,path in enumerate(entry['files']):
            file=(ROOT/path).resolve()
            if file in seen: raise ValueError(f'Duplicate source: {path}')
            seen.add(file)
            text=file.read_text(); meta,body=strip_meta(text)
            if entry['id'].startswith('ch') and meta.get('chapter_id')!=entry['id']:
                raise ValueError(f'Chapter metadata mismatch: {path}')
            if edition=='release' and entry['id'].startswith('ch') and meta.get('status')!='verified':
                raise ValueError(f'Unverified release chapter: {path}')
            docid=entry['id'] if len(entry['files'])==1 else f"{entry['id']}-{index+1}"
            if docid in names: raise ValueError(f'Duplicate document ID: {docid}')
            names.add(docid)
            parsed=json.loads(run([PANDOC,'-f',FORMAT,'-t','json'],input=body,text=True,capture_output=True).stdout)
            docs.append(dict(id=docid,path=path,file=file,meta=meta,body=body,ast=parsed,
                sha256=hashlib.sha256(text.encode()).hexdigest()))
    return config,docs


def combined(config, docs, allow_missing_internal=False):
    document={'pandoc-api-version':docs[0]['ast']['pandoc-api-version'],'meta':{},'blocks':[]}
    targets={}
    for d in docs:
        targets[d['file']]={'':d['id']}
        for n in visit(d['ast']['blocks']):
            if n['t']=='Header':
                old=n['c'][1][0]; new=d['id']+'--'+old
                targets[d['file']][old]=new
                n['c'][1][0]=new
    for d in docs:
        for n in visit(d['ast']['blocks']):
            if n['t']=='Link':
                url=n['c'][2][0]
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',url): continue
                path,_,anchor=url.partition('#')
                target=(d['file'].parent/path).resolve() if path else d['file']
                if target not in targets or anchor not in targets[target]:
                    if allow_missing_internal and target not in targets:
                        try:
                            target.relative_to(ROOT)
                        except ValueError:
                            raise ValueError(f'Unresolved book link: {d["path"]}: {url}')
                        if target.is_file():
                            n['c'][2][0]='https://github.com/cluster1900/minimind_reader/blob/main/'+target.relative_to(ROOT).as_posix()
                            if anchor:
                                n['c'][2][0]+='#'+anchor
                            continue
                    raise ValueError(f'Unresolved book link: {d["path"]}: {url}')
                n['c'][2][0]='#'+targets[target][anchor]
        blocks=d['ast']['blocks']
        if not blocks or blocks[0]['t']!='Header' or blocks[0]['c'][0]!=1:
            raise ValueError(f'Each publication input must start with one H1: {d["path"]}')
        # Top-level H1 lets Pandoc split EPUB into chapters. The body wrapper
        # retains its stable ID for independent round-trip verification.
        document['blocks'].append(blocks[0])
        document['blocks'].append({'t':'Div','c':[attr(d['id'],['book-document']),blocks[1:]]})
    return document
