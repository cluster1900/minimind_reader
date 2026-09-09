"""Verify that media conversion preserves source content before packaging."""
import base64,hashlib,json
from pathlib import Path
from xml.etree import ElementTree

def semantic_tokens(value):
    if isinstance(value,list):
        for child in value:yield from semantic_tokens(child)
    elif isinstance(value,dict):
        kind=value.get('t');c=value.get('c')
        if kind=='Str':yield ['text',c];return
        if kind in ('Space','SoftBreak','LineBreak'):yield ['space'];return
        if kind=='Code':yield ['inline-code',c[1]];return
        if kind=='CodeBlock':
            yield ['diagram' if 'mermaid' in c[0][1] else 'code-block',c[1]];return
        if kind=='Math':yield ['math',c[0]['t'],c[1]];return
        if kind=='Image':
            attrs=dict(c[0][2]);classes=c[0][1]
            if 'data-tex-b64' in attrs:
                yield ['math','DisplayMath' if attrs['data-display']=='true' else 'InlineMath',base64.b64decode(attrs['data-tex-b64']).decode()]
            elif 'diagram' in classes:
                svg=ElementTree.fromstring(Path(c[2][0]).read_bytes())
                yield ['diagram',base64.b64decode(svg.attrib['data-mermaid-b64']).decode()]
            else:yield ['image',c[2][0]]
            return
        if kind=='Para' and len(c)==1 and c[0].get('t')=='Image' and 'diagram' in c[0]['c'][0][1]:
            yield from semantic_tokens(c[0]);return
        containers={'Para','Plain','Header','Table','BulletList','OrderedList','Note','Emph','Strong'}
        if kind in containers:yield ['begin',kind,c[0] if kind=='Header' else None]
        if kind=='Link':yield ['link-target',c[2][0]]
        for child in value.values():
            if isinstance(child,(dict,list)):yield from semantic_tokens(child)
        if kind in containers:yield ['end',kind]

def verify(source,converted):
    a=list(semantic_tokens(source['blocks']));b=list(semantic_tokens(converted['blocks']))
    if a!=b:
        mismatch=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        raise ValueError(f'Content transformation mismatch at token {mismatch}: {a[mismatch:mismatch+2]} != {b[mismatch:mismatch+2]}')
    return {'semantic_tokens':len(a),'sha256':hashlib.sha256(json.dumps(a,ensure_ascii=False).encode()).hexdigest(),'status':'passed'}
