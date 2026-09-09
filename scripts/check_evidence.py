"""Detect edits made after content review; hashes are not a proof of mathematical truth."""
import hashlib
import json
from booklib import ROOT,visit


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def inventory(docs):
    result=[]
    for d in docs:
        items=[];counts={'math':0,'diagram':0}
        for n in visit(d['ast']['blocks']):
            kind=None
            if n['t']=='Math':
                kind='math';payload=n['c'][0]['t']+'\n'+n['c'][1]
            elif n['t']=='CodeBlock' and 'mermaid' in n['c'][0][1]:
                kind='diagram';payload=n['c'][1]
            if kind:
                counts[kind]+=1
                items.append({'id':f'{d["id"]}-{kind}-{counts[kind]:03d}',
                              'kind':kind,'sha256':digest(payload)})
        result.append({'id':d['id'],'path':d['path'],'body_sha256':digest(d['body']),
                       'items':items})
    return result


def validate_review(current,sources,review):
    if review.get('status')!='passed' or not review.get('reviewed_at') or not review.get('record'):
        raise ValueError('Missing dated content review and review record')
    old={d['id']:d for d in review['documents']}
    if set(old)!={d['id'] for d in current}:raise ValueError('Reviewed document set changed')
    for doc in current:
        if doc!=old[doc['id']]:
            changed=[x['id'] for x in doc['items'] if x not in old[doc['id']]['items']]
            raise ValueError(f'Content needs renewed review: {doc["path"]}; changed formulas/diagrams: {changed}')
    if sources!=review['source_files']:
        raise ValueError('Cited source snapshot changed; review formulas and diagrams against it again')


def check_evidence(docs,sources):
    path=ROOT/'meta/content-review.json'
    if not path.is_file():raise ValueError('No content-review.json; complete content review before release')
    review=json.loads(path.read_text())
    validate_review(inventory(docs),sources,review)
    if not (ROOT/review['record']).is_file():raise ValueError('Content review record is missing')
    descriptions=ROOT/'publishing/math-descriptions.json'
    if hashlib.sha256(descriptions.read_bytes()).hexdigest()!=review.get('math_descriptions_sha256'):
        raise ValueError('Formula alternative descriptions changed; review them before release')
    print('Reviewed prose, formulas, diagrams and immutable source snapshots match')
