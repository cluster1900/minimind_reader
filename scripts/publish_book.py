#!/usr/bin/env python3
"""Publish a reviewed pair; a changed manuscript or candidate invalidates approval."""
import hashlib,json,os,shutil,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def source_bundle(destination):
 files=[]
 for folder in ['book','glossary','experiments','scripts','tests','publishing','meta','assets']:
  files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and
               '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='.DS_Store')
 files.extend(ROOT/name for name in ['README.md','AGENTS.md','requirements-book.lock',
             'requirements-omni.lock','package.json','package-lock.json','.gitignore'])
 files=sorted(set(files))
 with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,'minimind_reader/'+p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(destination) as z:
  if z.testzip() is not None:raise ValueError('Source bundle ZIP failed integrity check')
  for p in files:
   data=z.read('minimind_reader/'+p.relative_to(ROOT).as_posix())
   if hashlib.sha256(data).hexdigest()!=digest(p):raise ValueError('Source changed while packaging: '+str(p))
 return {'file':destination.name,'sha256':digest(destination),'files':len(files)}

def main():
 from check_book import check as check_manuscript
 check_manuscript('release',None)
 review=json.loads((ROOT/'meta/final-review.json').read_text())
 if review.get('status')!='passed':raise ValueError('Final review has not passed')
 stage=ROOT/'build/release'
 inputs=json.loads((stage/'inputs.json').read_text())
 if inputs!=review['inputs']:raise ValueError('Candidate input manifest differs from reviewed snapshot')
 for entry in inputs:
  if digest(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Manuscript changed: '+entry['path'])
 for name,expected in review['output_sha256'].items():
  if digest(stage/name)!=expected:raise ValueError('Candidate changed: '+name)
 for name,expected in review['tool_input_sha256'].items():
  if digest(ROOT/name)!=expected:raise ValueError('Publishing configuration changed: '+name)
 if not review['pdf_visual']['passed'] or review['pdf_visual']['pages']!=review['pdf_text_check']['pages']:
  raise ValueError('PDF visual review incomplete')
 for reader in ['epubjs','foliate']:
  report=json.loads((stage/reader/'report.json').read_text())
  if report['status']!='passed' or report['sha256']!=review['output_sha256']['minimind-source-guide.epub']:
   raise ValueError('Reader result stale or failed: '+reader)
 from check_epub import check as check_epub
 from check_pdf import check as check_pdf
 check_epub(stage/'minimind-source-guide.epub',stage)
 check_pdf(stage/'minimind-source-guide.pdf',stage)
 out=ROOT/'dist';out.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(dir=out,prefix='.release-') as temporary:
  pending=Path(temporary)
  for name,expected in review['output_sha256'].items():
   shutil.copyfile(stage/name,pending/name)
   if digest(pending/name)!=expected:raise ValueError('Staging copy mismatch')
  manifest={'title':review['title'],'version':review['version'],'verified_at':review['verified_at'],
   'inputs':inputs,'sources':review['sources'],'checks':review['checks'],
   'output_sha256':review['output_sha256'],'review_sha256':digest(ROOT/'meta/final-review.json')}
  bundle=pending/'minimind-reader-source.zip'
  manifest['source_bundle']=source_bundle(bundle)
  (pending/'release-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
  for name in review['output_sha256']:os.replace(pending/name,out/name)
  os.replace(bundle,out/bundle.name)
  os.replace(pending/'release-manifest.json',out/'release-manifest.json')
 print('Published reviewed EPUB/PDF pair to',out)
if __name__=='__main__':main()
