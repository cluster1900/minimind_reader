#!/usr/bin/env python3
"""Collect actual checks after a separate visual review; never invent a visual pass."""
import argparse,datetime,hashlib,json,re,subprocess,sys
from pathlib import Path
from booklib import ROOT,load,PANDOC
from check_book import check as check_book
from check_epub import check as check_epub
from check_pdf import check as check_pdf

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def require(ok,message):
    if not ok:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--visual-record',type=Path,required=True)
    parser.add_argument('--review-record',type=Path,required=True)
    parser.add_argument('--pdffonts',default='pdffonts',help='Poppler pdffonts executable')
    args=parser.parse_args();audit=(ROOT/args.audit_dir).resolve();stage=ROOT/'build/release'
    require((ROOT/args.review_record).is_file(),'Missing substantive review record')
    check_book('release',None)
    config,docs=load('release');inputs=read(stage/'inputs.json')
    require(inputs==[{k:d[k] for k in ['id','path','sha256']} for d in docs],'Candidate inputs are stale')
    outputs={f'minimind-source-guide.{ext}':sha(stage/f'minimind-source-guide.{ext}') for ext in ['epub','pdf']}
    check_epub(stage/'minimind-source-guide.epub',stage);check_pdf(stage/'minimind-source-guide.pdf',stage)
    pdf=read(stage/'pdf-text-check.json');visual=read(ROOT/args.visual_record)
    require(visual.get('passed') is True and visual.get('pages')==pdf['pages'] and
            visual.get('pdf_sha256')==outputs['minimind-source-guide.pdf'],'Missing or stale visual review')
    check_output=subprocess.check_output(['java','-jar',str(ROOT/'.tools/epubcheck-5.3.0/epubcheck.jar'),
                                         str(stage/'minimind-source-guide.epub')],stderr=subprocess.STDOUT,text=True)
    require('0 fatals / 0 errors / 0 warnings' in check_output,'EPUBCheck reports issues')
    (audit/'epubcheck.log').write_text(check_output)
    ex=read(audit/'experiments.json')
    require(ex['status']=='passed' and len(ex['experiments'])==len(config['chapters']),'Chapter experiment audit is incomplete')
    for name,expected in ex['input_sha256'].items():
        if name.startswith('experiments/'):
            require(sha(ROOT/name)==expected,'Experiment changed since it ran: '+name)
    require(all(e['exit_code']==0 for e in ex['experiments']),'A chapter experiment failed')
    tests=(audit/'tests.log').read_text();count=re.findall(r'Ran (\d+) tests?',tests)
    require(count and re.search(r'^OK\s*$',tests,re.M),'Test suite did not pass')
    readers={name:read(stage/name/'report.json') for name in ['epubjs','foliate']}
    require(all(r['status']=='passed' and r['sha256']==outputs['minimind-source-guide.epub'] for r in readers.values()),
            'Reader reports failed or refer to another EPUB')
    display=read(stage/'display/report.json')
    display_manifest=read(stage/'display/manifest.json')
    require(display_manifest['status']=='passed' and display_manifest['epub_sha256']==outputs['minimind-source-guide.epub'] and
            display_manifest['report_sha256']==sha(stage/'display/report.json'),'XHTML display report is stale or changed')
    require(len(display)==3*len(docs) and all(not d['broken'] and not d['overflow'] and d['contentWidth']<=d['width']+1 for d in display),
            'XHTML display checks are incomplete or failed')
    font_output=subprocess.check_output([args.pdffonts,str(stage/'minimind-source-guide.pdf')],text=True)
    (audit/'pdffonts.txt').write_text(font_output)
    fontlines=font_output.splitlines()[2:]
    require(fontlines and all(line.split()[-5]=='yes' for line in fontlines),'PDF has unembedded fonts')
    assets=read(stage/'assets.json')
    versions={name:read(ROOT/'node_modules'/name/'package.json')['version'] for name in
              ['mathjax-full','mermaid','@mermaid-js/mermaid-cli','epubjs','puppeteer']}
    versions.update(python=sys.version.split()[0],node=subprocess.check_output(['node','--version'],text=True).strip(),
                    pandoc=subprocess.check_output([PANDOC,'--version'],text=True).splitlines()[0],
                    tectonic=subprocess.check_output([ROOT/'.tools/tectonic','--version'],text=True).strip(),
                    epubcheck='5.3.0',foliate_js=readers['foliate']['revision'])
    paths=[*ROOT.glob('scripts/*.py'),*ROOT.glob('scripts/*.cjs'),*ROOT.glob('publishing/**/*.json'),
           *ROOT.glob('publishing/**/*.css'),*ROOT.glob('publishing/**/*.tex'),*ROOT.glob('publishing/*.html'),
           ROOT/'publishing/book.yaml',ROOT/'package-lock.json',*ROOT.glob('assets/fonts/*.otf'),
           ROOT/'assets/cover.png',ROOT/'assets/cover.pdf',ROOT/'meta/content-review.json',ROOT/'meta/chapter-source-map.json',ROOT/args.review_record]
    checks=dict(manuscript='passed',source_mapping='passed',cpu_experiments=len(ex['experiments']),
      publication_tests=int(count[-1]),epubcheck='passed: zero errors/warnings',epub_documents=len(docs),
      formula_resources=len(assets['math']),diagrams=len(assets['diagrams']),pdf_text_probes=pdf['text_probes'],
      pdf_visual_pages=pdf['pages'],epubjs_cases=len(readers['epubjs']['checks']),foliate_cases=len(readers['foliate']['checks']),
      xhtml_display_cases=len(display),embedded_font_resources=len(fontlines),pdf_chapter_order='passed')
    report=dict(title=config['title'],version=config['version'],status='passed',verified_at=datetime.datetime.now().astimezone().isoformat(),
      inputs=inputs,sources=ex['sources'],checks=checks,pdf_text_check=pdf,pdf_visual=visual,readers=readers,
      tool_versions=versions,tool_input_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))},
      output_sha256=outputs,audit_sha256={str(p.relative_to(ROOT)):sha(p) for p in
          [audit/'tests.log',audit/'experiments.json',audit/'build.log',audit/'epubcheck.log',audit/'pdffonts.txt',ROOT/args.visual_record]},
      limits=['Source/CPU/synthetic-control-flow scope; no full default-weight training, GPU benchmark or audio-quality claim',
              'No real microphone/speaker latency measurement or full screen-reader certification'])
    (ROOT/'meta/final-experiments.json').write_bytes((audit/'experiments.json').read_bytes())
    (ROOT/'meta/final-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
