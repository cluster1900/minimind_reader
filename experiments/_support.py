"""Book experiment bookkeeping; never silently change upstream revisions."""
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REVISIONS={'minimind':'6fc918beb68a0d8c40452338df6319fe168014ba','minimind-v':'740d467ece78a0b7d2d976fcb424472095d4a688','minimind-o':'ccab6d98520ac5ac5d7583a3fa21a65105137e28'}
def source(repo,paths):
    p=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/repo
    assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==REVISIONS[repo]
    assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain','--',*paths],text=True).strip()
    sys.path.insert(0,str(p))
    return p,{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in paths}
def record(ch,script,files,results,repo='minimind'):
    import torch
    report=dict(status='passed',source_commit=REVISIONS[repo],source_repository=repo,files=files,
        script_sha256=hashlib.sha256(Path(script).read_bytes()).hexdigest(),support_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version,torch=torch.__version__,platform=platform.platform(),device='cpu',seed=42,results=results)
    p=ROOT/f'build/experiments/ch{ch:02d}-results.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
