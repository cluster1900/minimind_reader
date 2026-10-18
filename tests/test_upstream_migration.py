"""Focused migration checks; standard library only, not PyTorch integration.
Run: python tests/test_upstream_migration.py --source-root /path/to/fixed/sources
Sources must be the three candidate commits documented in the migration report.
"""
import argparse, ast, contextlib, hashlib, json, math, operator, random, tempfile, time
from pathlib import Path
from types import SimpleNamespace as NS


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def run(root):
    passed = []
    manifest=Path(__file__).resolve().parents[1]/'meta/upstream-migration-sources-2026-10-18.json'
    for record in json.loads(manifest.read_text()):
        data=(root/record['repository']/record['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==record['sha256'], record
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==record['blob_sha'], record
    passed.append('28 source files match fixed-candidate SHA-256 and Git blob manifest')
    M = root / 'minimind'
    ns = extract(M/'trainer/train_tokenizer.py', ['get_texts'], {'json':json, 'MAX_LINES':0})
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'texts.jsonl'
        p.write_text('\n'.join(['{bad', json.dumps({'text':' '}), json.dumps({'text':'one'}), json.dumps({'conversations':[{'content':'two'},{'content':'three'}]}), json.dumps({'text':'four'})]))
        assert list(ns['get_texts'](p))==['one','two\nthree','four']
        assert list(ns['get_texts'](p,1))==['one']
        assert list(ns['get_texts'](p,2))==['one','two\nthree']
        p.write_text('{bad\n{"text":" "}\n')
        try: list(ns['get_texts'](p)); raise AssertionError('empty corpus accepted')
        except ValueError: pass
    passed.append('get_texts: text/conversations, malformed/blank records, yielded-record cap, empty-corpus rejection')
    ns=extract(M/'dataset/lm_dataset.py',['post_processing_chat'],{'random':random})
    text='<think>\n\n</think>\n\nanswer'
    assert ns['post_processing_chat'](text,remove_empty_think=True)=='answer'
    assert ns['post_processing_chat'](text,remove_empty_think=False)==text
    tree=ast.parse((M/'dataset/lm_dataset.py').read_text())
    dpo=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DPODataset')
    calls=[n for n in ast.walk(dpo) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='post_processing_chat']
    assert len(calls)==2 and all(any(k.arg=='remove_empty_think' and isinstance(k.value,ast.Name) and k.value.id=='remove_empty_think' for k in n.keywords) for n in calls)
    passed.append('DPO: shared explicit removal decision at both call sites; helper true/false behavior')
    ns=extract(M/'trainer/trainer_utils.py',['safe_math_eval'],{'ast':ast,'operator':operator,'math':math})
    evaluate=ns['safe_math_eval']
    for expr,value in [('1+2',3),('sqrt(4)',2),('math.sqrt(4)',2),('（2+3）²',25),('2^3',8)]:assert evaluate(expr)==value
    for expr in ['', '1'*513, '__import__("os")', '(1).__class__', '10**99999', '9**9**9']:
        try:evaluate(expr);raise AssertionError(expr)
        except (ValueError,OverflowError):pass
    passed.append('safe_math_eval: permitted arithmetic and aliases; import/attribute/empty/oversize/huge-power rejection (not a sandbox proof)')
    # Execute the original train_epoch AST with a scalar derivative and I/O doubles.
    for trainer in ['train_pretrain','train_full_sft']:
        events=[]
        class Scalar:
            def __init__(self,value,grad=0):self.value=value;self.grad=grad
            def to(self,*a):return self
            def half(self):return self
            def cpu(self):return self
            def item(self):return self.value
            def __add__(self,other):return Scalar(self.value+other.value,self.grad+other.grad)
            def __truediv__(self,k):return Scalar(self.value/k,self.grad/k)
            def backward(self):model.grad+=self.grad;events.append('backward')
        class Model:
            def __init__(self):self.w=0.;self.grad=0.
            def __call__(self,inputs,labels):return NS(loss=Scalar((self.w-labels.value)**2,2*(self.w-labels.value)),aux_loss=Scalar(0))
            def parameters(self):return [self]
            def eval(self):pass
            def train(self):pass
            def state_dict(self):return {'w':Scalar(self.w)}
        model=Model()
        class Optimizer:
            param_groups=[{'lr':.1}]
            def step(self):model.w-=self.param_groups[0]['lr']*model.grad;events.append('step')
            def zero_grad(self,**kw):model.grad=0.;events.append('zero')
        optimizer=Optimizer()
        scaler=NS(scale=lambda x:x,unscale_=lambda o:None,step=lambda o:o.step(),update=lambda:None)
        snaps=[]
        def save(*a,**kw):events.append('save');snaps.append((model.w,model.grad,events.count('step')))
        torch=NS(nn=NS(utils=NS(clip_grad_norm_=lambda *a:None)),save=lambda *a:None)
        args=NS(device='cpu',epochs=1,learning_rate=.1,accumulation_steps=2,grad_clip=100,log_interval=100,save_interval=100,save_dir='unused',save_weight='unused')
        ns={'time':time,'model':model,'optimizer':optimizer,'scaler':scaler,'autocast_ctx':contextlib.nullcontext(),'args':args,'torch':torch,'Logger':lambda *a:None,'lm_config':NS(hidden_size=1,use_moe=False),'DistributedDataParallel':type('NeverMatch',(),{}),'is_main_process':lambda:True,'lm_checkpoint':save,'math':math}
        extract(M/'trainer/trainer_utils.py',['get_lr'],ns)
        extract(M/f'trainer/{trainer}.py',['train_epoch'],ns)
        ns['train_epoch'](0,[(Scalar(0),Scalar(1)) for _ in range(3)],3)
        assert snaps==[(model.w,0.,2)] and abs(model.w-.07435)<1e-10,(trainer,snaps,model.w)
        assert events[-3:]==['step','zero','save']
    passed.append('original pretrain/full_sft train_epoch AST: 3 batches, K=2, two updates before final save, zero residual gradient; scalar doubles, not torch/AMP')
    files=[M/f'trainer/train_{x}.py' for x in ['pretrain','full_sft','lora','distillation','dpo','grpo','agent']]
    files += [root/'minimind-v'/f'trainer/train_{x}_vlm.py' for x in ['pretrain','sft']]
    files += [root/'minimind-o'/'trainer/train_sft_omni.py']
    for p in files:
        tree=ast.parse(p.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.endswith('train_epoch')]
        assert len(functions)==1,p
        f=functions[0]
        assert any(isinstance(n,ast.If) and ast.unparse(n.test)=='step % args.accumulation_steps == 0 or step == iters' for n in ast.walk(f)),p
        assert not any(isinstance(n,ast.If) for n in f.body[f.body.index(next(n for n in f.body if isinstance(n,ast.For)))+1:]),p
    passed.append('10 trainer ASTs: in-loop final-step update condition, no post-loop conditional flush')
    model_source=(M/'model/model_minimind.py').read_text()
    assert '(xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)' in model_source
    assert 'self.gate(x_flat.detach())' in model_source and 'top1 - top1.detach() + 1.0' in model_source
    cores=[(root/repo/'model/model_minimind.py').read_bytes() for repo in ['minimind','minimind-v','minimind-o']]
    assert cores[0]==cores[1]==cores[2]
    passed.append('three model cores byte-identical; FP32 conversion placement and detached-input straight-through branch present (static only)')
    for script in ['train_ppo.py','train_grpo.py','train_agent.py']:
        s=(M/'trainer'/script).read_text();assert 'broadcast_buffers=False' in s
        assert 'model_unwrapped(' not in s and 'actor_unwrapped(' not in s and 'critic_unwrapped(' not in s
    s=(M/'trainer/train_agent.py').read_text()
    assert 'signal.alarm' not in s and 'prompt_ids + response_ids' in s and 'partition(marker)' in s and 'rollout_result.completion_mask[0].sum()' in s
    passed.append('RL/Agent static call-site guards: DDP wrappers, no SIGALRM, preserved token history, marker and completion mask')
    return {'status':'passed','scope':'Focused standard-library AST, helper and scalar-control-flow checks only; no torch/transformers, GPU, multi-process, browser or publication acceptance','checks':passed}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(run(args.source_root),ensure_ascii=False,indent=2))
