"""Exercise actual train_epoch control flow with a scalar model, no GPU."""
import hashlib,importlib.util,json,os,subprocess,sys,types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/'minimind'
COMMIT='233d00fa068ec2b8c0d4f07b703934d6391a5a73'
TRACKED=['trainer/train_pretrain.py','trainer/train_full_sft.py','trainer/trainer_utils.py']
os.environ['HF_DATASETS_CACHE']=str(ROOT/'build/hf-datasets')

def snapshot_of(model,optimizer,step):
    grad=model.w.grad
    return dict(step=step,weight=model.w.item(),grad=None if grad is None else grad.item(),updates=optimizer.count)

def load_trainer(name,stage,scaler):
    spec=importlib.util.spec_from_file_location(name,SOURCE/f'trainer/{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.scaler=scaler;m.autocast_ctx=nullcontext()
    m.lm_config=types.SimpleNamespace(hidden_size=1,use_moe=False)
    m.args=types.SimpleNamespace(device='cpu',epochs=1,learning_rate=.1,accumulation_steps=2,grad_clip=100.,
        log_interval=100,save_interval=100,save_dir=str(stage),save_weight=name)
    return m

def reset(m,accumulation_steps,save_interval,save_weight,Scalar,CountingSGD,torch):
    m.model=Scalar();m.optimizer=CountingSGD(m.model.parameters())
    m.scaler=torch.amp.GradScaler('cpu',enabled=False)
    m.args.accumulation_steps=accumulation_steps
    m.args.save_interval=save_interval
    m.args.save_weight=save_weight

def run_epoch(m,n_batches,iters=None,start_step=0):
    snapshots=[]
    def capture(*a,**kw):snapshots.append(snapshot_of(m.model,m.optimizer,kw['step']))
    import torch
    loader=[(torch.tensor([0.]),torch.tensor([1.])) for _ in range(n_batches)]
    if iters is None:iters=n_batches+start_step
    with patch.object(m,'lm_checkpoint',side_effect=capture),patch.object(m,'Logger'):
        m.train_epoch(0,loader,iters,start_step)
    return snapshots

def main():
    import torch
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*TRACKED],text=True).strip()
    sys.path.insert(0,str(SOURCE))
    from trainer.trainer_utils import get_lr,lm_checkpoint,SkipBatchSampler
    class Scalar(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,input_ids,labels):return types.SimpleNamespace(loss=(self.w-labels).square().mean(),aux_loss=self.w.new_zeros(()))
    class CountingSGD(torch.optim.SGD):
        def __init__(self,params):super().__init__(params,lr=.1);self.count=0
        def step(self,*a,**kw):self.count+=1;return super().step(*a,**kw)
    stage=ROOT/'build/experiments/ch07';stage.mkdir(parents=True,exist_ok=True)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    results={}
    # Tail group: 3 micro-batches, K=2. New baseline updates leftover grads before the final save.
    for name in ['train_pretrain','train_full_sft']:
        m=load_trainer(name,stage,scaler)
        reset(m,2,100,name,Scalar,CountingSGD,torch)
        snaps=run_epoch(m,3)
        assert m.optimizer.count==2
        assert len(snaps)==1 and snaps[0]['updates']==2 and snaps[0]['step']==3
        assert snaps[0]['grad'] is None
        assert abs(snaps[0]['weight']-.07435)<1e-6
        assert abs(m.model.w.item()-.07435)<1e-6
        saved=torch.load(stage/f'{name}_1.pth',map_location='cpu',weights_only=True)
        assert saved['w'].dtype==torch.float16
        assert torch.equal(saved['w'],m.model.w.detach().half().cpu())
        results[name]=dict(snapshot=snaps[0],final_weight=m.model.w.item(),updates=m.optimizer.count)
    m=load_trainer('train_pretrain',stage,scaler)
    # Full groups: 4 micro-batches, K=2. Last step is both an accumulation boundary and iters.
    reset(m,2,100,'full_group',Scalar,CountingSGD,torch)
    snaps=run_epoch(m,4)
    assert m.optimizer.count==2
    assert len(snaps)==1 and snaps[0]['updates']==2 and snaps[0]['step']==4
    assert snaps[0]['grad'] is None
    assert abs(m.model.w.item()-snaps[0]['weight'])<1e-9
    results['full_group']=dict(snapshot=snaps[0],final_weight=m.model.w.item(),updates=m.optimizer.count)
    # Periodic save inside an accumulation group: K=4, save_interval=3, 5 batches.
    reset(m,4,3,'mid_group',Scalar,CountingSGD,torch)
    snaps=run_epoch(m,5)
    assert len(snaps)==2 and snaps[0]['step']==3 and snaps[0]['updates']==0
    assert snaps[0]['grad'] is not None and abs(snaps[0]['weight'])<1e-9
    assert snaps[1]['step']==5 and snaps[1]['updates']==2 and snaps[1]['grad'] is None
    assert m.optimizer.count==2
    results['mid_accumulation_save']=dict(snapshots=snaps,final_weight=m.model.w.item(),updates=m.optimizer.count)
    # start_step resume continues 1-based step numbers; save uses the continued step, not a restarted 1.
    reset(m,2,100,'start_step',Scalar,CountingSGD,torch)
    snaps=run_epoch(m,2,iters=4,start_step=2)
    assert m.optimizer.count==1
    assert len(snaps)==1 and snaps[0]['step']==4 and snaps[0]['updates']==1
    assert snaps[0]['grad'] is None
    results['start_step_resume']=dict(start_step=2,iters=4,loader_len=2,snapshot=snaps[0],updates=m.optimizer.count)
    # Empty loader: no in-loop save/update, even if iters is nonzero.
    reset(m,2,100,'empty',Scalar,CountingSGD,torch)
    snaps=run_epoch(m,0,iters=3,start_step=0)
    assert snaps==[] and m.optimizer.count==0 and abs(m.model.w.item())<1e-9
    results['empty_loader']=dict(updates=0,saves=0,iters=3)
    assert list(SkipBatchSampler(list(range(5)),2,1))==[[2,3],[4]]
    assert abs(get_lr(0,10,.1)-.1)<1e-9 and abs(get_lr(10,10,.1)-.01)<1e-9
    # Resume dict stays FP32; inference *.pth is a separate FP16 file (043b22a / 233d00fa).
    model=Scalar();optim=torch.optim.AdamW(model.parameters(),lr=.01)
    model(torch.tensor([0.]),torch.tensor([1.])).loss.backward();optim.step();optim.zero_grad()
    tiny=1e-8
    model.w.data.fill_(tiny)
    kept=model.w.detach().cpu().clone()
    config=types.SimpleNamespace(hidden_size=1,use_moe=False)
    ckpt_dir=stage/'checkpoint'
    lm_checkpoint(config,weight='roundtrip',model=model,optimizer=optim,epoch=1,step=2,save_dir=str(ckpt_dir),scaler=m.scaler)
    state=lm_checkpoint(config,weight='roundtrip',save_dir=str(ckpt_dir))
    infer=torch.load(ckpt_dir/'roundtrip_1.pth',map_location='cpu',weights_only=True)
    assert state['model']['w'].dtype==torch.float32
    assert infer['w'].dtype==torch.float16
    assert torch.equal(state['model']['w'],kept)
    assert abs(state['model']['w'].item()-tiny)<1e-15
    assert infer['w'].item()==0.0
    assert set(state)=={'model','optimizer','epoch','step','world_size','wandb_id','scaler'}
    assert 'exp_avg' in next(iter(state['optimizer']['state'].values()))
    report=dict(status='passed',source_commit=COMMIT,device='cpu',torch=torch.__version__,control_flow=results,
        checkpoint_fields=sorted(state),checkpoint_model_dtype=str(state['model']['w'].dtype),
        checkpoint_inference_dtype=str(infer['w'].dtype),
        resume_weight=state['model']['w'].item(),inference_weight=infer['w'].item(),
        skip_batches=[[2,3],[4]],
        files={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in TRACKED},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    build=ROOT/'build/experiments/ch07-results.json';build.parent.mkdir(parents=True,exist_ok=True);build.write_text(text)
    print(text,end='')
if __name__=='__main__':main()
