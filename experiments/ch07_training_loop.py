"""Exercise actual train_epoch control flow with a scalar model, no GPU."""
import hashlib,importlib.util,json,os,subprocess,sys,types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/'minimind'
COMMIT='6fc918beb68a0d8c40452338df6319fe168014ba'
os.environ['HF_DATASETS_CACHE']=str(ROOT/'build/hf-datasets')
def main():
    import torch
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    sys.path.insert(0,str(SOURCE))
    from trainer.trainer_utils import get_lr,lm_checkpoint,SkipBatchSampler
    class Scalar(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,input_ids,labels):return types.SimpleNamespace(loss=(self.w-labels).square().mean(),aux_loss=self.w.new_zeros(()))
    class CountingSGD(torch.optim.SGD):
        def __init__(self,params):super().__init__(params,lr=.1);self.count=0
        def step(self,*a,**kw):self.count+=1;return super().step(*a,**kw)
    stage=ROOT/'build/experiments/ch07';stage.mkdir(parents=True,exist_ok=True)
    results={}
    for name in ['train_pretrain','train_full_sft']:
        file=SOURCE/f'trainer/{name}.py';spec=importlib.util.spec_from_file_location(name,file)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        m.model=Scalar();m.optimizer=CountingSGD(m.model.parameters());m.scaler=torch.amp.GradScaler('cpu',enabled=False)
        m.autocast_ctx=nullcontext();m.lm_config=types.SimpleNamespace(hidden_size=1,use_moe=False)
        m.args=types.SimpleNamespace(device='cpu',epochs=1,learning_rate=.1,accumulation_steps=2,grad_clip=100.,
            log_interval=100,save_interval=100,save_dir=str(stage),save_weight=name)
        snapshots=[]
        def capture(*a,**kw):snapshots.append(dict(step=kw['step'],weight=m.model.w.item(),grad=m.model.w.grad.item(),updates=m.optimizer.count))
        loader=[(torch.tensor([0.]),torch.tensor([1.])) for _ in range(3)]
        with patch.object(m,'lm_checkpoint',side_effect=capture),patch.object(m,'Logger'):
            m.train_epoch(0,loader,3)
        assert m.optimizer.count==2
        assert snapshots[0]['updates']==1 and snapshots[0]['step']==3
        assert abs(snapshots[0]['weight']-.065)<1e-6
        assert abs(m.model.w.item()-.07435)<1e-6
        saved=torch.load(stage/f'{name}_1.pth')
        assert torch.equal(saved['w'],torch.tensor(snapshots[0]['weight']).half())
        results[name]=dict(snapshot=snapshots[0],final_weight=m.model.w.item(),updates=m.optimizer.count)
    assert list(SkipBatchSampler(list(range(5)),2,1))==[[2,3],[4]]
    assert abs(get_lr(0,10,.1)-.1)<1e-9 and abs(get_lr(10,10,.1)-.01)<1e-9
    model=Scalar();optim=torch.optim.AdamW(model.parameters(),lr=.01)
    model(torch.tensor([0.]),torch.tensor([1.])).loss.backward();optim.step();optim.zero_grad()
    config=types.SimpleNamespace(hidden_size=1,use_moe=False)
    lm_checkpoint(config,weight='roundtrip',model=model,optimizer=optim,epoch=1,step=2,save_dir=str(stage/'checkpoint'),scaler=m.scaler)
    state=lm_checkpoint(config,weight='roundtrip',save_dir=str(stage/'checkpoint'))
    assert state['model']['w'].dtype==torch.float16
    assert set(state)=={'model','optimizer','epoch','step','world_size','wandb_id','scaler'}
    assert 'exp_avg' in next(iter(state['optimizer']['state'].values()))
    report=dict(status='passed',source_commit=COMMIT,device='cpu',control_flow=results,
        checkpoint_fields=sorted(state),checkpoint_model_dtype=str(state['model']['w'].dtype),skip_batches=[[2,3],[4]],
        files={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in ['trainer/train_pretrain.py','trainer/train_full_sft.py','trainer/trainer_utils.py']},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=ROOT/'build/experiments/ch07-results.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
