"""Real freezing strategies, Parquet Dataset, collator, checkpoint contents, load_from routing, and V train_epoch boundaries."""
from _support import source,record,ROOT
import _support
_support.REVISIONS={**_support.REVISIONS,'minimind-v':'041d528e037a8a373c3c8aa7278a3ba3bd26b6dc'}
import importlib.util,io,json,os,random,types
from contextlib import nullcontext
from unittest.mock import patch
os.environ['HF_DATASETS_CACHE']=str(ROOT/'build/hf-datasets');os.environ['HF_HUB_OFFLINE']='1'
import torch

def classify_load_from(eval_vlm,load_from):
    """Execute real init_model and record which constructor branch ran."""
    args=types.SimpleNamespace(load_from=load_from,hidden_size=32,num_hidden_layers=2,use_moe=0,
        save_dir='out',weight='sft_vlm',device='cpu')
    dummy=types.SimpleNamespace(config=types.SimpleNamespace(),processor=object())
    dummy.load_state_dict=lambda *a,**k:None
    dummy.half=lambda:dummy
    dummy.eval=lambda:dummy
    dummy.to=lambda *a,**k:dummy
    seen={'native':False,'transformers':False}
    def MiniMind(*a,**k):
        seen['native']=True;return dummy
    MiniMind.get_vision_model=staticmethod(lambda *a,**k:(object(),object()))
    class AutoModel:
        @staticmethod
        def from_pretrained(*a,**k):
            seen['transformers']=True;return dummy
    class AutoTok:
        @staticmethod
        def from_pretrained(*a,**k):
            return object()
    with patch.object(eval_vlm,'MiniMindVLM',MiniMind),patch.object(eval_vlm,'AutoModelForCausalLM',AutoModel),\
         patch.object(eval_vlm,'AutoTokenizer',AutoTok),patch.object(eval_vlm.torch,'load',return_value={}),\
         patch.object(eval_vlm,'get_model_params'):
        eval_vlm.init_model(args)
    assert seen['native']^seen['transformers']
    return 'native' if seen['native'] else 'transformers'

def snapshot_of(model,optimizer,step):
    grad=model.w.grad
    return dict(step=step,weight=model.w.item(),grad=None if grad is None else grad.item(),updates=optimizer.count)

def load_trainer(name,src,stage,scaler):
    spec=importlib.util.spec_from_file_location(name,src/f'trainer/{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.scaler=scaler;m.autocast_ctx=nullcontext()
    m.vlm_config=types.SimpleNamespace(hidden_size=1,use_moe=False)
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
    loader=[(torch.tensor([0.]),torch.tensor([1.]),torch.tensor([0.])) for _ in range(n_batches)]
    if iters is None:iters=n_batches+start_step
    with patch.object(m,'vlm_checkpoint',side_effect=capture),patch.object(m,'Logger'):
        m.train_epoch(0,loader,iters,start_step)
    return snapshots

def main():
    src,files=source('minimind-v',['trainer/trainer_utils.py','dataset/lm_dataset.py','trainer/train_pretrain_vlm.py',
        'trainer/train_sft_vlm.py','eval_vlm.py'])
    eval_src=(src/'eval_vlm.py').read_text()
    assert "os.path.basename(os.path.normpath(args.load_from)) == 'model'" in eval_src
    assert "'model' in args.load_from" not in eval_src
    spec=importlib.util.spec_from_file_location('eval_vlm_ch15',src/'eval_vlm.py')
    eval_vlm=importlib.util.module_from_spec(spec);spec.loader.exec_module(eval_vlm)
    expected={
        'model':'native','../model':'native','./model':'native','model/':'native','/abs/path/model':'native',
        'foo_model_bar':'transformers','../foo_model_bar':'transformers','out/foo_model_bar':'transformers',
        'my_model':'transformers','model_ckpt':'transformers','../model_out':'transformers',
    }
    load_from_routing={path:classify_load_from(eval_vlm,path) for path in expected}
    assert load_from_routing==expected
    assert load_from_routing['foo_model_bar']=='transformers'
    from model.model_vlm import VLMConfig
    from trainer.trainer_utils import init_vlm_model,vlm_collate_fn,vlm_checkpoint
    from dataset.lm_dataset import VLMDataset
    from _tiny_vision import tiny_vision_loading
    from PIL import Image
    import pyarrow as pa,pyarrow.parquet as pq
    torch.manual_seed(42);torch.set_num_threads(1)
    stage=ROOT/'build/experiments/ch15';(stage/'vision').mkdir(parents=True,exist_ok=True)
    config=VLMConfig(hidden_size=32,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
        intermediate_size=64,vocab_size=6400,max_position_embeddings=128,flash_attn=False,image_hidden_size=16,image_token_len=4)
    groups={}
    for mode in [0,1,2]:
        with tiny_vision_loading():
            model,tok,processor=init_vlm_model(config,from_weight='none',tokenizer_path=str(src/'model'),vision_model_path=str(stage/'vision'),device='cpu',freeze_llm=mode)
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        assert all('vision_encoder' not in n for n in trainable)
        if mode==2:assert all(n.startswith('vision_proj.') for n in trainable)
        if mode==1:
            assert any('layers.0.' in n for n in trainable) and any('layers.3.' in n for n in trainable)
            assert not any('layers.1.' in n or 'layers.2.' in n or 'embed_tokens' in n for n in trainable)
        groups[str(mode)]=dict(parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),names=trainable)
    image=Image.new('RGB',(32,32),(120,80,30));buf=io.BytesIO();image.save(buf,format='PNG');data=buf.getvalue()
    rows=[]
    for count in [1,2]:
        conversations=[{'role':'system','content':'测试图片'},{'role':'user','content':'\n'.join(['<image>']*count)+'是什么？'},{'role':'assistant','content':'测试图。'}]
        rows.append({'conversations':json.dumps(conversations,ensure_ascii=False),'image_bytes':[data]*count})
    file=stage/'images.parquet';pq.write_table(pa.Table.from_pylist(rows),file)
    ds=VLMDataset(str(file),tok,preprocess=processor,max_length=64,image_token_len=4)
    random.seed(42);one=ds[0];two=ds[1]
    assert int((one[0]==12).sum())==4 and int((two[0]==12).sum())==8
    assert one[2]['pixel_values'].shape==(1,3,32,32) and two[2]['pixel_values'].shape==(2,3,32,32)
    batch=vlm_collate_fn([one,one]);assert batch[2]['pixel_values'].shape==(2,1,3,32,32)
    try:vlm_collate_fn([one,two])
    except RuntimeError:unequal_rejected=True
    else:raise AssertionError('Expected unequal image counts to fail stack')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    out=model(batch[0],labels=batch[1],pixel_values=batch[2])
    (out.loss+out.aux_loss).backward()
    projector_grad=sum(p.grad.abs().sum().item() for p in model.vision_proj.parameters() if p.grad is not None)
    assert projector_grad>0
    assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('vision_proj.'))
    opt.step()
    # V resume still writes the live full state_dict (includes encoder). It is not M/O's {k: v.cpu()} rematerialization.
    utils_src=(src/'trainer/trainer_utils.py').read_text()
    assert "state_dict = raw_model.state_dict()" in utils_src
    assert "'model': state_dict," in utils_src
    assert "state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}" not in utils_src
    proj_key=next(n for n,_ in model.named_parameters() if n.startswith('vision_proj.'))
    enc_key=next(n for n,_ in model.named_parameters() if n.startswith('vision_encoder.'))
    tiny=1e-8
    with torch.no_grad():dict(model.named_parameters())[proj_key].fill_(tiny)
    vlm_checkpoint(config,weight='test',model=model,optimizer=opt,save_dir=str(stage/'checkpoint'))
    infer=torch.load(stage/'checkpoint/test_32.pth',map_location='cpu',weights_only=True)
    resume=vlm_checkpoint(config,weight='test',save_dir=str(stage/'checkpoint'))
    assert set(resume['model'])==set(model.state_dict())
    assert enc_key in resume['model'] and enc_key not in infer
    assert proj_key in infer and not any(k.startswith('vision_encoder.') for k in infer)
    assert resume['model'][proj_key].dtype==torch.float32
    assert infer[proj_key].dtype==torch.float16
    assert abs(resume['model'][proj_key].reshape(-1)[0].item()-tiny)<1e-15
    assert infer[proj_key].reshape(-1)[0].item()==0.0
    class Scalar(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,input_ids,labels,pixel_values=None):
            return types.SimpleNamespace(loss=(self.w-labels).square().mean(),aux_loss=self.w.new_zeros(()))
    class CountingSGD(torch.optim.SGD):
        def __init__(self,params):super().__init__(params,lr=.1);self.count=0
        def step(self,*a,**kw):self.count+=1;return super().step(*a,**kw)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    loop={}
    for name in ['train_pretrain_vlm','train_sft_vlm']:
        m=load_trainer(name,src,stage,scaler)
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
        loop[name]=dict(snapshot=snaps[0],final_weight=m.model.w.item(),updates=m.optimizer.count)
    m=load_trainer('train_pretrain_vlm',src,stage,scaler)
    reset(m,2,100,'full_group',Scalar,CountingSGD,torch)
    snaps=run_epoch(m,4)
    assert m.optimizer.count==2
    assert len(snaps)==1 and snaps[0]['updates']==2 and snaps[0]['step']==4
    assert snaps[0]['grad'] is None
    assert abs(m.model.w.item()-snaps[0]['weight'])<1e-9
    loop['full_group']=dict(snapshot=snaps[0],final_weight=m.model.w.item(),updates=m.optimizer.count)
    reset(m,4,3,'mid_group',Scalar,CountingSGD,torch)
    snaps=run_epoch(m,5)
    assert len(snaps)==2 and snaps[0]['step']==3 and snaps[0]['updates']==0
    assert snaps[0]['grad'] is not None and abs(snaps[0]['weight'])<1e-9
    assert snaps[1]['step']==5 and snaps[1]['updates']==2 and snaps[1]['grad'] is None
    assert m.optimizer.count==2
    loop['mid_accumulation_save']=dict(snapshots=snaps,final_weight=m.model.w.item(),updates=m.optimizer.count)
    reset(m,2,100,'start_step',Scalar,CountingSGD,torch)
    snaps=run_epoch(m,2,iters=4,start_step=2)
    assert m.optimizer.count==1
    assert len(snaps)==1 and snaps[0]['step']==4 and snaps[0]['updates']==1
    assert snaps[0]['grad'] is None
    loop['start_step_resume']=dict(start_step=2,iters=4,loader_len=2,snapshot=snaps[0],updates=m.optimizer.count)
    reset(m,2,100,'empty',Scalar,CountingSGD,torch)
    snaps=run_epoch(m,0,iters=3,start_step=0)
    assert snaps==[] and m.optimizer.count==0 and abs(m.model.w.item())<1e-9
    loop['empty_loader']=dict(updates=0,saves=0,iters=3)
    record(15,__file__,files,dict(encoder='tiny random real SigLIP, only loading mocked',freeze_counts={k:v['parameters'] for k,v in groups.items()},
        freeze_trainable_names={k:v['names'] for k,v in groups.items()},batch_shape=list(batch[2]['pixel_values'].shape),unequal_image_count_collate_rejected=unequal_rejected,
        projector_gradient_with_frozen_llm=projector_grad,export_excludes_encoder=True,resume_contains_encoder=True,resume_is_full_state_dict=True,
        export_dtype='float16',resume_dtype='float32',resume_tiny=tiny,inference_tiny=float(infer[proj_key].reshape(-1)[0].item()),
        load_from_routing=load_from_routing,control_flow=loop),repo='minimind-v')
if __name__=='__main__':main()
