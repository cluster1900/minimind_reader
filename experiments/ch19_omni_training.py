"""Real initialization/freezing/collation, source loss arithmetic, resume, eval routing, and train_epoch order."""
from _support import source,record,ROOT
import _support
_support.REVISIONS={**_support.REVISIONS,'minimind-o':'b5d1bc11ece8de17404cbc034c3e09718d0fecdf'}
import ast,os,types,math
from contextlib import nullcontext
from unittest.mock import patch
import torch
os.environ.setdefault('HF_DATASETS_CACHE',str(ROOT/'build/hf-datasets'))
os.environ.setdefault('HF_HUB_OFFLINE','1')

def classify_load_from(src,load_from):
    tree=ast.parse((src/'eval_omni.py').read_text())
    init_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='init_model')
    dummy=types.SimpleNamespace(audio_encoder=None,vision_encoder=None)
    dummy.load_state_dict=lambda *a,**k:None
    dummy.half=lambda:dummy
    dummy.eval=lambda:dummy
    dummy.to=lambda *a,**k:dummy
    seen={'native':False,'transformers':False}
    class MiniMind:
        def __new__(cls,*a,**k):
            seen['native']=True;return dummy
        @staticmethod
        def load_sensevoice(*a,**k):return None,None
        @staticmethod
        def load_vision(*a,**k):return None,None
    class AutoModel:
        @staticmethod
        def from_pretrained(*a,**k):
            seen['transformers']=True;return dummy
    class AutoTok:
        @staticmethod
        def from_pretrained(*a,**k):return object()
    class Mimi:
        @staticmethod
        def from_pretrained(*a,**k):
            m=types.SimpleNamespace();m.eval=lambda:m;return m
    ns=dict(os=os,torch=torch,MiniMindOmni=MiniMind,OmniConfig=lambda **k:None,
            AutoTokenizer=AutoTok,AutoModelForCausalLM=AutoModel,MimiModel=Mimi,
            log_model_params=lambda *a,**k:None)
    exec(compile(ast.Module(body=[init_node],type_ignores=[]),str(src/'eval_omni.py'),'exec'),ns)
    args=types.SimpleNamespace(load_from=load_from,use_moe=0,save_dir='out',weight='sft_omni',
        hidden_size=32,num_hidden_layers=2,device='cpu')
    with patch.object(torch,'load',return_value={}):
        ns['init_model'](args)
    assert seen['native']^seen['transformers']
    return 'native' if seen['native'] else 'transformers'

def snapshot_of(model,optimizer,step):
    grad=model.w.grad
    return dict(step=step,weight=model.w.item(),grad=None if grad is None else grad.item(),updates=optimizer.count)

def omni_batch():
    return (torch.zeros(1,2,dtype=torch.long),torch.full((1,2),-100),torch.full((1,8,2),-100),
            None,torch.zeros(1,dtype=torch.long),None,torch.zeros(1,4))

def run_omni_epoch(mod,n_batches,iters=None,start_step=0):
    snapshots=[]
    def capture(*a,**kw):snapshots.append(snapshot_of(mod.model,mod.optimizer,kw['step']))
    loader=[omni_batch() for _ in range(n_batches)]
    if iters is None:iters=n_batches+start_step
    with patch.object(mod,'omni_checkpoint',side_effect=capture),patch.object(mod,'Logger'):
        mod.train_epoch(0,loader,iters,start_step)
    return snapshots

def reset_omni_loop(mod,accumulation_steps,save_interval,save_weight,stage,Scalar,CountingSGD):
    mod.model=Scalar();mod.optimizer=CountingSGD(mod.model.parameters())
    mod.scaler=torch.amp.GradScaler('cpu',enabled=False)
    mod.autocast_ctx=nullcontext()
    mod.omni_config=types.SimpleNamespace(hidden_size=32,use_moe=False)
    mod.args=types.SimpleNamespace(device='cpu',epochs=1,learning_rate=.1,accumulation_steps=accumulation_steps,
        grad_clip=100.,log_interval=100,save_interval=save_interval,save_dir=str(stage),save_weight=save_weight)

def main():
    src,files=source('minimind-o',['trainer/train_sft_omni.py','trainer/trainer_utils.py','dataset/omni_dataset.py',
        'model/model_omni.py','trainer/train.sh','eval_omni.py'])
    from _omni_fixture import tiny_omni
    from model.model_omni import OmniConfig
    from model.model_minimind import MiniMindForCausalLM
    from trainer.trainer_utils import init_omni_model,omni_checkpoint
    from trainer.train_sft_omni import omni_collate_fn
    import trainer.train_sft_omni as omni_train
    from dataset.omni_dataset import OmniDataset
    torch.manual_seed(42);torch.set_num_threads(1)
    stage=ROOT/'build/experiments/ch19';stage.mkdir(parents=True,exist_ok=True)
    cfg=OmniConfig(hidden_size=32,num_hidden_layers=4,num_attention_heads=8,num_key_value_heads=4,
        talker_hidden_size=32,num_talker_hidden_layers=2,intermediate_size=128,max_position_embeddings=128,flash_attn=False)
    base=MiniMindForCausalLM(cfg);torch.save(base.state_dict(),stage/'llm_32.pth')
    frozen={}
    for mode in ['none','all','last1']:
        m,tok=init_omni_model(cfg,from_weight='llm',tokenizer_path=str(src/'model'),audio_encoder_path='',vision_model_path=None,save_dir=str(stage),device='cpu',freeze_backbone=mode)
        for i in range(2):
            for key,v in m.talker.layers[i].state_dict().items():torch.testing.assert_close(v,m.thinker.layers[i+2].state_dict()[key])
        if mode=='all':assert all(not p.requires_grad for p in m.thinker.parameters())
        if mode=='last1':
            assert all(not p.requires_grad for p in m.thinker.layers[0].parameters())
            assert all(p.requires_grad for p in m.thinker.layers[-1].parameters())
        frozen[mode]=sum(p.numel() for p in m.parameters() if p.requires_grad)
    tree=ast.parse((src/'trainer/train_sft_omni.py').read_text())
    main_node=next(n for n in tree.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test))
    mode_if=next(n for n in main_node.body if isinstance(n,ast.If) and ast.unparse(n.test)=="args.mode == 'audio_proj'")
    counts={}
    for mode,prefix in [('audio_proj','audio_proj.'),('vision_proj','vision_proj.')]:
        m.requires_grad_(True)
        exec(compile(ast.Module(body=[mode_if],type_ignores=[]),str(src/'trainer/train_sft_omni.py'),'exec'),dict(args=types.SimpleNamespace(mode=mode),model=m))
        assert all(n.startswith(prefix) for n,p in m.named_parameters() if p.requires_grad)
        counts[mode]=sum(p.numel() for p in m.parameters() if p.requires_grad)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_epoch')
    with_node=next(n for n in ast.walk(fn) if isinstance(n,ast.With) and any(isinstance(z,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss_fct' for t in z.targets) for z in n.body))
    begin=next(i for i,n in enumerate(with_node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss_fct' for t in n.targets))
    text=torch.zeros(1,3,3,requires_grad=True);heads=[torch.zeros(1,3,2112,requires_grad=True) for _ in range(8)]
    labels=torch.tensor([[0,-100,2]]);audio_labels=torch.full((1,8,3),-100);audio_labels[0,0,:2]=torch.tensor([1,2050])
    env=dict(torch=torch,nn=torch.nn,res=types.SimpleNamespace(logits=text,audio_logits=heads,aux_loss=torch.tensor(0.)),labels=labels,audio_labels=audio_labels,args=types.SimpleNamespace(accumulation_steps=1))
    exec(compile(ast.Module(body=with_node.body[begin:],type_ignores=[]),str(src/'trainer/train_sft_omni.py'),'exec'),env)
    expected_audio=math.log(2112)*11/2/8
    torch.testing.assert_close(env['text_loss'],torch.tensor(math.log(3.)))
    torch.testing.assert_close(env['audio_loss'],torch.tensor(expected_audio))
    env['loss'].backward();ratio=(heads[0].grad[0,1].norm()/heads[0].grad[0,0].norm()).item();assert abs(ratio-10)<1e-4
    empty_env={**env,'labels':torch.full_like(labels,-100),'audio_labels':torch.full_like(audio_labels,-100)}
    exec(compile(ast.Module(body=with_node.body[begin:],type_ignores=[]),str(src/'trainer/train_sft_omni.py'),'exec'),empty_env)
    assert empty_env['text_loss'].item()==0. and empty_env['audio_loss'].item()==0.
    sample=lambda audio:(torch.zeros(9,8,dtype=torch.long),torch.full((8,),-100),torch.full((8,8),-100),audio,0,None,torch.zeros(192))
    good=omni_collate_fn([sample(torch.ones(1,3,560)),sample(torch.zeros(1,1,560))])
    bad=omni_collate_fn([sample(torch.ones(1,3,560)),sample(None)])
    assert good[3].shape==(2,3,560) and bad[3].shape[0]==1 and bad[4].shape[0]==2
    ds=object.__new__(OmniDataset);ds.scheduled_sampling_prob=1.;ds.audio_vocab_size=2112;ds.text_vocab_size=6400;ds.image_token_id=12
    inputs=torch.full((9,3),2049,dtype=torch.long);inputs[8]=torch.tensor([12,16,99])
    al=torch.full((8,3),-100);al[0,[0,2]]=1
    torch.manual_seed(42);changed=ds.apply_scheduled_sampling(inputs,al,torch.ones(3,dtype=torch.long))
    assert changed[8,0]==12 and changed[8,1]!=16
    class EncoderTagged(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.keep=torch.nn.Parameter(torch.tensor(1e-8))
            self.audio_encoder=torch.nn.Linear(2,2,bias=False)
            self.vision_encoder=torch.nn.Linear(2,2,bias=False)
    tagged=EncoderTagged()
    optimizer=torch.optim.AdamW(tagged.parameters(),lr=.01)
    tagged.keep.data.fill_(1e-8)
    kept=tagged.keep.detach().cpu().clone()
    ckpt_dir=stage/'checkpoint'
    omni_checkpoint(cfg,weight='inspect',model=tagged,optimizer=optimizer,save_dir=str(ckpt_dir))
    checkpoint=omni_checkpoint(cfg,weight='inspect',save_dir=str(ckpt_dir))
    infer=torch.load(ckpt_dir/f'inspect_{cfg.hidden_size}.pth',map_location='cpu',weights_only=True)
    assert all(not key.startswith(('audio_encoder.','vision_encoder.')) for key in checkpoint['model'])
    assert all(not key.startswith(('audio_encoder.','vision_encoder.')) for key in infer)
    assert any(key.startswith('audio_encoder.') for key in tagged.state_dict())
    assert any(key.startswith('vision_encoder.') for key in tagged.state_dict())
    assert checkpoint['model']['keep'].dtype==torch.float32
    assert infer['keep'].dtype==torch.float16
    assert torch.equal(checkpoint['model']['keep'],kept)
    assert abs(checkpoint['model']['keep'].item()-1e-8)<1e-15
    assert infer['keep'].item()==0.0
    eval_src=(src/'eval_omni.py').read_text()
    assert "os.path.basename(os.path.normpath(args.load_from)) == 'model'" in eval_src
    assert "'model' in args.load_from" not in eval_src
    expected={
        'model':'native','../model':'native','./model':'native','model/':'native','/abs/path/model':'native',
        'foo_model_bar':'transformers','../foo_model_bar':'transformers','out/foo_model_bar':'transformers',
        'my_model':'transformers','model_ckpt':'transformers','../model_out':'transformers',
    }
    load_from_routing={path:classify_load_from(src,path) for path in expected}
    assert load_from_routing==expected
    assert load_from_routing['foo_model_bar']=='transformers'
    class Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,input_ids,audio_inputs=None,audio_lens=None,pixel_values=None,spk_emb=None):
            B,S=input_ids.shape
            return types.SimpleNamespace(logits=self.w.new_zeros(B,S,3),
                audio_logits=[self.w.new_zeros(B,S,4) for _ in range(8)],aux_loss=(self.w-1).square())
    class CountingSGD(torch.optim.SGD):
        def __init__(self,params):super().__init__(params,lr=.1);self.count=0
        def step(self,*a,**kw):self.count+=1;return super().step(*a,**kw)
    loop_stage=stage/'loop';loop_stage.mkdir(exist_ok=True)
    reset_omni_loop(omni_train,2,100,'tail',loop_stage,Scalar,CountingSGD)
    snaps=run_omni_epoch(omni_train,3)
    assert omni_train.optimizer.count==2
    assert len(snaps)==1 and snaps[0]['updates']==2 and snaps[0]['step']==3
    assert snaps[0]['grad'] is None
    assert abs(snaps[0]['weight']-.07435)<1e-6
    assert abs(omni_train.model.w.item()-.07435)<1e-6
    tail=dict(snapshot=snaps[0],final_weight=omni_train.model.w.item(),updates=omni_train.optimizer.count)
    reset_omni_loop(omni_train,2,100,'full_group',loop_stage,Scalar,CountingSGD)
    snaps=run_omni_epoch(omni_train,4)
    assert omni_train.optimizer.count==2
    assert len(snaps)==1 and snaps[0]['updates']==2 and snaps[0]['step']==4
    assert snaps[0]['grad'] is None
    assert abs(omni_train.model.w.item()-snaps[0]['weight'])<1e-9
    full=dict(snapshot=snaps[0],final_weight=omni_train.model.w.item(),updates=omni_train.optimizer.count)
    reset_omni_loop(omni_train,4,3,'mid_group',loop_stage,Scalar,CountingSGD)
    snaps=run_omni_epoch(omni_train,5)
    assert len(snaps)==2 and snaps[0]['step']==3 and snaps[0]['updates']==0
    assert snaps[0]['grad'] is not None and abs(snaps[0]['weight'])<1e-9
    assert snaps[1]['step']==5 and snaps[1]['updates']==2 and snaps[1]['grad'] is None
    assert omni_train.optimizer.count==2
    mid=dict(snapshots=snaps,final_weight=omni_train.model.w.item(),updates=omni_train.optimizer.count)
    reset_omni_loop(omni_train,2,100,'start_step',loop_stage,Scalar,CountingSGD)
    snaps=run_omni_epoch(omni_train,2,iters=4,start_step=2)
    assert omni_train.optimizer.count==1
    assert len(snaps)==1 and snaps[0]['step']==4 and snaps[0]['updates']==1
    assert snaps[0]['grad'] is None
    start=dict(start_step=2,iters=4,loader_len=2,snapshot=snaps[0],updates=omni_train.optimizer.count)
    reset_omni_loop(omni_train,2,100,'empty',loop_stage,Scalar,CountingSGD)
    snaps=run_omni_epoch(omni_train,0,iters=3,start_step=0)
    assert snaps==[] and omni_train.optimizer.count==0 and abs(omni_train.model.w.item())<1e-9
    empty=dict(updates=0,saves=0,iters=3)
    record(19,__file__,files,dict(talker_copied_from_last_thinker_layers=True,freeze_backbone_counts=frozen,
        empty_text_loss=empty_env['text_loss'].item(),empty_audio_loss=float(empty_env['audio_loss']),
        projector_modes=counts,text_loss=env['text_loss'].item(),audio_loss=env['audio_loss'].item(),stop_gradient_ratio=ratio,
        valid_audio_batch=list(good[3].shape),none_row_mismatch=[bad[3].shape[0],bad[4].shape[0]],
        scheduled_sampling_is_random_token_corruption=True,image_marker_protected=True,audio_marker_not_explicitly_protected=True,
        resume_model_dtype='float32',resume_excludes_external_encoders=True,inference_dtype='float16',
        resume_weight=checkpoint['model']['keep'].item(),inference_weight=infer['keep'].item(),
        load_from_routing=load_from_routing,
        control_flow=dict(tail_group=tail,full_group=full,mid_accumulation_save=mid,start_step_resume=start,empty_loader=empty)),
        repo='minimind-o')
if __name__=='__main__':main()
