"""Real initialization/freezing/collation and exact source loss arithmetic."""
from _support import source,record,ROOT
import ast,json,math,types,random
from unittest.mock import patch
import torch
def main():
    src,files=source('minimind-o',['trainer/train_sft_omni.py','trainer/trainer_utils.py','dataset/omni_dataset.py','model/model_omni.py','trainer/train.sh'])
    from _omni_fixture import tiny_omni
    from model.model_omni import OmniConfig
    from model.model_minimind import MiniMindForCausalLM
    from trainer.trainer_utils import init_omni_model,omni_checkpoint
    from trainer.train_sft_omni import omni_collate_fn
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
    optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])
    omni_checkpoint(cfg,weight='inspect',model=m,optimizer=optimizer,save_dir=str(stage/'checkpoint'))
    checkpoint=omni_checkpoint(cfg,weight='inspect',save_dir=str(stage/'checkpoint'))
    assert all(not key.startswith(('audio_encoder.','vision_encoder.')) for key in checkpoint['model'])
    assert next(iter(checkpoint['model'].values())).dtype==torch.float16
    record(19,__file__,files,dict(talker_copied_from_last_thinker_layers=True,freeze_backbone_counts=frozen,
        empty_text_loss=empty_env['text_loss'].item(),empty_audio_loss=float(empty_env['audio_loss']),
        projector_modes=counts,text_loss=env['text_loss'].item(),audio_loss=env['audio_loss'].item(),stop_gradient_ratio=ratio,
        valid_audio_batch=list(good[3].shape),none_row_mismatch=[bad[3].shape[0],bad[4].shape[0]],
        scheduled_sampling_is_random_token_corruption=True,image_marker_protected=True,audio_marker_not_explicitly_protected=True,
        resume_model_dtype='float16',resume_excludes_external_encoders=True),repo='minimind-o')
if __name__=='__main__':main()
