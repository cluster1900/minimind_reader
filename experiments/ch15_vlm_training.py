"""Real freezing strategies, Parquet Dataset, collator and checkpoint contents."""
from _support import source,record,ROOT
import io,json,os,random
os.environ['HF_DATASETS_CACHE']=str(ROOT/'build/hf-datasets');os.environ['HF_HUB_OFFLINE']='1'
import torch
def main():
    src,files=source('minimind-v',['trainer/trainer_utils.py','dataset/lm_dataset.py','trainer/train_pretrain_vlm.py','trainer/train_sft_vlm.py'])
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
    vlm_checkpoint(config,weight='test',model=model,optimizer=opt,save_dir=str(stage/'checkpoint'))
    infer=torch.load(stage/'checkpoint/test_32.pth');resume=vlm_checkpoint(config,weight='test',save_dir=str(stage/'checkpoint'))
    assert not any(k.startswith('vision_encoder.') for k in infer)
    assert any(k.startswith('vision_encoder.') for k in resume['model'])
    assert next(iter(infer.values())).dtype==torch.float16
    assert next(iter(resume['model'].values())).dtype==torch.float32
    record(15,__file__,files,dict(encoder='tiny random real SigLIP, only loading mocked',freeze_counts={k:v['parameters'] for k,v in groups.items()},
        freeze_trainable_names={k:v['names'] for k,v in groups.items()},batch_shape=list(batch[2]['pixel_values'].shape),unequal_image_count_collate_rejected=unequal_rejected,
        projector_gradient_with_frozen_llm=projector_grad,export_excludes_encoder=True,resume_contains_encoder=True,export_dtype='float16',resume_dtype='float32'),repo='minimind-v')
if __name__=='__main__':main()
