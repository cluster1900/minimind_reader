"""Use real Dataset classes to verify role masks, one shift and empty targets."""
import hashlib,json,os,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/'minimind'
COMMIT='6fc918beb68a0d8c40452338df6319fe168014ba'
os.environ['HF_DATASETS_CACHE']=str(ROOT/'build/hf-datasets')
os.environ['HF_HUB_OFFLINE']='1'
def main():
    import torch
    from transformers import AutoTokenizer
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    sys.path.insert(0,str(SOURCE))
    from dataset.lm_dataset import PretrainDataset,SFTDataset
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
    torch.manual_seed(42);torch.set_num_threads(1)
    stage=ROOT/'build/experiments/ch06';stage.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(SOURCE/'model',local_files_only=True)
    (stage/'pretrain.jsonl').write_text(json.dumps({'text':'甲乙'},ensure_ascii=False)+'\n')
    pre=PretrainDataset(str(stage/'pretrain.jsonl'),tok,max_length=12);x,y=pre[0]
    assert torch.equal(x[y!=-100],y[y!=-100]);assert x[0]==tok.bos_token_id
    assert (y[x==tok.pad_token_id]==-100).all()
    conversation=[{'role':'system','content':'测试系统'}, {'role':'user','content':'问题一'},
        {'role':'assistant','content':'甲答案'}, {'role':'user','content':'问题二'}, {'role':'assistant','content':'乙答案'}]
    (stage/'sft.jsonl').write_text(json.dumps({'conversations':conversation},ensure_ascii=False)+'\n')
    sft=SFTDataset(str(stage/'sft.jsonl'),tok,max_length=128)
    random.seed(42);sx,sy=sft[0]
    supervised=tok.decode(sy[sy!=-100],skip_special_tokens=False)
    assert supervised=='甲答案<|im_end|>\n乙答案<|im_end|>\n',repr(supervised)
    cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        intermediate_size=64,vocab_size=len(tok),max_position_embeddings=128,flash_attn=False)
    model=MiniMindForCausalLM(cfg)
    output=model(sx[None],labels=sy[None])
    logits=output.logits.detach().clone().requires_grad_(True)
    targets=sy[1:][None]
    ce=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,len(tok)),targets.reshape(-1),ignore_index=-100)
    torch.testing.assert_close(ce,output.loss)
    ce.backward()
    grad=logits.grad[:,:-1]
    assert torch.equal(grad[targets==-100],torch.zeros_like(grad[targets==-100]))
    assert grad[targets!=-100].abs().sum()>0
    captured=[]
    def retain(module,inputs,hidden):hidden.retain_grad();captured.append(hidden)
    handle=model.model.embed_tokens.register_forward_hook(retain)
    model.zero_grad(set_to_none=True);model(sx[None],labels=sy[None]).loss.backward();handle.remove()
    assert sy[0]==-100 and captured[0].grad[0,0].abs().sum()>0
    short=SFTDataset(str(stage/'sft.jsonl'),tok,max_length=4)
    random.seed(42);tx,ty=short[0]
    assert (ty==-100).all()
    empty_loss=model(tx[None],labels=ty[None]).loss
    assert torch.isnan(empty_loss)
    report=dict(status='passed',source_commit=COMMIT,device='cpu',seed=42,pretrain_ids=x.tolist(),pretrain_labels=y.tolist(),
        supervised_text=supervised,supervised_count=int((sy[1:]!=-100).sum()),shift_loss=ce.item(),
        ignored_logits_gradient_zero=True,prompt_embedding_gradient_nonzero=True,empty_target_loss='nan',
        files={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in ['dataset/lm_dataset.py','model/model_minimind.py']},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=ROOT/'build/experiments/ch06-results.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
