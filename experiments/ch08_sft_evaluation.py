"""Small real-model SFT overfit check and token-weighted held-out NLL."""
import hashlib,json,math,os,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/'minimind'
COMMIT='6fc918beb68a0d8c40452338df6319fe168014ba'
os.environ['HF_DATASETS_CACHE']=str(ROOT/'build/hf-datasets');os.environ['HF_HUB_OFFLINE']='1'
def main():
    import torch
    from transformers import AutoTokenizer
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    sys.path.insert(0,str(SOURCE))
    from dataset.lm_dataset import SFTDataset
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
    torch.manual_seed(42);torch.set_num_threads(1)
    stage=ROOT/'build/experiments/ch08';stage.mkdir(parents=True,exist_ok=True)
    pairs=[('问题甲','答甲'),('问题乙','答乙'),('未见问题丙','答丙')]
    rows=[{'conversations':[{'role':'system','content':'这是合成测试。'},{'role':'user','content':q},{'role':'assistant','content':a}]} for q,a in pairs]
    corpus=stage/'sft.jsonl';corpus.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
    tok=AutoTokenizer.from_pretrained(SOURCE/'model',local_files_only=True);ds=SFTDataset(str(corpus),tok,max_length=64)
    fixed=[]
    for i in range(3):random.seed(42);fixed.append(ds[i])
    train=fixed[:2];held=fixed[2:]
    assert all((labels[1:]!=-100).any() for _,labels in fixed)
    cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        intermediate_size=64,vocab_size=len(tok),max_position_embeddings=128,flash_attn=False,dropout=0.)
    model=MiniMindForCausalLM(cfg)
    def evaluate(samples):
        model.eval();total=0.;count=0
        with torch.no_grad():
            for x,y in samples:
                logits=model(x[None]).logits[:,:-1];target=y[1:][None]
                total+=torch.nn.functional.cross_entropy(logits.reshape(-1,len(tok)),target.reshape(-1),ignore_index=-100,reduction='sum').item()
                count+=int((target!=-100).sum())
        assert count>0
        return dict(nll_sum=total,valid_tokens=count,nll=total/count,perplexity=math.exp(total/count))
    before_train=evaluate(train);before_held=evaluate(held)
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    x=torch.stack([p[0] for p in train]);y=torch.stack([p[1] for p in train])
    history=[]
    for step in range(80):
        model.train();opt.zero_grad(set_to_none=True)
        output=model(x,labels=y);loss=output.loss+output.aux_loss
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step in [0,19,39,79]:history.append(dict(step=step+1,loss=loss.item()))
    after_train=evaluate(train);after_held=evaluate(held)
    assert after_train['nll']<before_train['nll']*.2
    generated=[]
    model.eval()
    for row in rows:
        prompt=tok.apply_chat_template(row['conversations'][:-1],tokenize=False,add_generation_prompt=True,open_thinking=False)
        variants={}
        for name,text in [('default_template',prompt),('empty_think_removed',prompt.replace('<think>\n\n</think>\n\n',''))]:
            ids=tok(text,return_tensors='pt',add_special_tokens=False).input_ids
            output=model.generate(ids,max_new_tokens=12,temperature=1.,top_k=0,top_p=1.,do_sample=False,eos_token_id=tok.eos_token_id)
            variants[name]=tok.decode(output[0,ids.size(1):],skip_special_tokens=False)
        generated.append(dict(question=row['conversations'][1]['content'],expected=row['conversations'][-1]['content'],outputs=variants))
    # A counterexample to averaging per-batch means when counts differ.
    unequal_token_mean=(1.+27.)/10;unequal_batch_mean=(1.+3.)/2
    assert unequal_token_mean==2.8 and unequal_batch_mean==2.
    report=dict(status='passed',source_commit=COMMIT,device='cpu',seed=42,steps=80,learning_rate=.003,
        train_rows=2,held_out_rows=1,before_train=before_train,after_train=after_train,
        before_held_out=before_held,after_held_out=after_held,history=history,generated=generated,
        aggregation_counterexample=dict(token_mean=unequal_token_mean,batch_mean=unequal_batch_mean),
        corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in ['dataset/lm_dataset.py','model/model_minimind.py']})
    target=ROOT/'build/experiments/ch08-results.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
