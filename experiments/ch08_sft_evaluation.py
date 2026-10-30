"""Small real-model SFT overfit check, token-weighted held-out NLL, and eval_llm load_from routing."""
import hashlib,importlib.util,json,math,os,random,subprocess,sys,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/'minimind'
COMMIT='233d00fa068ec2b8c0d4f07b703934d6391a5a73'
TRACKED=['dataset/lm_dataset.py','model/model_minimind.py','eval_llm.py']
os.environ['HF_DATASETS_CACHE']=str(ROOT/'build/hf-datasets');os.environ['HF_HUB_OFFLINE']='1'

def load_eval_llm():
    spec=importlib.util.spec_from_file_location('eval_llm_ch08',SOURCE/'eval_llm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def classify_load_from(eval_llm,load_from):
    """Execute real init_model and record which constructor branch ran."""
    args=types.SimpleNamespace(load_from=load_from,hidden_size=32,num_hidden_layers=2,use_moe=0,
        inference_rope_scaling=False,save_dir='out',weight='full_sft',lora_weight='None',lora_rank=16,device='cpu')
    dummy=types.SimpleNamespace(config=types.SimpleNamespace())
    dummy.load_state_dict=lambda *a,**k:None
    dummy.half=lambda:dummy
    dummy.eval=lambda:dummy
    dummy.to=lambda *a,**k:dummy
    seen={'native':False,'transformers':False}
    def MiniMind(*a,**k):
        seen['native']=True;return dummy
    class AutoModel:
        @staticmethod
        def from_pretrained(*a,**k):
            seen['transformers']=True;return dummy
    class AutoTok:
        @staticmethod
        def from_pretrained(*a,**k):
            return object()
    with patch.object(eval_llm,'MiniMindForCausalLM',MiniMind),patch.object(eval_llm,'AutoModelForCausalLM',AutoModel),\
         patch.object(eval_llm,'AutoTokenizer',AutoTok),patch.object(eval_llm.torch,'load',return_value={}),\
         patch.object(eval_llm,'get_model_params'):
        eval_llm.init_model(args)
    assert seen['native']^seen['transformers']
    return 'native' if seen['native'] else 'transformers'

def main():
    import torch
    from transformers import AutoTokenizer
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*TRACKED],text=True).strip()
    sys.path.insert(0,str(SOURCE))
    eval_src=(SOURCE/'eval_llm.py').read_text()
    assert "os.path.basename(os.path.normpath(args.load_from)) == 'model'" in eval_src
    assert "'model' in args.load_from" not in eval_src
    eval_llm=load_eval_llm()
    # New baseline: only a final path segment exactly equal to 'model' uses native weights.
    # A substring such as foo_model_bar used to match `'model' in path` and is now transformers.
    expected={
        'model':'native','../model':'native','./model':'native','model/':'native','/abs/path/model':'native',
        'foo_model_bar':'transformers','../foo_model_bar':'transformers','out/foo_model_bar':'transformers',
        'my_model':'transformers','model_ckpt':'transformers','../model_out':'transformers',
    }
    load_from_routing={path:classify_load_from(eval_llm,path) for path in expected}
    assert load_from_routing==expected
    assert load_from_routing['foo_model_bar']=='transformers'
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
    unequal_token_mean=(1.+27.)/10;unequal_batch_mean=(1.+3.)/2
    assert unequal_token_mean==2.8 and unequal_batch_mean==2.
    report=dict(status='passed',source_commit=COMMIT,device='cpu',torch=torch.__version__,seed=42,steps=80,learning_rate=.003,
        train_rows=2,held_out_rows=1,before_train=before_train,after_train=after_train,
        before_held_out=before_held,after_held_out=after_held,history=history,generated=generated,
        aggregation_counterexample=dict(token_mean=unequal_token_mean,batch_mean=unequal_batch_mean),
        load_from_routing=load_from_routing,
        corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in TRACKED})
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    build=ROOT/'build/experiments/ch08-results.json';build.parent.mkdir(parents=True,exist_ok=True);build.write_text(text)
    print(text,end='')
if __name__=='__main__':main()
