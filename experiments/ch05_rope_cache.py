"""RoPE invariants, cached decoding and sampling support in the real model."""
import hashlib,json,math,os,subprocess,sys,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/'minimind'
COMMIT='6fc918beb68a0d8c40452338df6319fe168014ba'

def main():
    import torch
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--','model/model_minimind.py'],text=True).strip()
    sys.path.insert(0,str(SOURCE))
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM,precompute_freqs_cis,apply_rotary_pos_emb
    torch.manual_seed(42);torch.set_num_threads(1)
    cos,sin=precompute_freqs_cis(8,64)
    q=torch.randn(1,1,1,8);k=torch.randn_like(q)
    def rotate(x,pos):return apply_rotary_pos_emb(x,x,cos[pos:pos+1],sin[pos:pos+1])[0]
    torch.testing.assert_close(rotate(q,3).square().sum(),q.square().sum())
    lhs=(rotate(q,3)*rotate(k,5)).sum();rhs=(q*rotate(k,2)).sum()
    torch.testing.assert_close(lhs,rhs)
    cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        vocab_size=32,intermediate_size=64,max_position_embeddings=64,flash_attn=False,dropout=0.0)
    model=MiniMindForCausalLM(cfg).eval();ids=torch.tensor([[1,4,7,9,2]])
    with torch.no_grad():
        full=model(ids).logits
        prefix=model(ids[:,:3],use_cache=True)
        suffix=model(ids[:,3:],past_key_values=prefix.past_key_values,use_cache=True)
        torch.testing.assert_close(full[:,3:],suffix.logits,atol=2e-6,rtol=2e-5)
        past=None;pieces=[];lengths=[]
        for i in range(ids.size(1)):
            out=model(ids[:,i:i+1],past_key_values=past,use_cache=True)
            past=out.past_key_values;pieces.append(out.logits);lengths.append(past[0][0].shape[1])
        torch.testing.assert_close(full,torch.cat(pieces,1),atol=2e-6,rtol=2e-5)
    opts=dict(max_new_tokens=3,do_sample=False,top_k=0,top_p=1.,temperature=1.,eos_token_id=None)
    a=model.generate(ids[:,:4],use_cache=True,return_kv=True,**opts)
    b=model.generate(ids[:,:4],use_cache=False,**opts)
    assert torch.equal(a['generated_ids'],b)
    assert a['past_kv'][0][0].shape[1]==a['generated_ids'].shape[1]-1
    observed=[]
    def sample(probs,num_samples):observed.append(probs.clone());return probs.argmax(-1,keepdim=True)
    toy=MiniMindForCausalLM(MiniMindConfig(hidden_size=8,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,vocab_size=4,intermediate_size=16,max_position_embeddings=16)).eval()
    def generate_distribution(logits,prompt,**kwargs):
        def forward(self,*args,**kw):return types.SimpleNamespace(logits=logits[None,None,:],past_key_values=None)
        toy.forward=types.MethodType(forward,toy)
        with patch('torch.multinomial',side_effect=sample):
            toy.generate(torch.tensor([prompt]),max_new_tokens=1,use_cache=False,eos_token_id=None,temperature=1.,**kwargs)
        return observed[-1][0]
    nucleus=generate_distribution(torch.tensor([.4,.3,.2,.1]).log(),[3],top_k=0,top_p=.6)
    torch.testing.assert_close(nucleus,torch.tensor([4/7,3/7,0.,0.]))
    ties=generate_distribution(torch.tensor([1.,1.,1.,0.]),[3],top_k=2,top_p=1.)
    assert (ties>0).sum()==3
    penalty=generate_distribution(torch.tensor([-2.,2.,0.,1.]),[0,1],top_k=0,top_p=1.,repetition_penalty=2.)
    torch.testing.assert_close(penalty,torch.tensor([-4.,1.,0.,1.]).softmax(0))
    report=dict(status='passed',source_commit=COMMIT,device='cpu',seed=42,torch=torch.__version__,
        source_sha256=hashlib.sha256((SOURCE/'model/model_minimind.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rope_relative_error=(lhs-rhs).abs().item(),chunk_cache_error=(full[:,3:]-suffix.logits).abs().max().item(),
        token_cache_error=(full-torch.cat(pieces,1)).abs().max().item(),cache_lengths=lengths,
        generated_ids=b.tolist(),returned_cache_length=a['past_kv'][0][0].shape[1],
        nucleus_distribution=nucleus.tolist(),top_k_tie_support=int((ties>0).sum()),penalty_distribution=penalty.tolist())
    target=ROOT/'build/experiments/ch05-results.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
