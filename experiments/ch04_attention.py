"""Numerical attention intuition, causality and padding tests on real source."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/'minimind'
COMMIT='233d00fa068ec2b8c0d4f07b703934d6391a5a73'


def main():
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--','model/model_minimind.py'],text=True).strip()
    sys.path.insert(0,str(SOURCE))
    import torch
    from model.model_minimind import Attention,MiniMindConfig
    torch.set_num_threads(1);torch.manual_seed(42)
    q=torch.eye(2);k=q.clone();v=torch.tensor([[10.,0.],[0.,20.]])
    scores=q@k.T/math.sqrt(2)
    scores=scores.masked_fill(torch.ones(2,2,dtype=torch.bool).triu(1),float('-inf'))
    weights=scores.softmax(-1);toy_output=weights@v
    p=math.exp(1/math.sqrt(2))/(1+math.exp(1/math.sqrt(2)))
    torch.testing.assert_close(toy_output,torch.tensor([[10.,0.],[10*(1-p),20*p]]))
    torch.testing.assert_close(weights.sum(-1),torch.ones(2))
    config=MiniMindConfig(hidden_size=32,num_attention_heads=4,num_key_value_heads=2,head_dim=8,dropout=0.0,flash_attn=False)
    manual=Attention(config).eval()
    sdpa_config=MiniMindConfig(hidden_size=32,num_attention_heads=4,num_key_value_heads=2,head_dim=8,dropout=0.0,flash_attn=True)
    fast=Attention(sdpa_config).eval();fast.load_state_dict(manual.state_dict())
    x=torch.randn(1,4,32)
    # Identity rotation isolates this chapter from the RoPE mechanism.
    pos=(torch.ones(4,8),torch.zeros(4,8))
    a,cache=manual(x,pos,use_cache=True)
    assert a.shape==x.shape and cache[0].shape==(1,4,2,8)
    real_sdpa=torch.nn.functional.scaled_dot_product_attention
    with patch('torch.nn.functional.scaled_dot_product_attention',wraps=real_sdpa) as call:
        b,_=fast(x,pos)
        assert call.call_count==1
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    changed=x.clone();changed[:,3,:]+=torch.randn(1,32)*20
    future,_=manual(changed,pos)
    torch.testing.assert_close(a[:,:3],future[:,:3])
    mask=torch.tensor([[1.,0.,1.,1.]])
    base,_=manual(x,pos,attention_mask=mask)
    changed=x.clone();changed[:,1,:]+=torch.randn(1,32)*20
    masked,_=manual(changed,pos,attention_mask=mask)
    torch.testing.assert_close(base[:,[0,2,3]],masked[:,[0,2,3]],atol=2e-6,rtol=2e-5)
    with patch('torch.nn.functional.scaled_dot_product_attention',wraps=real_sdpa) as call:
        fast(x,pos,attention_mask=mask)
        assert call.call_count==0
    with patch('torch.nn.functional.scaled_dot_product_attention',wraps=real_sdpa) as call:
        fast(x[:,:1],(pos[0][:1],pos[1][:1]))
        assert call.call_count==0
    # PyTorch 2.6 uses upper-left causal alignment for non-square Q/K.
    rq=torch.zeros(1,1,2,8);rk=torch.zeros(1,1,4,8)
    rv=torch.arange(1.,5.).view(1,1,4,1).expand(1,1,4,8)
    rectangular=real_sdpa(rq,rk,rv,is_causal=True)
    torch.testing.assert_close(rectangular[0,0,:,0],torch.tensor([1.,1.5]))
    # MiniMind uses its explicit tail-aligned mask when a prefix cache exists.
    _,prefix_cache=fast(x[:,:2],(pos[0][:2],pos[1][:2]),use_cache=True)
    with patch('torch.nn.functional.scaled_dot_product_attention',wraps=real_sdpa) as call:
        cached,_=fast(x[:,2:],(pos[0][2:],pos[1][2:]),past_key_value=prefix_cache)
        assert call.call_count==0
    torch.testing.assert_close(cached,a[:,2:],atol=2e-6,rtol=2e-5)
    all_masked,_=manual(x,pos,attention_mask=torch.zeros_like(mask))
    assert torch.isfinite(all_masked).all()
    assert all_masked.abs().sum()>0
    # New baseline: (xq @ xk.T).float() / sqrt, then mask; softmax consumes FP32 scores.
    # Use fp16 activations with an fp32 mask so (1-mask)*-1e9 stays a finite -1e9.
    # An fp16 mask overflows that term to -inf before it is added, which would hide the cast timing.
    low=Attention(config).eval();low.load_state_dict(manual.state_dict());low.half()
    x_h,pos_h=x.half(),(pos[0].half(),pos[1].half())
    real_softmax=torch.nn.functional.softmax
    with patch('torch.nn.functional.softmax',wraps=real_softmax) as sm:
        low_out,_=low(x_h,pos_h,attention_mask=mask)
        assert sm.call_count==1
        low_scores=sm.call_args[0][0]
    assert low_out.dtype==torch.float16 and torch.isfinite(low_out).all()
    assert low_scores.dtype==torch.float32
    padded=low_scores[0,:,1:,1]
    assert torch.isfinite(padded).all() and not torch.isneginf(padded).any()
    padded_residual=(padded+1e9).abs().max().item()
    assert padded_residual<50
    assert torch.isneginf(low_scores[0,:,0,1]).all()
    with patch('torch.nn.functional.softmax',wraps=real_softmax) as sm:
        low_zero,_=low(x_h,pos_h,attention_mask=torch.zeros_like(mask))
        zero_scores=sm.call_args[0][0]
    assert torch.isfinite(low_zero).all() and low_zero.abs().sum()>0
    assert torch.isfinite(zero_scores[0,:,-1]).all()
    qk16=torch.full((1,1,1,8),1000.,dtype=torch.float16)
    overflowed=qk16@qk16.transpose(-2,-1)
    assert torch.isinf(overflowed).all() and torch.isinf(overflowed.float()).all()
    recovered=qk16.float()@qk16.float().transpose(-2,-1)
    assert torch.isfinite(recovered).all()
    torch.testing.assert_close(recovered,torch.tensor([[[[8e6]]]]))
    overflow=Attention(config).eval().half();overflow.load_state_dict(low.state_dict())
    with torch.no_grad():
        overflow.q_norm.weight.fill_(1000);overflow.k_norm.weight.fill_(1000)
    overflow_out,_=overflow(x_h,pos_h)
    assert not torch.isfinite(overflow_out).all()
    report=dict(status='passed',source_commit=COMMIT,device='cpu',torch=torch.__version__,seed=42,
        source_sha256=hashlib.sha256((SOURCE/'model/model_minimind.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        toy_weights=weights.tolist(),toy_output=toy_output.tolist(),
        attention_output_shape=list(a.shape),cache_shape=list(cache[0].shape),
        sdpa_manual_max_error=(a-b).abs().max().item(),
        future_perturbation_prefix_error=(a[:,:3]-future[:,:3]).abs().max().item(),
        masked_key_valid_query_error=(base[:,[0,2,3]]-masked[:,[0,2,3]]).abs().max().item(),
        all_zero_mask_output_nonzero=True,sdpa_called_for_unmasked_prefill=True,
        sdpa_skipped_for_padding_and_single_token=True,
        rectangular_sdpa_first_channel=rectangular[0,0,:,0].tolist(),
        cached_multi_token_skips_sdpa=True,cached_multi_token_error=(cached-a[:,2:]).abs().max().item(),
        fp16_explicit_output_dtype=str(low_out.dtype),
        fp16_explicit_softmax_scores_dtype=str(low_scores.dtype),
        fp16_explicit_padded_key_residual=padded_residual,
        fp16_all_zero_fp32_mask_finite=True,
        fp16_qk_matmul_overflows=True,fp16_cast_after_overflow_stays_inf=True,
        fp16_matmul_in_fp32_recovers=8e6,fp16_large_qk_norm_output_finite=False)
    target=ROOT/'build/experiments/ch04-results.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
