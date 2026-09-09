"""Tensor basics and a real small MiniMind forward/backward on CPU."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/'minimind'
COMMIT='6fc918beb68a0d8c40452338df6319fe168014ba'


def main():
    actual=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    assert actual==COMMIT,f'Source revision changed: {actual}'
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--','model/model_minimind.py'],text=True).strip()
    sys.path.insert(0,str(SOURCE))
    import torch
    import transformers
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
    torch.set_num_threads(1);torch.manual_seed(42)
    a=torch.arange(24,dtype=torch.float32).reshape(2,3,4)
    assert a[1,2,3].item()==23
    assert a.mean(-1,keepdim=True).shape==(2,3,1)
    assert (a+torch.tensor([1.,2.,3.,4.])).shape==a.shape
    torch.testing.assert_close((a+torch.tensor([1.,2.,3.,4.]))[0,0],torch.tensor([1.,3.,5.,7.]))
    layer=torch.nn.Linear(2,3,bias=False)
    with torch.no_grad():layer.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[1.,1.]]))
    y=layer(torch.tensor([[1.,2.]]))
    torch.testing.assert_close(y,torch.tensor([[1.,2.,3.]]))
    w=torch.tensor(2.,requires_grad=True)
    loss=(3*w-7).square();loss.backward()
    torch.testing.assert_close(w.grad,torch.tensor(-6.))
    with torch.no_grad():w-=0.05*w.grad
    new_loss=(3*w-7).square()
    assert new_loss<loss
    c=MiniMindConfig(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=2,intermediate_size=64,vocab_size=32,max_position_embeddings=64,
        flash_attn=False,dropout=0.0,use_moe=False)
    model=MiniMindForCausalLM(c)
    ids=torch.tensor([[1,4,7,2],[1,5,3,2]],dtype=torch.long)
    out=model(ids,labels=ids)
    assert out.logits.shape==(2,4,32)
    assert torch.isfinite(out.loss)
    torch.testing.assert_close(out.logits.softmax(-1).sum(-1),torch.ones(2,4))
    out.loss.backward()
    assert model.model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.model.embed_tokens.weight.grad).all()
    report=dict(status='passed',source_commit=actual,device='cpu',seed=42,
        python=sys.version,torch=torch.__version__,transformers=transformers.__version__,platform=platform.platform(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((SOURCE/'model/model_minimind.py').read_bytes()).hexdigest(),
        tensor_last_value=a[1,2,3].item(),linear_output=y.tolist(),scalar_gradient=w.grad.item(),
        scalar_after_update=w.item(),loss_before=loss.item(),loss_after=new_loss.item(),
        logits_shape=list(out.logits.shape),model_loss=out.loss.item(),parameters=sum(p.numel() for p in model.parameters()))
    target=ROOT/'build/experiments/ch01-results.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
