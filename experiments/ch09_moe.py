"""Routing invariants and gradient paths of the real MoE implementation."""
import _support
from _support import source,record
_support.REVISIONS={**_support.REVISIONS,'minimind':'233d00fa068ec2b8c0d4f07b703934d6391a5a73'}
import torch
def main():
    _,files=source('minimind',['model/model_minimind.py'])
    from model.model_minimind import MiniMindConfig,MOEFeedForward,MiniMindForCausalLM
    torch.manual_seed(42);torch.set_num_threads(1)
    cfg=MiniMindConfig(hidden_size=8,intermediate_size=16,use_moe=True,num_experts=4,num_experts_per_tok=1,router_aux_loss_coef=.01)
    assert cfg.norm_topk_prob and cfg.num_experts_per_tok==1
    moe=MOEFeedForward(cfg).train()
    with torch.no_grad():
        moe.gate.weight.zero_();moe.gate.weight[:,0]=torch.tensor([2.,1.,-.5,-1.])
    x=torch.randn(1,3,8);x[:,:,0]=torch.tensor([1.,2.,3.])
    y=moe(x)
    torch.testing.assert_close(y,moe.experts[0](x))
    # Top-1 four-way reverse: gate / input / expert / aux
    x=x.detach().requires_grad_(True)
    moe.zero_grad(set_to_none=True)
    y=moe(x);y.square().mean().backward()
    main_gate_grad=moe.gate.weight.grad.abs().max().item()
    main_input_grad=x.grad.abs().max().item()
    selected_expert_grad=max(p.grad.abs().max().item() for p in moe.experts[0].parameters())
    assert main_gate_grad>1e-5
    assert main_input_grad>1e-5
    assert selected_expert_grad>1e-5
    assert all(p.grad is not None and p.grad.abs().sum()==0 for e in moe.experts[1:] for p in e.parameters())
    x_ref=x.detach().requires_grad_(True)
    moe.experts[0](x_ref).square().mean().backward()
    torch.testing.assert_close(x.grad,x_ref.grad)
    moe.zero_grad(set_to_none=True);x.grad=None
    y=moe(x);moe.aux_loss.backward()
    aux_only_gate_grad=moe.gate.weight.grad.abs().max().item()
    aux_only_input_grad=x.grad.abs().max().item()
    assert aux_only_gate_grad>1e-5
    assert aux_only_input_grad>1e-5
    assert all(p.grad is None for e in moe.experts for p in e.parameters())
    moe.zero_grad(set_to_none=True);x.grad=None
    y=moe(x);aux=moe.aux_loss
    (y.square().mean()+aux).backward()
    aux_gate_grad=moe.gate.weight.grad.abs().max().item();assert aux_gate_grad>1e-5
    x_det=x.detach()
    perm=torch.tensor([2,0,1]);torch.testing.assert_close(moe(x_det[:,perm]),moe(x_det)[:,perm])
    moe.eval();ev=moe(x_det);assert moe.aux_loss.item()==0;torch.testing.assert_close(ev,y)
    uniform=[]
    for k in [1,2]:
        c=MiniMindConfig(hidden_size=8,intermediate_size=16,num_experts=4,num_experts_per_tok=k,router_aux_loss_coef=.01)
        m=MOEFeedForward(c).train()
        with torch.no_grad():m.gate.weight.zero_()
        m(x_det);torch.testing.assert_close(m.aux_loss,torch.tensor(.01*k));uniform.append(m.aux_loss.item())
    c=MiniMindConfig(hidden_size=16,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        vocab_size=32,intermediate_size=32,max_position_embeddings=32,use_moe=True)
    model=MiniMindForCausalLM(c);out=model(torch.tensor([[1,4,7,2]]),labels=torch.tensor([[1,4,7,2]]))
    torch.testing.assert_close(out.aux_loss,sum(l.mlp.aux_loss for l in model.model.layers))
    record(9,__file__,files,dict(top1_main_gate_grad=main_gate_grad,top1_main_input_grad=main_input_grad,
        top1_selected_expert_grad=selected_expert_grad,inactive_expert_grads='present and zero',
        top1_main_input_matches_expert_only=True,top1_aux_gate_grad=aux_only_gate_grad,
        top1_aux_input_grad=aux_only_input_grad,top1_aux_expert_grads='none',
        gate_grad_with_aux=aux_gate_grad,token_permutation_equivariant=True,
        uniform_prob_aux_k1_k2=uniform,model_aux_sum=out.aux_loss.item()))
if __name__=='__main__':main()
