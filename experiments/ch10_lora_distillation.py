"""Real LoRA injection/merge and real distillation loss, without teacher weights."""
from _support import source,record,ROOT
import torch
def main():
    _,files=source('minimind',['model/model_minimind.py','model/model_lora.py','trainer/train_lora.py','trainer/train_distillation.py'])
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
    from model.model_lora import apply_lora,save_lora,merge_lora
    from trainer.train_distillation import distillation_loss
    torch.manual_seed(42);torch.set_num_threads(1)
    cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        intermediate_size=64,vocab_size=40,max_position_embeddings=32,flash_attn=False)
    model=MiniMindForCausalLM(cfg).eval();ids=torch.tensor([[1,4,7,2]])
    before=model(ids).logits.detach();apply_lora(model,rank=4)
    torch.testing.assert_close(before,model(ids).logits)
    targets=[name for name,m in model.named_modules() if hasattr(m,'lora')]
    assert targets==[f'model.layers.{i}.self_attn.{p}' for i in range(2) for p in ['q_proj','o_proj']]
    for n,p in model.named_parameters():p.requires_grad_('lora' in n)
    model(ids,labels=ids).loss.backward()
    a_grad=sum(p.grad.abs().sum().item() for n,p in model.named_parameters() if 'lora.A.' in n)
    b_grad=sum(p.grad.abs().sum().item() for n,p in model.named_parameters() if 'lora.B.' in n)
    assert a_grad==0 and b_grad>0
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.5);opt.step();opt.zero_grad(set_to_none=True)
    model(ids,labels=ids).loss.backward()
    assert sum(p.grad.abs().sum().item() for n,p in model.named_parameters() if 'lora.A.' in n)>0
    stage=ROOT/'build/experiments/ch10';stage.mkdir(parents=True,exist_ok=True)
    live_before_merge=model(ids).logits.detach()
    save_lora(model,stage/'adapter.pth');merge_lora(model,stage/'adapter.pth',stage/'merged.pth')
    fresh=MiniMindForCausalLM(cfg).eval();fresh.load_state_dict(torch.load(stage/'merged.pth'))
    error=(fresh(ids).logits-live_before_merge).abs().max().item();assert error<.002
    assert not any('lora' in k for k in torch.load(stage/'merged.pth'))
    square_cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
        intermediate_size=64,vocab_size=32,max_position_embeddings=32,flash_attn=False)
    square=MiniMindForCausalLM(square_cfg).eval();apply_lora(square,rank=4)
    assert hasattr(square.lm_head,'lora')
    with torch.no_grad():
        square.lm_head.lora.A.weight.zero_()
        square.lm_head.lora.A.weight[:,:4]=torch.eye(4)*.5
        square.lm_head.lora.B.weight.fill_(.25)
        square_live=square(ids).logits.clone();square_embedding=square.model.embed_tokens.weight.clone()
    save_lora(square,stage/'square-adapter.pth');merge_lora(square,stage/'square-adapter.pth',stage/'square-merged.pth')
    square_fresh=MiniMindForCausalLM(square_cfg).eval();square_fresh.load_state_dict(torch.load(stage/'square-merged.pth'))
    square_error=(square_fresh(ids).logits-square_live).abs().max().item()
    square_embedding_error=(square_fresh.model.embed_tokens.weight-square_embedding).abs().max().item()
    assert square_error>.01 and square_embedding_error>.1
    s=torch.tensor([[.2,-.1,.3],[1.,-.5,0.]],requires_grad=True)
    t=torch.tensor([[1.,0.,-.2],[.5,.4,.1]],requires_grad=True);T=2.
    actual=distillation_loss(s,t,temperature=T)
    pt=(t.detach()/T).softmax(-1);ps=(s/T).softmax(-1)
    expected=T*T*(pt*(pt.log()-ps.log())).sum(-1).mean()
    torch.testing.assert_close(actual,expected);actual.backward()
    assert s.grad.abs().sum()>0 and t.grad is None
    equal=distillation_loss(t.detach(),t.detach(),temperature=T).item();assert abs(equal)<1e-6
    # Shape-compatible vocabulary permutation changes meaning and loss.
    permuted=distillation_loss(t.detach(),t.detach()[:,[1,0,2]],temperature=T).item();assert permuted>1e-3
    record(10,__file__,files,dict(injected=targets,lora_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        square_tied_head_merge_error=square_error,square_tied_embedding_change=square_embedding_error,
        first_a_gradient=a_grad,first_b_gradient=b_grad,merge_max_logit_error=error,
        distillation_loss=actual.item(),equal_distribution_loss=equal,permuted_vocabulary_loss=permuted,teacher_gradient=None))
if __name__=='__main__':main()
