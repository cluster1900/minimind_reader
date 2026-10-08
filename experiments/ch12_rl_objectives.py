"""Execute unmodified source AST blocks with supplied toy intermediates.

This isolates objective arithmetic; it does not run a reward model or a full RL job.
"""
from _support import source,record
import ast,types
import torch
def fragment(path,function,start,end,env):
    tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    def assigned(n):
        if not isinstance(n,ast.Assign):return set()
        return {x.id for t in n.targets for x in ast.walk(t) if isinstance(x,ast.Name)}
    for node in ast.walk(fn):
        body=getattr(node,'body',None)
        if not isinstance(body,list):continue
        starts=[i for i,n in enumerate(body) if start in assigned(n)]
        ends=[i for i,n in enumerate(body) if end in assigned(n)]
        if starts and ends and ends[-1]>=starts[0]:
            block=ast.Module(body=body[starts[0]:ends[-1]+1],type_ignores=[])
            exec(compile(ast.fix_missing_locations(block),str(path),'exec'),env)
            return
    raise AssertionError(f'Cannot locate source block {function}: {start}..{end}')
def main():
    src,files=source('minimind',['trainer/train_ppo.py','trainer/train_grpo.py','trainer/rollout_engine.py','model/model_minimind.py'])
    from trainer.train_ppo import CriticModel
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
    from trainer.rollout_engine import compute_per_token_logps
    torch.manual_seed(42);torch.set_num_threads(1)
    ppo=src/'trainer/train_ppo.py';grpo=src/'trainer/train_grpo.py'
    env=dict(torch=torch,args=types.SimpleNamespace(gamma=1.,lam=1.,device='cpu'),B=1,
        old_resp_values=torch.tensor([[.2,.4,0.]]),token_rewards=torch.tensor([[0.,1.,0.]]),resp_policy_mask=torch.tensor([[1.,1.,0.]]))
    fragment(ppo,'ppo_train_epoch','gen_len','advantages',env)
    torch.testing.assert_close(env['returns'],torch.tensor([[1.,1.,0.]]))
    torch.testing.assert_close(env['advantages'],torch.tensor([[1.,-1.,0.]]),rtol=2e-5,atol=2e-5)
    group=dict(torch=torch,rewards=torch.tensor([1.,3.,2.,2.]),args=types.SimpleNamespace(num_generations=2))
    fragment(grpo,'grpo_train_epoch','grouped_rewards','advantages',group)
    torch.testing.assert_close(group['advantages'],torch.tensor([-1/1.0001,1/1.0001,0.,0.]))
    grads={};losses={}
    for mode in ['grpo','cispo']:
        logps=torch.tensor([[.4],[.4]]).log().requires_grad_(True)
        e=dict(torch=torch,args=types.SimpleNamespace(loss_type=mode,epsilon=.2,epsilon_high=5.,beta=0.),
            ref_per_token_logps=logps.detach().clone(),per_token_logps=logps,
            old_per_token_logps=torch.tensor([[.2],[.2]]).log(),advantages=torch.tensor([1.,-1.]),completion_mask=torch.ones(2,1))
        fragment(grpo,'grpo_train_epoch','kl_div','policy_loss',e)
        e['policy_loss'].backward();grads[mode]=logps.grad.flatten().tolist();losses[mode]=e['policy_loss'].item()
    torch.testing.assert_close(torch.tensor(grads['grpo']),torch.tensor([0.,1.]))
    torch.testing.assert_close(torch.tensor(grads['cispo']),torch.tensor([-1.,1.]))
    # Distinguish the source's reply mean from a global token mean.
    per_token=torch.tensor([[1.,99.,99.],[3.,3.,3.]],requires_grad=True)
    mask=torch.tensor([[1.,0.,0.],[1.,1.,1.]])
    reduction=dict(torch=torch,per_token_loss=per_token,completion_mask=mask)
    fragment(grpo,'grpo_train_epoch','policy_loss','policy_loss',reduction)
    reply_mean=reduction['policy_loss'];reply_mean.backward()
    torch.testing.assert_close(reply_mean,torch.tensor(2.))
    token_mean=(per_token.detach()*mask).sum()/mask.sum()
    torch.testing.assert_close(token_mean,torch.tensor(2.5))
    torch.testing.assert_close(per_token.grad,torch.tensor([[.5,0.,0.],[1/6,1/6,1/6]]))
    cfg=MiniMindConfig(hidden_size=16,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
        vocab_size=64,intermediate_size=32,max_position_embeddings=32,flash_attn=False)
    critic=CriticModel(cfg).eval();calls=[]
    handle=critic.model.norm.register_forward_hook(lambda *a:calls.append(1));vals=critic(torch.tensor([[1,4,2]]));handle.remove()
    assert vals.shape==(1,3) and len(calls)==2
    model=MiniMindForCausalLM(cfg).eval();ids=torch.tensor([[1,4,7,2]])
    lp=compute_per_token_logps(model,ids,2)
    logits=model(ids).logits[:,1:3]
    manual=logits.log_softmax(-1).gather(-1,ids[:,2:,None]).squeeze(-1)
    torch.testing.assert_close(lp,manual)
    assert compute_per_token_logps(model,ids,0).shape==(1,0)
    from trainer.rollout_engine import SGLangRolloutEngine
    remote=object.__new__(SGLangRolloutEngine)
    remote.base_url='http://unused.invalid';remote.timeout=1
    remote.tokenizer=types.SimpleNamespace(pad_token_id=0,eos_token_id=2,decode=lambda ids,**kw:'controlled')
    response=types.SimpleNamespace(raise_for_status=lambda:None,
        json=lambda:[{'meta_info':{'output_ids':[7,2],'output_token_logprobs':[[-.7,2]]}}])
    remote.http=types.SimpleNamespace(post=lambda *a,**kw:response)
    sampled=remote.rollout(torch.tensor([[0,1,4]]),torch.tensor([[0,1,1]]),1,2)
    torch.testing.assert_close(sampled.per_token_logps,torch.tensor([[0.,-.7]]))
    assert sampled.completion_mask.tolist()==[[1,1]] and sampled.prompt_lens.tolist()==[2]
    record(12,__file__,files,dict(validation='source AST arithmetic blocks and real Critic/logprob helper; no full RL run',
        missing_remote_logprobs=sampled.per_token_logps.tolist(),missing_remote_logprob_mask=sampled.completion_mask.tolist(),
        gae_returns=env['returns'].tolist(),normalized_gae=env['advantages'].tolist(),group_advantages=group['advantages'].tolist(),
        clipped_gradients=grads,objective_values=losses,critic_norm_calls=len(calls),critic_shape=list(vals.shape),logprob_alignment=True,
        unequal_lengths=dict(reply_mean=reply_mean.item(),global_token_mean=token_mean.item(),gradients=per_token.grad.tolist())))
if __name__=='__main__':main()
