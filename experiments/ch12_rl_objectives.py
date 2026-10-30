"""Execute unmodified source AST blocks with supplied toy intermediates.

This isolates objective arithmetic; it does not run a reward model or a full RL job.
"""
import _support
from _support import source,record
_support.REVISIONS={**_support.REVISIONS,'minimind':'233d00fa068ec2b8c0d4f07b703934d6391a5a73'}
import ast,json,os,socket,subprocess,sys,textwrap,types
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel

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

def run_exact(path,blob,env):
    assert blob in path.read_text(),f'missing source span in {path}'
    exec(compile(textwrap.dedent(blob),str(path),'exec'),env)

def _ddp_worker():
    source_root=Path(os.environ['MINIMIND_SOURCE_ROOT'])/'minimind'
    sys.path.insert(0,str(source_root))
    import torch.distributed as dist
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
    from trainer.train_ppo import CriticModel
    from trainer.rollout_engine import compute_per_token_logps
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);port=int(os.environ['CH12_DDP_PORT'])
    out=Path(os.environ['CH12_DDP_OUT'])/f'rank{rank}.json'
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world)
    try:
        torch.manual_seed(0)
        cfg=MiniMindConfig(hidden_size=16,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
            vocab_size=64,intermediate_size=32,max_position_embeddings=32,flash_attn=False)
        model=MiniMindForCausalLM(cfg)
        wrapped=DistributedDataParallel(model,broadcast_buffers=False)
        assert wrapped.broadcast_buffers is False
        ids=torch.tensor([[1,4,7,2]])
        lp=compute_per_token_logps(wrapped,ids,2)
        logits=wrapped.module(ids).logits[:,1:3]
        manual=logits.log_softmax(-1).gather(-1,ids[:,2:,None]).squeeze(-1)
        torch.testing.assert_close(lp,manual)
        out_tr=wrapped(ids)
        assert out_tr.logits.shape==(1,4,64)
        critic=DistributedDataParallel(CriticModel(cfg),broadcast_buffers=False)
        vals=critic(ids)
        assert vals.shape==(1,4)
        before=wrapped.module.model.freqs_cos.clone()
        if rank==0:wrapped.module.model.freqs_cos.add_(1)
        wrapped(ids)
        delta=torch.tensor([(wrapped.module.model.freqs_cos-before).abs().max().item()])
        gathered=[torch.zeros(1) for _ in range(world)]
        dist.all_gather(gathered,delta)
        if rank==0:
            assert gathered[0].item()>0.5 and gathered[1].item()<1e-6
        wrapped.zero_grad()
        local=torch.tensor([[1,4,7,2]] if rank==0 else [[3,5,6,1]])
        wrapped(local).logits.mean().backward()
        g=next(p.grad for p in wrapped.parameters() if p.grad is not None).detach().clone()
        glist=[torch.zeros_like(g) for _ in range(world)]
        dist.all_gather(glist,g)
        torch.testing.assert_close(glist[0],glist[1])
        payload=dict(ok=True,rank=rank,backend='gloo',world_size=world,broadcast_buffers=False,
            logps_unwrapped=True,training_forward_through_ddp=True,critic_through_ddp=True,
            buffers_not_broadcast=True,grads_equal_after_backward=True,
            buffer_deltas=[t.item() for t in gathered])
        out.write_text(json.dumps(payload))
    finally:
        dist.barrier();dist.destroy_process_group()

def run_ddp():
    port=None
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    stage=Path(__file__).resolve().parents[1]/'build/experiments/ch12-ddp'
    if stage.exists():
        for old in stage.glob('rank*.json'):old.unlink()
    else:stage.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'CH12_DDP_WORKER':'1','CH12_DDP_PORT':str(port),'CH12_DDP_OUT':str(stage),
         'MINIMIND_SOURCE_ROOT':os.environ.get('MINIMIND_SOURCE_ROOT','/workspace/repos/minimind_resources')}
    procs=[]
    try:
        for rank in range(2):
            e={**env,'RANK':str(rank),'WORLD_SIZE':'2','LOCAL_RANK':str(rank)}
            procs.append(subprocess.Popen([sys.executable,__file__],env=e,cwd=str(Path(__file__).resolve().parents[1])))
        for p in procs:
            p.wait(timeout=180)
            if p.returncode!=0:raise AssertionError(f'DDP worker exited {p.returncode}')
    except Exception:
        for p in procs:
            if p.poll() is None:p.kill()
        raise
    reports=[]
    for rank in range(2):
        path=stage/f'rank{rank}.json'
        assert path.is_file(),f'missing DDP report {path}'
        reports.append(json.loads(path.read_text()))
    assert all(r['ok'] for r in reports)
    return reports[0]

def main():
    if os.environ.get('CH12_DDP_WORKER'):
        _ddp_worker();return
    src,files=source('minimind',['trainer/train_ppo.py','trainer/train_grpo.py','trainer/rollout_engine.py','model/model_minimind.py'])
    from trainer.train_ppo import CriticModel
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
    from trainer.rollout_engine import compute_per_token_logps,TorchRolloutEngine,SGLangRolloutEngine
    torch.manual_seed(42);torch.set_num_threads(1)
    ppo=src/'trainer/train_ppo.py';grpo=src/'trainer/train_grpo.py';roll=src/'trainer/rollout_engine.py'
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

    tok=types.SimpleNamespace(pad_token_id=0,eos_token_id=2)
    # Generated token equals PAD ID: naive !=PAD would mask it; scatter restores completion_mask.
    pad_env=dict(torch=torch,outputs=torch.tensor([[0,1,4,0,7,2]]),tokenizer=tok,
        completion_ids=torch.tensor([[0,7,2]]),prompt_lens=torch.tensor([3]),
        args=types.SimpleNamespace(device='cpu'),
        rollout_result=types.SimpleNamespace(completion_mask=torch.tensor([[1,1,1]])))
    naive=(pad_env['outputs']!=tok.pad_token_id).long()
    assert naive.tolist()==[[0,1,1,0,1,1]]
    run_exact(grpo,
        '        full_mask = (outputs != tokenizer.pad_token_id).long()\n'
        '        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)\n'
        '        full_mask.scatter_(1, logp_pos + 1, rollout_result.completion_mask.to(args.device, dtype=full_mask.dtype))\n',
        pad_env)
    assert pad_env['full_mask'].tolist()==[[0,1,1,1,1,1]]
    assert pad_env['logp_pos'].tolist()==[[2,3,4]]
    ppo_pad=dict(torch=torch,tokenizer=tok,gen_out=torch.tensor([[0,1,4,0,7,2]]),
        completion_ids=torch.tensor([[0,7,2]]),prompt_lens=torch.tensor([3]),
        rollout_result=types.SimpleNamespace(completion_mask=torch.ones(1,3)),
        args=types.SimpleNamespace(device='cpu'),prompts=['x'])
    run_exact(ppo,
        '        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]\n'
        '        labels = gen_out[:, 1:].clone()  # [B, P+R-1]\n'
        '        B = len(prompts)\n'
        '        resp_labels = completion_ids\n'
        '        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)\n'
        '        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx\n'
        '        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()\n'
        '        full_mask.scatter_(1, logp_pos + 1, resp_pad_mask.to(full_mask.dtype))\n'
        '        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask\n'
        '        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)\n'
        '        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)\n'
        '        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()\n',
        ppo_pad)
    assert ppo_pad['full_mask'].tolist()==[[0,1,1,1,1,1]]
    assert ppo_pad['resp_policy_mask'].tolist()==[[1.,1.,1.]]

    # First EOS is kept; tokens after it drop even if rollout mask is still 1.
    eos_env=dict(torch=torch,args=types.SimpleNamespace(device='cpu'),tokenizer=tok,
        completion_ids=torch.tensor([[7,2,9,0],[7,9,0,2]]),
        rollout_result=types.SimpleNamespace(completion_mask=torch.tensor([[1,1,1,0],[1,1,0,0]])))
    run_exact(grpo,
        '        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()\n'
        '        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]\n'
        '        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)\n'
        '        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]\n'
        '        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]\n',
        eos_env)
    assert eos_env['completion_mask'].tolist()==[[1,1,0,0],[1,1,0,0]]
    assert eos_env['eos_idx'].tolist()==[1,3]
    ppo_eos=dict(torch=torch,tokenizer=tok,gen_out=torch.tensor([[0,1,4,7,2,9]]),
        completion_ids=torch.tensor([[7,2,9]]),prompt_lens=torch.tensor([3]),
        rollout_result=types.SimpleNamespace(completion_mask=torch.ones(1,3)),
        args=types.SimpleNamespace(device='cpu'),prompts=['x'])
    run_exact(ppo,
        '        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]\n'
        '        labels = gen_out[:, 1:].clone()  # [B, P+R-1]\n'
        '        B = len(prompts)\n'
        '        resp_labels = completion_ids\n'
        '        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)\n'
        '        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx\n'
        '        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()\n'
        '        full_mask.scatter_(1, logp_pos + 1, resp_pad_mask.to(full_mask.dtype))\n'
        '        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask\n'
        '        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)\n'
        '        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)\n'
        '        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()\n',
        ppo_eos)
    assert ppo_eos['resp_policy_mask'].tolist()==[[1.,1.,0.]] and ppo_eos['resp_lengths'].tolist()==[2]

    ppo_txt=ppo.read_text();grpo_txt=grpo.read_text();roll_txt=roll.read_text()
    assert 'padding_side="left"' in ppo_txt and 'padding_side="left"' in grpo_txt
    assert 'DistributedDataParallel(actor_model, device_ids=[local_rank], broadcast_buffers=False)' in ppo_txt
    assert 'DistributedDataParallel(critic_model, device_ids=[local_rank], broadcast_buffers=False)' in ppo_txt
    assert 'DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)' in grpo_txt
    assert 'res = actor_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])' in ppo_txt
    assert 'res = model(outputs, attention_mask=full_mask)' in grpo_txt
    assert 'unwrapped = model.module if isinstance(model, DistributedDataParallel) else model' in roll_txt
    assert 'prompt_len = prompt_ids.size(1)' in roll_txt
    assert 'valid_ids = ids[mask.bool()].tolist()' in roll_txt

    # Torch backend: prompt_len is the left-padded rectangular width, completion_mask starts all-1.
    fake_tok=types.SimpleNamespace(pad_token_id=0,eos_token_id=2,batch_decode=lambda ids,**kw:['x']*len(ids))
    torch_engine=TorchRolloutEngine(model,fake_tok,device='cpu')
    prompt_ids=torch.tensor([[0,0,1,4],[0,5,6,7]])
    attention_mask=torch.tensor([[0,0,1,1],[0,1,1,1]])
    torch_out=torch_engine.rollout(prompt_ids,attention_mask,num_generations=1,max_new_tokens=2,temperature=1.)
    assert torch_out.prompt_lens.tolist()==[4,4]
    assert torch_out.completion_mask.tolist()==[[1]*torch_out.completion_ids.size(1)]*2
    assert torch_out.output_ids.size(1)==4+torch_out.completion_ids.size(1)
    torch_two=torch_engine.rollout(prompt_ids[:1],attention_mask[:1],num_generations=2,max_new_tokens=2,temperature=1.)
    assert torch_two.prompt_lens.tolist()==[4,4] and torch_two.output_ids.size(0)==2

    remote=object.__new__(SGLangRolloutEngine)
    remote.base_url='http://unused.invalid';remote.timeout=1
    remote.tokenizer=types.SimpleNamespace(pad_token_id=0,eos_token_id=2,decode=lambda ids,**kw:'controlled')
    posted={}
    response=types.SimpleNamespace(raise_for_status=lambda:None,
        json=lambda:[{'meta_info':{'output_ids':[7,2],'output_token_logprobs':[[-.7,2]]}}])
    remote.http=types.SimpleNamespace(post=lambda *a,**kw:response)
    sampled=remote.rollout(torch.tensor([[0,1,4]]),torch.tensor([[0,1,1]]),1,2)
    torch.testing.assert_close(sampled.per_token_logps,torch.tensor([[0.,-.7]]))
    assert sampled.completion_mask.tolist()==[[1,1]] and sampled.prompt_lens.tolist()==[2]

    # Left-pad strip + unequal completions: SGLang records true prompt_lens and right-pads.
    def capture_post(*a,json=None,**kw):
        posted['payload']=json
        return types.SimpleNamespace(raise_for_status=lambda:None,json=lambda:[
            {'meta_info':{'output_ids':[8,2],'output_token_logprobs':[[-.5,8],[-.7,2]]}},
            {'meta_info':{'output_ids':[9],'output_token_logprobs':[[-.3,9]]}}])
    remote.http=types.SimpleNamespace(post=capture_post)
    sgl=remote.rollout(prompt_ids,attention_mask,1,3)
    assert posted['payload']['input_ids']==[[1,4],[5,6,7]]
    assert sgl.prompt_lens.tolist()==[2,3]
    assert sgl.completion_ids.tolist()==[[8,2],[9,0]]
    assert sgl.completion_mask.tolist()==[[1,1],[1,0]]
    torch.testing.assert_close(sgl.per_token_logps,torch.tensor([[-.5,-.7],[-.3,0.]]))
    assert sgl.output_ids.tolist()==[[1,4,8,2,0],[5,6,7,9,0]]

    ddp=run_ddp()
    record(12,__file__,files,dict(validation='source AST arithmetic blocks and real Critic/logprob helper; no full RL run',
        missing_remote_logprobs=sampled.per_token_logps.tolist(),missing_remote_logprob_mask=sampled.completion_mask.tolist(),
        gae_returns=env['returns'].tolist(),normalized_gae=env['advantages'].tolist(),group_advantages=group['advantages'].tolist(),
        clipped_gradients=grads,objective_values=losses,critic_norm_calls=len(calls),critic_shape=list(vals.shape),logprob_alignment=True,
        unequal_lengths=dict(reply_mean=reply_mean.item(),global_token_mean=token_mean.item(),gradients=per_token.grad.tolist()),
        pad_id_naive_mask=naive.tolist(),pad_id_restored_mask=pad_env['full_mask'].tolist(),
        eos_completion_mask=eos_env['completion_mask'].tolist(),ppo_eos_policy_mask=ppo_eos['resp_policy_mask'].tolist(),
        torch_prompt_lens=torch_out.prompt_lens.tolist(),torch_completion_mask=torch_out.completion_mask.tolist(),
        sglang_stripped_input_ids=posted['payload']['input_ids'],sglang_prompt_lens=sgl.prompt_lens.tolist(),
        sglang_right_padded_completion_ids=sgl.completion_ids.tolist(),sglang_right_padded_output_ids=sgl.output_ids.tolist(),
        sglang_completion_mask=sgl.completion_mask.tolist(),sglang_right_padded_logps=sgl.per_token_logps.tolist(),
        ddp=ddp,real_sglang_service=False,ddp_device_ids_cuda_path=False))
if __name__=='__main__':main()
