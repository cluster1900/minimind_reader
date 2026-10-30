"""Scripted generation through real multi-turn packing and real mock tools."""
from _support import source,record
import _support
_support.REVISIONS={**_support.REVISIONS,'minimind':'233d00fa068ec2b8c0d4f07b703934d6391a5a73'}
import json,random,re
import torch
def main():
    src,files=source('minimind',['trainer/train_agent.py','trainer/rollout_engine.py','model/tokenizer_config.json'])
    from trainer.train_agent import rollout_single,parse_tool_calls,execute_tool,calculate_rewards,validate_gt_in_text,TOOLS
    from trainer.rollout_engine import RolloutResult
    from transformers import AutoTokenizer
    torch.manual_seed(42);random.seed(42)
    tok=AutoTokenizer.from_pretrained(src/'model',local_files_only=True)
    call='<tool_call>\n'+json.dumps({'name':'calculate_math','arguments':{'expression':'1+2'}})+'\n</tool_call>'
    answer='计算结果是3。'
    class Scripted:
        def __init__(self,texts,mask=None,logps=None):
            self.texts=iter(texts);self.calls=0;self.seen=[];self.mask=mask;self.logps=logps
        def rollout(self,prompt_ids,attention_mask,**kw):
            self.calls+=1;self.seen.append(prompt_ids[0].tolist())
            text=next(self.texts);ids=torch.tensor([tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]])
            cmask=self.mask if self.mask is not None else torch.ones_like(ids)
            lp=self.logps if self.logps is not None else torch.full(ids.shape,-.5)
            return RolloutResult(torch.cat([prompt_ids,ids],1),ids,lp,[text],torch.tensor([prompt_ids.size(1)]),cmask)
    tools=[TOOLS[0]];messages=[{'role':'user','content':'计算1加2'}]
    engine=Scripted([call,answer])
    result=rollout_single(engine,tok,[dict(m) for m in messages],tools,device='cpu',thinking_ratio=0.)
    final,context,p,r,mask,lp,turns,unfinished=result
    assert engine.calls==2 and final==answer and not unfinished
    assert len(r)==len(mask)==len(lp)
    # Next-round input is the retained prompt+response prefix, not a re-encode of final_context.
    ans_ids=tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]
    assert engine.seen[1]==p+r[:-len(ans_ids)]
    assert tok.encode(context,add_special_tokens=False)!=p+r
    # EOS stays in the trajectory; policy mask is 0 so it does not enter the action set.
    assert tok.eos_token_id in r
    assert all(m==0 for i,m in zip(r,mask) if i==tok.eos_token_id)
    generated=[i for i,m in zip(r,mask) if m]
    observed=[i for i,m in zip(r,mask) if not m]
    assert tok.eos_token_id not in generated
    assert '<tool_response>' in tok.decode(observed,skip_special_tokens=False)
    assert context.count('<tool_response>')==1 and '<|agent_observation' not in context
    assert all(v==-.5 for v,m in zip(lp,mask) if m)
    assert all(v==0. for i,m,v in zip(r,mask,lp) if not m and i!=tok.eos_token_id)
    assert execute_tool('calculate_math',{'expression':'1+2'})=={'result':'3'}
    assert execute_tool('not_a_tool',{}) is None
    assert parse_tool_calls('<tool_call>{bad}</tool_call>')==[]
    # Non-object JSON is dropped at parse time; rollout no longer calls .get on a list.
    for payload in ('[1]','"x"','1','true','null'):
        assert parse_tool_calls(f'<tool_call>{payload}</tool_call>')==[]
    array_call='<tool_call>[1]</tool_call>'
    array=rollout_single(Scripted([array_call]),tok,[dict(m) for m in messages],tools,device='cpu',thinking_ratio=0.)
    assert array[0]==array_call and array[-1] is False
    # Dict call with non-dict arguments is coerced to {} and still executed.
    bad_args='<tool_call>'+json.dumps({'name':'calculate_math','arguments':[1,2]})+'</tool_call>'
    coerced=rollout_single(Scripted([bad_args]),tok,[dict(m) for m in messages],tools,max_turns=1,device='cpu',thinking_ratio=0.)
    assert '{"result": "0"}' in coerced[1] and coerced[-1] is True
    class DropMarker:
        def __init__(self,real):self._real=real
        def apply_chat_template(self,*a,**kw):
            return re.sub(r'<\|agent_observation_[^|]+\|>','',self._real.apply_chat_template(*a,**kw))
        def __getattr__(self,name):return getattr(self._real,name)
        def __call__(self,*a,**kw):return self._real(*a,**kw)
    try:rollout_single(Scripted([call]),DropMarker(tok),[dict(m) for m in messages],tools,device='cpu',thinking_ratio=0.)
    except RuntimeError as exc:assert 'assistant content boundary' in str(exc)
    else:raise AssertionError('Expected marker loss to raise RuntimeError')
    # completion_mask.sum() is used as a prefix length; shorter log-probs fail closed.
    short=Scripted([answer],logps=torch.full((1,4),-.5))
    try:rollout_single(short,tok,[dict(m) for m in messages],tools,device='cpu',thinking_ratio=0.)
    except RuntimeError as exc:assert 'token/logprob length mismatch' in str(exc)
    else:raise AssertionError('Expected log-prob length error')
    # A hole in completion_mask is not a boolean keep-mask: prefix[:sum] keeps the hole and drops the tail.
    hole_ids=tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]
    hole_mask=torch.ones(1,len(hole_ids),dtype=torch.long);hole_mask[0,1]=0
    hole=rollout_single(Scripted([answer],mask=hole_mask),tok,[dict(m) for m in messages],tools,device='cpu',thinking_ratio=0.)
    prefix=hole_ids[:int(hole_mask[0].sum().item())]
    kept_ones=[t for t,m in zip(hole_ids,hole_mask[0].tolist()) if m]
    assert hole[3]==prefix and prefix!=kept_ones and hole_ids[1] in prefix
    assert validate_gt_in_text('答案是13',['3'])=={'3'}
    one=rollout_single(Scripted([call]),tok,[dict(m) for m in messages],tools,max_turns=1,device='cpu',thinking_ratio=0.)
    assert one[-1] is True and '<tool_response>' in one[1]
    reward=calculate_rewards(['prompt'],[final],[['3']],[tools],1,None,device='cpu',turn_outputs_batch=[turns],unfinished_batch=[False])
    assert reward.item()==3.
    unfinished_reward=calculate_rewards(['prompt'],[call],[['3']],[tools],1,None,device='cpu',turn_outputs_batch=[[call]],unfinished_batch=[True])
    assert unfinished_reward.item()<reward.item()
    record(13,__file__,files,dict(generation='scripted, not a learned policy',turns=engine.calls,
        response_tokens=len(r),generated_tokens=sum(mask),observation_tokens=len(mask)-sum(mask),
        token_reconstruction_exact=False,next_round_input_equals_retained_trajectory=True,
        eos_retained_in_response=True,eos_policy_mask_zero=True,observation_appended_once=True,
        ids_mask_logps_aligned=True,final_answer=final,reward=reward.item(),unfinished_reward=unfinished_reward.item(),
        non_object_json_calls=[],non_object_json_rollout_no_get_crash=True,
        marker_missing_raises=True,logprob_length_mismatch_raises=True,
        noncontiguous_mask_uses_prefix_sum=True,
        malformed_json_calls=[],unknown_tool=None,max_turns_still_executes_tool=True,numeric_substring_gt_false_positive=True))
if __name__=='__main__':main()
