"""Scripted generation through real multi-turn packing and real mock tools."""
from _support import source,record
import json,random,types
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
        def __init__(self,texts):self.texts=iter(texts);self.calls=0
        def rollout(self,prompt_ids,attention_mask,**kw):
            self.calls+=1;text=next(self.texts);ids=torch.tensor([tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]])
            return RolloutResult(torch.cat([prompt_ids,ids],1),ids,torch.full(ids.shape,-.5),[text],torch.tensor([prompt_ids.size(1)]),torch.ones_like(ids))
    tools=[TOOLS[0]];messages=[{'role':'user','content':'计算1加2'}]
    engine=Scripted([call,answer])
    result=rollout_single(engine,tok,[dict(m) for m in messages],tools,device='cpu',thinking_ratio=0.)
    final,context,p,r,mask,lp,turns,unfinished=result
    assert engine.calls==2 and final==answer and not unfinished
    assert len(r)==len(mask)==len(lp)
    assert tok.encode(context,add_special_tokens=False)==p+r
    generated=[i for i,m in zip(r,mask) if m];observed=[i for i,m in zip(r,mask) if not m]
    assert tok.eos_token_id not in generated
    assert '<tool_response>' in tok.decode(observed,skip_special_tokens=False)
    assert all(v==0. for v,m in zip(lp,mask) if not m)
    assert execute_tool('calculate_math',{'expression':'1+2'})=={'result':'3'}
    assert execute_tool('not_a_tool',{}) is None
    assert parse_tool_calls('<tool_call>{bad}</tool_call>')==[]
    array_call='<tool_call>[1]</tool_call>'
    assert parse_tool_calls(array_call)==[[1]]
    try:rollout_single(Scripted([array_call]),tok,[dict(m) for m in messages],tools,device='cpu',thinking_ratio=0.)
    except AttributeError as exc:assert 'get' in str(exc)
    else:raise AssertionError('Expected missing schema validation to surface at call.get')
    assert validate_gt_in_text('答案是13' ,['3'])=={'3'}
    one=rollout_single(Scripted([call]),tok,[dict(m) for m in messages],tools,max_turns=1,device='cpu',thinking_ratio=0.)
    assert one[-1] is True and '<tool_response>' in one[1]
    reward=calculate_rewards(['prompt'],[final],[['3']],[tools],1,None,device='cpu',turn_outputs_batch=[turns],unfinished_batch=[False])
    assert reward.item()==3.
    unfinished_reward=calculate_rewards(['prompt'],[call],[['3']],[tools],1,None,device='cpu',turn_outputs_batch=[[call]],unfinished_batch=[True])
    assert unfinished_reward.item()<reward.item()
    record(13,__file__,files,dict(generation='scripted, not a learned policy',turns=engine.calls,
        response_tokens=len(r),generated_tokens=sum(mask),observation_tokens=len(mask)-sum(mask),
        token_reconstruction_exact=True,final_answer=final,reward=reward.item(),unfinished_reward=unfinished_reward.item(),
        json_array_accepted_then_rollout_errors=True,
        malformed_json_calls=[],unknown_tool=None,max_turns_still_executes_tool=True,numeric_substring_gt_false_positive=True))
if __name__=='__main__':main()
