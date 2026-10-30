"""Real DPO objective, gradient direction and Dataset alignment."""
import _support
from _support import source,record,ROOT
_support.REVISIONS={**_support.REVISIONS,'minimind':'233d00fa068ec2b8c0d4f07b703934d6391a5a73'}
import json,math,os,random
os.environ['HF_DATASETS_CACHE']=str(ROOT/'build/hf-datasets');os.environ['HF_HUB_OFFLINE']='1'
import torch
EMPTY_THINK='<think>\n\n</think>\n\n'
def reconstruct(tok,item,side):
    return tok.decode(torch.cat([item[f'x_{side}'][:1],item[f'y_{side}']]).tolist(),skip_special_tokens=False)
def main():
    src,files=source('minimind',['trainer/train_dpo.py','dataset/lm_dataset.py'])
    from trainer.train_dpo import dpo_loss,logits_to_log_probs
    from dataset.lm_dataset import DPODataset,post_processing_chat
    from transformers import AutoTokenizer
    torch.manual_seed(42);torch.set_num_threads(1)
    ref=torch.tensor([[-1.,-2.,-3.],[-1.,-2.,-3.]])
    mask=torch.tensor([[1.,1.,0.],[1.,1.,0.]])
    policy=ref.clone().requires_grad_(True);beta=.15
    baseline=dpo_loss(ref,policy,mask,beta);torch.testing.assert_close(baseline,torch.tensor(math.log(2.)))
    baseline.backward();assert (policy.grad[0,:2]<0).all() and (policy.grad[1,:2]>0).all()
    assert torch.equal(policy.grad[:,2],torch.zeros(2))
    better=ref.clone();better[0,0]+=.4;worse=ref.clone();worse[1,0]+=.4
    assert dpo_loss(ref,better,mask,beta)<baseline<dpo_loss(ref,worse,mask,beta)
    ignored=ref.clone();ignored[:,2]+=100;torch.testing.assert_close(dpo_loss(ref,ignored,mask,beta),baseline)
    logits=torch.tensor([[[1.,2.,3.],[2.,1.,0.]]]);labels=torch.tensor([[2,0]])
    expected=torch.tensor([[3.,2.]])-logits.logsumexp(-1)
    torch.testing.assert_close(logits_to_log_probs(logits,labels),expected)
    tok=AutoTokenizer.from_pretrained(src/'model',local_files_only=True)
    prompt=[{'role':'user','content':'问题'}]
    row={'chosen':prompt+[{'role':'assistant','content':'较好答案'}], 'rejected':prompt+[{'role':'assistant','content':'较差答案'}]}
    stage=ROOT/'build/experiments/ch11';stage.mkdir(parents=True,exist_ok=True)
    same={'chosen':row['chosen'],'rejected':row['chosen']}
    file=stage/'pair.jsonl';file.write_text(json.dumps(row,ensure_ascii=False)+'\n'+json.dumps(same,ensure_ascii=False)+'\n')
    ds=DPODataset(str(file),tok,max_length=64);random.seed(42);item=ds[0]
    for side in ['chosen','rejected']:
        x=item['x_'+side];y=item['y_'+side];m=item['mask_'+side]
        assert x.shape==y.shape==m.shape==(63,)
        assert torch.equal(x[1:],y[:-1]);assert (y>=0).all();assert m.sum()>0
    empty=dpo_loss(ref,ref,torch.zeros_like(mask),beta).item();assert abs(empty-math.log(2))<1e-6
    chosen_prompt=tok.apply_chat_template(row['chosen'],tokenize=False,add_generation_prompt=False)
    rejected_prompt=tok.apply_chat_template(row['rejected'],tokenize=False,add_generation_prompt=False)
    same_prompt=tok.apply_chat_template(same['chosen'],tokenize=False,add_generation_prompt=False)
    assert EMPTY_THINK in chosen_prompt and EMPTY_THINK in rejected_prompt and EMPTY_THINK in same_prompt
    for flag in (True,False):
        processed_chosen=post_processing_chat(chosen_prompt,remove_empty_think=flag)
        processed_rejected=post_processing_chat(rejected_prompt,remove_empty_think=flag)
        assert (EMPTY_THINK in processed_chosen)==(EMPTY_THINK in processed_rejected)
        assert (EMPTY_THINK not in processed_chosen) if flag else (EMPTY_THINK in processed_chosen)
    assert post_processing_chat(same_prompt,remove_empty_think=True)!=post_processing_chat(same_prompt,remove_empty_think=False)
    random.seed(42);identical=ds[1]
    assert torch.equal(identical['x_chosen'],identical['x_rejected'])
    assert torch.equal(identical['y_chosen'],identical['y_rejected'])
    assert torch.equal(identical['mask_chosen'],identical['mask_rejected'])
    item_has_empty=EMPTY_THINK in reconstruct(tok,item,'chosen')
    assert item_has_empty==(EMPTY_THINK in reconstruct(tok,item,'rejected'))
    identical_has_empty=EMPTY_THINK in reconstruct(tok,identical,'chosen')
    assert identical_has_empty==(EMPTY_THINK in reconstruct(tok,identical,'rejected'))
    record(11,__file__,files,dict(initial_loss=baseline.item(),chosen_gradient=policy.grad[0].tolist(),
        rejected_gradient=policy.grad[1].tolist(),improved_chosen_loss=dpo_loss(ref,better,mask,beta).item(),
        empty_mask_loss=empty,dataset_length=63,chosen_supervised=int(item['mask_chosen'].sum()),
        rejected_supervised=int(item['mask_rejected'].sum()),identical_answers_share_decision=True,
        identical_pair_counts=[int(identical['mask_chosen'].sum()),int(identical['mask_rejected'].sum())],
        different_pair_keeps_empty_think=item_has_empty,
        identical_pair_keeps_empty_think=identical_has_empty))
if __name__=='__main__':main()
