"""Real DPO objective, gradient direction and Dataset alignment."""
from _support import source,record,ROOT
import json,math,os,random
os.environ['HF_DATASETS_CACHE']=str(ROOT/'build/hf-datasets');os.environ['HF_HUB_OFFLINE']='1'
import torch
def main():
    src,files=source('minimind',['trainer/train_dpo.py','dataset/lm_dataset.py'])
    from trainer.train_dpo import dpo_loss,logits_to_log_probs
    from dataset.lm_dataset import DPODataset
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
    random.seed(42);identical=ds[1]
    assert not torch.equal(identical['x_chosen'],identical['x_rejected'])
    record(11,__file__,files,dict(initial_loss=baseline.item(),chosen_gradient=policy.grad[0].tolist(),
        rejected_gradient=policy.grad[1].tolist(),improved_chosen_loss=dpo_loss(ref,better,mask,beta).item(),
        empty_mask_loss=empty,dataset_length=63,chosen_supervised=int(item['mask_chosen'].sum()),
        rejected_supervised=int(item['mask_rejected'].sum()),identical_answers_augmented_differently=True,
        identical_pair_counts=[int(identical['mask_chosen'].sum()),int(identical['mask_rejected'].sum())]))
if __name__=='__main__':main()
