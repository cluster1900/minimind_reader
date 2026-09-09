"""Actual VLM forward with tiny real SigLIP, plus placeholder boundary checks."""
from _support import source,record,ROOT
import torch
def main():
    _,files=source('minimind-v',['model/model_vlm.py','model/model_minimind.py'])
    from model.model_vlm import VLMConfig,MiniMindVLM
    from _tiny_vision import tiny_vision_loading
    torch.manual_seed(42);torch.set_num_threads(1)
    stage=ROOT/'build/experiments/ch14/vision';stage.mkdir(parents=True,exist_ok=True)
    c=VLMConfig(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        intermediate_size=64,vocab_size=6400,max_position_embeddings=64,flash_attn=False,image_hidden_size=16,image_token_len=4)
    with tiny_vision_loading():model=MiniMindVLM(c,vision_model_path=str(stage)).eval()
    assert all(not p.requires_grad for p in model.vision_encoder.parameters())
    ids=torch.tensor([[1,12,12,12,12,7,2]]);pixels={'pixel_values':torch.randn(1,3,32,32)}
    labels=ids.clone();labels[:,:5]=-100
    out=model(ids,labels=labels,pixel_values=pixels,use_cache=True);out.loss.backward()
    assert out.logits.shape==(1,7,6400)
    assert sum(p.grad.abs().sum() for p in model.vision_proj.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    calls=[];h=model.vision_encoder.register_forward_hook(lambda *a:calls.append(1))
    with torch.no_grad():
        pref=model(ids,pixel_values=pixels,use_cache=True)
        step=model(torch.tensor([[5]]),pixel_values=pixels,past_key_values=pref.past_key_values,use_cache=True)
        assert len(calls)==1
        full=model(torch.cat([ids,torch.tensor([[5]])],1),pixel_values=pixels)
        torch.testing.assert_close(step.logits,full.logits[:,-1:],atol=3e-6,rtol=3e-5)
    h.remove()
    tokens=torch.tensor([[1,12,12,12,12,3,12,12,12,12,2]])
    hidden=torch.randn(1,11,32);features=torch.stack([torch.ones(4,32),torch.full((4,32),2.)])[None]
    replaced=model.count_vision_proj(tokens,hidden,features,seqlen=11)
    torch.testing.assert_close(replaced[:,1:5],features[:,0]);torch.testing.assert_close(replaced[:,6:10],features[:,1])
    torch.testing.assert_close(replaced[:,[0,5,10]],hidden[:,[0,5,10]])
    adjacent=torch.tensor([[1]+[12]*8+[2]])
    shorter=model.count_vision_proj(adjacent,torch.zeros(1,10,32),features,seqlen=10)
    assert shorter.shape==(1,6,32)
    record(14,__file__,files,dict(encoder='real tiny randomly initialized SigLIP; loading mocked, no pretrained weights',
        logits_shape=list(out.logits.shape),projector_gradient_nonzero=True,encoder_gradients_none=True,
        cached_decode_error=(step.logits-full.logits[:,-1:]).abs().max().item(),prefill_only_encoder_call=True,
        separated_images_correct=True,adjacent_marker_input_length=10,adjacent_marker_output_length=6),repo='minimind-v')
if __name__=='__main__':main()
