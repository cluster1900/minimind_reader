"""Real branch topology, bridge capture, speaker condition and dual cache."""
from _support import source,record
import torch
from unittest.mock import patch
def main():
    _,files=source('minimind-o',['model/model_omni.py','model/model_minimind.py'])
    from model.model_omni import OmniConfig
    from _omni_fixture import tiny_omni
    torch.manual_seed(42);torch.set_num_threads(1)
    model=tiny_omni().eval();ids=torch.tensor([[1,4,7]])
    assert model.thinker is model.model and model.thinker.lm_head is model.lm_head
    assert OmniConfig().bridge_layer==3
    states=[];bridge=[]
    h1=model.thinker.layers[1].register_forward_hook(lambda m,a,o:states.append(o[0].detach().clone()))
    h2=model.talker.embed_proj.register_forward_pre_hook(lambda m,a:bridge.append(a[0].detach().clone()))
    with torch.no_grad():base=model(ids,use_cache=True)
    h1.remove();h2.remove();torch.testing.assert_close(states[0],bridge[0])
    audio=torch.full((1,8,3),model.audio_pad_token,dtype=torch.long);audio[:,0,1]=17
    with torch.no_grad():changed=model(torch.cat([audio,ids[:,None]],1))
    torch.testing.assert_close(base.logits,changed.logits)
    assert (base.audio_logits[0]-changed.audio_logits[0]).abs().max()>1e-5
    speaker=torch.randn(1,192);marked=audio.clone();marked[:,:,0]=model.audio_spk_token
    with torch.no_grad():
        a=model(torch.cat([marked,ids[:,None]],1),spk_emb=torch.zeros_like(speaker))
        b=model(torch.cat([marked,ids[:,None]],1),spk_emb=speaker)
        unmarked_a=model(ids,spk_emb=torch.zeros_like(speaker));unmarked_b=model(ids,spk_emb=speaker)
    torch.testing.assert_close(a.logits,b.logits);assert (a.audio_logits[0]-b.audio_logits[0]).abs().max()>1e-5
    torch.testing.assert_close(unmarked_a.audio_logits[0],unmarked_b.audio_logits[0])
    with torch.no_grad():
        step=model(torch.tensor([[9]]),past_key_values=base.past_key_values,use_cache=True)
        full=model(torch.tensor([[1,4,7,9]]))
        torch.testing.assert_close(step.logits,full.logits[:,-1:],atol=2e-6,rtol=3e-5)
        for x,y in zip(step.audio_logits,full.audio_logits):torch.testing.assert_close(x,y[:,-1:],atol=2e-6,rtol=3e-5)
    assert len(base.past_key_values)==6 and all(k.shape[1]==4 for k,v in step.past_key_values)
    model.zero_grad(set_to_none=True);out=model(ids);out.audio_logits[0].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.thinker.layers[0].parameters())
    assert all(p.grad is None for p in model.thinker.layers[3].parameters())
    bad=tiny_omni(num_hidden_layers=1,num_talker_hidden_layers=1).eval();seen=[]
    h=bad.talker.embed_proj.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
    with torch.no_grad():bad(ids);expected=bad.thinker.embed_tokens(ids)
    h.remove();assert bad.config.bridge_layer==-1;torch.testing.assert_close(seen[0],expected)
    # The fallback preserves already injected features, not just token embeddings.
    audio_ids=torch.tensor([[1,16,7]]);seen.clear();feature=torch.full((1,32),.125)
    h=bad.talker.embed_proj.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
    with torch.no_grad(),patch.object(bad,'encode_audio_inputs',return_value=[feature]):
        bad(audio_ids,audio_inputs=torch.ones(1,1,560),audio_lens=torch.tensor([1]))
    h.remove();torch.testing.assert_close(seen[0][0,1:2],feature)
    assert not torch.allclose(seen[0][0,1],bad.thinker.embed_tokens(audio_ids)[0,1])
    record(17,__file__,files,dict(real_random_model=True,default_bridge_index=3,tiny_bridge_index=1,
        bridge_matches_second_block=True,text_independent_of_output_audio_history=True,speaker_marker_required=True,
        cache_layers=6,audio_only_loss_reaches_prefix_not_later_blocks=True,
        invalid_single_layer_bridge=-1,invalid_bridge_without_modalities_matches_embedding=True,invalid_bridge_preserves_injected_audio=True),repo='minimind-o')
if __name__=='__main__':main()
