"""Real waveform loading/resampling and injection, with explicit encoder substitutes."""
from _support import source,record,ROOT
import _support
_support.REVISIONS={**_support.REVISIONS,'minimind-o':'b5d1bc11ece8de17404cbc034c3e09718d0fecdf'}
import numpy as np
import torch
import soundfile as sf
import os
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'build/numba')
def main():
    _,files=source('minimind-o',['model/model_omni.py','dataset/omni_dataset.py'])
    from model.model_omni import SenseVoiceAudioProcessor
    from dataset.omni_dataset import OmniDataset
    from _omni_fixture import tiny_omni
    torch.manual_seed(42);torch.set_num_threads(1)
    class Frontend(torch.nn.Module):
        def forward(self,wav,lengths):
            self.shape=tuple(wav.shape);self.lengths=lengths.tolist()
            return torch.arange(4*560,dtype=torch.float32).reshape(1,4,560),torch.tensor([3])
    frontend=Frontend();processor=SenseVoiceAudioProcessor(frontend)
    stage=ROOT/'build/experiments/ch16';stage.mkdir(parents=True,exist_ok=True)
    wave=np.sin(2*np.pi*440*np.arange(800)/8000).astype(np.float32)*.2
    sf.write(stage/'stereo.wav',np.stack([wave,wave*.5],-1),8000,subtype='FLOAT')
    features,length=OmniDataset.process_audio(str(stage/'stereo.wav'),processor)
    assert frontend.shape==(1,1600) and features.shape==(4,560) and length==3
    processor(wave,sampling_rate=8000);assert frontend.shape==(1,800)
    model=tiny_omni().eval()
    class Encoder(torch.nn.Module):
        def __init__(self):super().__init__();self.proj=torch.nn.Linear(560,512)
        def forward(self,x,lengths):return self.proj(x),lengths
    encoder=Encoder().requires_grad_(False);object.__setattr__(model,'audio_encoder',encoder)
    inputs=torch.stack([features,torch.zeros_like(features)])
    encoded=model.encode_audio_inputs(inputs,torch.tensor([3,0]))
    assert encoded[0].shape==(3,32) and encoded[1] is None
    encoded[0].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.audio_proj.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    tokens=torch.tensor([[1,16,16,16,16,9,2],[1,16,16,7,8,9,2]])
    hidden=torch.randn(2,7,32);out=model.inject_audio_features(tokens,hidden,encoded,7)
    torch.testing.assert_close(out[0,1:4],encoded[0]);torch.testing.assert_close(out[0,4:],hidden[0,4:])
    torch.testing.assert_close(out[1],hidden[1]);assert out.shape==hidden.shape
    assert not any(n.startswith('audio_encoder.') for n,_ in model.named_parameters())
    class ChangedLengthEncoder(Encoder):
        def forward(self,x,lengths):return self.proj(x),torch.ones_like(lengths)
    object.__setattr__(model,'audio_encoder',ChangedLengthEncoder().requires_grad_(False))
    changed_length=model.encode_audio_inputs(inputs,torch.tensor([3,0]))
    assert changed_length[0].shape==(3,32)
    record(16,__file__,files,dict(wave_input_samples=800,wave_output_samples=1600,input_sr=8000,target_sr=16000,
        frontend='synthetic output, real processor wrapper',encoder='small Linear substitute, not SenseVoice',
        feature_shape=list(features.shape),valid_length=length,encoded_shape=list(encoded[0].shape),
        encoder_reported_length=1,retained_length_from_input=changed_length[0].shape[0],
        dummy_row_preserved=True,injection_preserves_length=True,unfilled_marker_preserved=True,
        projector_gradient=True,encoder_not_registered=True,processor_does_not_resample=True),repo='minimind-o')
if __name__=='__main__':main()
