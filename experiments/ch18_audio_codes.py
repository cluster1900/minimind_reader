"""Real dataset delay layout, scripted logits through real stream loop, and tiny Mimi decode."""
from _support import source,record,ROOT
import json,random,types
from unittest.mock import patch
import torch
def main():
    src,files=source('minimind-o',['model/model_omni.py','dataset/omni_dataset.py','eval_omni.py'])
    from model.model_omni import MiniMindOmni
    from dataset.omni_dataset import OmniDataset
    from transformers import AutoTokenizer,MimiConfig,MimiModel
    import pyarrow as pa,pyarrow.parquet as pq
    torch.manual_seed(42);random.seed(42);torch.set_num_threads(1)
    tok=AutoTokenizer.from_pretrained(src/'model',local_files_only=True)
    assert tok.encode('\n',add_special_tokens=False)==[234] and tok.decode([201])!='\n'
    codes=[[10+100*i+t for i in range(8)] for t in range(3)]
    row={'conversations':json.dumps([{'role':'system','content':'测试'},{'role':'user','content':'说你好'},{'role':'assistant','content':'你好'}],ensure_ascii=False),
        'answer_audios':[[v for frame in codes for v in frame]]}
    stage=ROOT/'build/experiments/ch18';stage.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row]),stage/'sample.parquet')
    ds=OmniDataset(str(stage/'sample.parquet'),tok,max_length=128,scheduled_sampling=0.)
    random.seed(42);x,y,ay,*_=ds[0]
    start=int((y!=-100).nonzero()[0])+1
    first=[]
    for i in range(8):
        pos=(ay[i]!=-100).nonzero().flatten();first.append(int(pos[0]))
        assert int(pos[0])==start+i
        assert ay[i,pos].tolist()==[codes[t][i] for t in range(3)]+[2050]
        assert x[i,pos[0]]==2049 and x[i,pos[0]+1]==codes[0][i]
    class Scripted:
        generate=MiniMindOmni.generate;stream_generate=MiniMindOmni.stream_generate
        def __init__(self,thinking):
            self.audio_pad_token=2049;self.audio_spk_token=2051
            self.config=types.SimpleNamespace(think_end_ids=[26,234,234]);self.n=0;self.thinking=thinking;self.grad=[]
        def forward(self,*args,**kw):
            step=self.n;self.n+=1;self.grad.append(torch.is_grad_enabled())
            text_seq=[26,234,234,7,8,2] if self.thinking else [7,8,2]
            tid=text_seq[min(step,len(text_seq)-1)]
            text=torch.full((1,1,300),-1000.);text[0,0,tid]=1000.
            audio=[];offset=4 if self.thinking else 1
            for i in range(8):
                frame=step-offset-i;code=10+100*i+frame if 0<=frame<3 else 2050
                logits=torch.full((1,1,2112),-1000.);logits[0,0,code]=1000.;audio.append(logits)
            return types.SimpleNamespace(logits=text,audio_logits=audio,past_key_values=None)
    outputs={}
    for thinking in [False,True]:
        model=Scripted(thinking)
        with torch.no_grad(),patch('torch.multinomial',side_effect=lambda p,n:p.argmax(-1,keepdim=True)):
            generated=list(model.generate(torch.tensor([[1,4]]),stream=True,return_audio_codes=True,max_new_tokens=24,use_cache=False,open_thinking=thinking))
        frames=[f for _,f in generated if f is not None];assert frames==codes
        outputs[str(thinking)]={'frames':frames,'steps':model.n,'audio_after_text_end':any(t is None and f is not None for t,f in generated)}
    probe=Scripted(False)
    with torch.enable_grad(),patch('torch.multinomial',side_effect=lambda p,n:p.argmax(-1,keepdim=True)):
        stream=probe.generate(torch.tensor([[1]]),stream=True,max_new_tokens=1,use_cache=False);list(stream)
    assert probe.grad==[True] # decorator covered generator creation, not later iteration
    nonstream=Scripted(False)
    with torch.enable_grad(),patch('torch.multinomial',side_effect=lambda p,n:p.argmax(-1,keepdim=True)):
        last=nonstream.generate(torch.tensor([[1,4]]),stream=False,return_audio_codes=True,
            max_new_tokens=24,use_cache=False)
    assert last==(None,codes[-1]) and not any(nonstream.grad), (last,nonstream.grad)
    text_prefixes=[t.tolist()[0] for t,f in generated if t is not None]
    assert all(b[:len(a)]==a for a,b in zip(text_prefixes,text_prefixes[1:]))
    config=MimiConfig(hidden_size=32,num_filters=4,codebook_dim=16,vector_quantization_hidden_dimension=16,
        num_quantizers=8,num_semantic_quantizers=1,upsample_groups=32,num_hidden_layers=1,intermediate_size=64,
        num_attention_heads=4,num_key_value_heads=4,frame_rate=12.5)
    codec=MimiModel(config).eval()
    with torch.no_grad():wave=codec.decode(torch.tensor(codes).T[None]).audio_values
    assert wave.shape==(1,1,5760) and torch.isfinite(wave).all()
    record(18,__file__,files,dict(dataset_shape=list(x.shape),assistant_start=start,first_audio_label_positions=first,
        nonstream_last_yield=list(last),nonstream_disables_grad=True,text_yields_are_prefixes=True,
        streamed=outputs,generator_iteration_grad_enabled_without_outer_context=True,
        codec='real randomly initialized tiny Mimi',decoded_shape=list(wave.shape),sampling_rate=24000,
        frame_rate=12.5,duration_seconds=5760/24000,newline_token_ids=[234],default_enter_id_201_is_not_newline=True),repo='minimind-o')
if __name__=='__main__':main()
