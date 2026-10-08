"""Actual session/consumer/PCM and browser handler code with controlled external boundaries."""
from _support import source,record
import ast,base64,json,queue,subprocess,threading,types
from unittest.mock import patch
import numpy as np
import torch
def main():
    torch.manual_seed(42);torch.set_num_threads(1)
    src,files=source('minimind-o',['model/model_omni.py','webui/web_demo.py','webui/web_demo.html'])
    import model.model_omni as omni
    class VAD:
        def __init__(self,path):pass
        def reset(self):pass
        def __call__(self,w,sr):return 1. if np.abs(w).mean()>.25 else 0.
    hi=np.full(2048,12000,dtype=np.int16).tobytes();lo=np.zeros(13*1024,dtype=np.int16).tobytes()
    with patch.object(omni,'SileroVAD',VAD):
        s=omni.RealtimeSession('controlled')
        assert s.push_chunk(np.zeros(1024,dtype=np.float32))=='listening'
        assert s.push_chunk(np.ones(2048,dtype=np.float32))=='listening' and s.speaking
        assert s.push_chunk(np.zeros(13*1024,dtype=np.float32))=='speech_end'
        kept=len(s.get_audio());assert kept==4096
        s.reset();s.generating=True
        assert s.push_chunk(np.ones(2048,dtype=np.float32))=='interrupt'
        s.reset();s.generating=True
        assert s.push_chunk(np.ones(4096,dtype=np.float32))=='interrupt'
        interrupted_retained=len(s.get_audio());assert interrupted_retained==2048
        s.reset();assert s.push_chunk(np.ones(2048,dtype=np.float32))=='listening'
        mixed=np.concatenate([np.zeros(13*1024,dtype=np.float32),np.ones(2048,dtype=np.float32)])
        assert s.push_chunk(mixed)=='speech_end' and not s.speaking
        ended_audio=s.get_audio()
        assert len(ended_audio)==3072 and not ended_audio[-1024:].any()
        assert s.speech_samples==0 # trailing 2048 speech samples were never consumed
    tree=ast.parse((src/'webui/web_demo.py').read_text())
    names={'_mimi_decode','pcm_bytes','stream_pcm','run_generate','realtime'}
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    for n in definitions:n.decorator_list=[]
    module=ast.Module(body=definitions,type_ignores=[])
    class NoThread:
        def __init__(self,*a,**kw):pass
        def start(self):pass
    class Codec:
        def decode(self,codes):return types.SimpleNamespace(audio_values=torch.full((1,1,codes.shape[-1]*1920),.1))
    scenarios={}
    for mode in ['normal','stop','speech']:
        events=queue.Queue();events.put(hi);events.put(lo);sent=[];grad=[]
        class Model:
            def generate(self,*a,**kw):
                for i in range(3):
                    grad.append(torch.is_grad_enabled())
                    yield torch.tensor([[10+j for j in range(i+1)]]) if i<2 else None,[10+j for j in range(8)]
        class WS:
            def send(self,value):
                d=json.loads(value);sent.append(d)
                if d['type']=='generating' and mode!='normal':events.put(json.dumps({'type':'stop'}) if mode=='stop' else hi)
                if d['type']=='done':events.put(json.dumps({'type':'end'}))
        cfg=types.SimpleNamespace(max_history_turns=2,audio_chunk_frames=2,audio_overlap=1)
        tokenizer=types.SimpleNamespace(eos_token_id=2,decode=lambda ids,**kw:''.join(chr(65+i-10) for i in ids))
        env=dict(np=np,torch=torch,json=json,queue=types.SimpleNamespace(Queue=lambda:events,Empty=queue.Empty),
            threading=types.SimpleNamespace(Thread=NoThread),base64=base64,RealtimeSession=omni.RealtimeSession,
            MODEL_LOCK=threading.Lock(),M={'vad_path':'controlled','cfg':cfg,'model':Model(),'tokenizer':tokenizer,'mimi':Codec(),'device':'cpu'},
            prepare_turn=lambda *a,**kw:(None,None,None,'问题','问题',None,[None]),
            build_ids=lambda *a:torch.tensor([[1]]),voice_args=lambda *a:{})
        exec(compile(module,str(src/'webui/web_demo.py'),'exec'),env)
        with patch.object(omni,'SileroVAD',VAD):env['realtime'](WS())
        done=[d for d in sent if d['type']=='done'];assert len(done)==1
        pcm=[len(base64.b64decode(d['data'])) for d in sent if d['type']=='pcm']
        assert done[0]['interrupted']==(mode!='normal') and not any(grad)
        if mode=='normal':assert pcm==[7680,3840]
        else:assert pcm==[] and len(grad)==1
        scenarios[mode]={'pcm_bytes':pcm,'model_yields':len(grad),'interrupted':done[0]['interrupted']}
    html=(src/'webui/web_demo.html').read_text()
    stop=next(line.strip() for line in html.splitlines() if line.strip().startswith('const stopPcm='))
    start=html.index('    ws.onmessage=e=>{');end=html.index('\n    ws.onclose=',start);handler=html[start:end]
    js='''let stopped=0, playing=[{stop(){stopped++}},{stop(){stopped++}}], asstPlay=true, nextT=0;
let ac={currentTime:5}, curText='AB', hist=[], curMsg=null, segs=[], ws={}, statuses=[];
const setSt=(a,b)=>statuses.push([a,b]);
'''+stop+'\n'+handler+'''\nws.onmessage({data:JSON.stringify({type:'done',interrupted:true})});
process.stdout.write(JSON.stringify({stopped,playing:playing.length,asstPlay,nextT,hist,statuses}));'''
    browser=json.loads(subprocess.check_output(['node','-e',js],text=True))
    assert browser['stopped']==2 and browser['playing']==0 and not browser['asstPlay'] and browser['nextT']==5
    record(20,__file__,files,dict(validation='real source bodies; synthetic VAD, controlled queue/socket, fake PCM decoder and WebAudio nodes',
        vad_window_samples=1024,speech_trigger_samples=2048,silence_trigger_samples=13312,retained_samples=kept,
        scenarios=scenarios,browser_stop=browser,wrapper_disables_grad=True,
        speech_end_input_samples=len(mixed),speech_end_dropped_tail_samples=2048,
        interrupted_input_chunk=4096,interrupted_retained_samples=interrupted_retained),repo='minimind-o')
if __name__=='__main__':main()
