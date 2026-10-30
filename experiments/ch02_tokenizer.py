"""Inspect the real tokenizer/template and train only a disposable toy tokenizer."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('MINIMIND_SOURCE_ROOT',ROOT.parent/'minimind_resources'))/'minimind'
COMMIT='233d00fa068ec2b8c0d4f07b703934d6391a5a73'


def main():
    import torch
    from transformers import AutoTokenizer
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    checked=['model/tokenizer.json','model/tokenizer_config.json','trainer/train_tokenizer.py']
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*checked],text=True).strip()
    tokenizer=AutoTokenizer.from_pretrained(SOURCE/'model',local_files_only=True)
    samples=[]
    for text in ['你好，MiniMind！','hello world','空格  和换行\n都要保留。']:
        ids=tokenizer.encode(text,add_special_tokens=False)
        assert tokenizer.decode(ids,skip_special_tokens=False)==text
        samples.append(dict(text=text,ids=ids,characters=len(text),tokens=len(ids),ratio=len(text)/len(ids)))
    token_info={}
    for text in ['<|im_start|>','<|im_end|>','<|image_pad|>','<think>','<tool_call>']:
        ids=tokenizer.encode(text,add_special_tokens=False);assert len(ids)==1
        token_info[text]=dict(id=ids[0],decoded_skip=tokenizer.decode(ids,skip_special_tokens=True))
    assert token_info['<think>']['decoded_skip']=='<think>'
    assert token_info['<tool_call>']['decoded_skip']=='<tool_call>'
    assert token_info['<|image_pad|>']['decoded_skip']==''
    msgs=[{'role':'user','content':'你好'}]
    prompts={}
    for thinking in [False,True]:
        rendered=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,open_thinking=thinking)
        direct=tokenizer.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,open_thinking=thinking)
        assert direct==tokenizer.encode(rendered,add_special_tokens=False)
        assert '\n' in rendered and '\\n' not in rendered
        suffix='<think>\n' if thinking else '<think>\n\n</think>\n\n'
        assert rendered.endswith(suffix)
        prompts[str(thinking)]=rendered
    completed=tokenizer.apply_chat_template(msgs+[{'role':'assistant','content':'你好！'}],tokenize=False,add_generation_prompt=False)
    assert '<think>\n\n</think>' in completed
    tools=[{'type':'function','function':{'name':'add','description':'求和','parameters':{'type':'object','properties':{'a':{'type':'number'}}}}}]
    tool_prompt=tokenizer.apply_chat_template(msgs,tools=tools,tokenize=False,add_generation_prompt=True)
    assert '<tools>' in tool_prompt and 'add' in tool_prompt

    spec=importlib.util.spec_from_file_location('book_train_tokenizer',SOURCE/'trainer/train_tokenizer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stage=ROOT/'build/experiments/ch02-toy';stage.mkdir(parents=True,exist_ok=True)
    mixed=stage/'mixed.jsonl'
    mixed.write_text('\n'.join([
        '{bad',
        json.dumps({'text':' '},ensure_ascii=False),
        json.dumps({'text':'one'},ensure_ascii=False),
        json.dumps({'conversations':[{'role':'user','content':'two'},{'role':'assistant','content':'three'}]},ensure_ascii=False),
        json.dumps({'conversations':[{'role':'user','content':''}]},ensure_ascii=False),
        json.dumps({'text':'four'},ensure_ascii=False),
    ])+'\n')
    texts=list(module.get_texts(str(mixed)))
    assert texts==['one','two\nthree','four']
    assert list(module.get_texts(str(mixed),1))==['one']
    assert list(module.get_texts(str(mixed),2))==['one','two\nthree']
    empty=stage/'empty.jsonl'
    empty.write_text('{bad\n{"text":" "}\n{"conversations":[]}\n')
    try:
        list(module.get_texts(str(empty)))
        raise AssertionError('empty corpus accepted')
    except ValueError as exc:
        assert '没有可用文本' in str(exc)
        empty_error=str(exc)
    try:
        module.train_tokenizer(str(empty),str(stage/'empty-tokenizer'),vocab_size=384)
        raise AssertionError('empty corpus training accepted')
    except ValueError as exc:
        assert '没有可用文本' in str(exc)

    corpus=stage/'tiny.jsonl'
    corpus.write_text('\n'.join(json.dumps({'conversations':[{'role':'user','content':s} ]},ensure_ascii=False)
        for s in ['你好 世界 hello world','low lower newer newest','我们学习模型与代码','token tensor training test']*4)+'\n')
    module.train_tokenizer(str(corpus),str(stage/'tokenizer'),vocab_size=384)
    toy=AutoTokenizer.from_pretrained(stage/'tokenizer',local_files_only=True)
    text='你好 hello';toy_ids=toy.encode(text,add_special_tokens=False)
    assert toy.decode(toy_ids,skip_special_tokens=False)==text
    report=dict(status='passed',source_commit=COMMIT,torch=torch.__version__,
        files={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in checked},
        vocabulary=len(tokenizer),bos=tokenizer.bos_token_id,eos=tokenizer.eos_token_id,pad=tokenizer.pad_token_id,
        samples=samples,tokens=token_info,prompts=prompts,assistant_training_template=completed,
        get_texts=dict(yielded=texts,max_lines_1=['one'],max_lines_2=['one','two\nthree'],
                       empty_error=empty_error),
        toy_vocabulary=len(toy),toy_ids=toy_ids,toy_roundtrip=text,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=ROOT/'build/experiments/ch02-results.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
