import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from content_contract import verify
class ContentContract(unittest.TestCase):
    def test_dropped_paragraph_fails(self):
        source={'blocks':[{'t':'Para','c':[{'t':'Str','c':'不能被丢掉的正文'}]}]}
        with self.assertRaises(ValueError):verify(source,{'blocks':[]})
    def test_changed_code_fails(self):
        source={'blocks':[{'t':'CodeBlock','c':[['',[],[]],'x = 1']}]}
        other=copy.deepcopy(source);other['blocks'][0]['c'][1]='x = 2'
        with self.assertRaises(ValueError):verify(source,other)
    def test_unchanged_content(self):
        source={'blocks':[{'t':'Para','c':[{'t':'Str','c':'正文'}]}]}
        self.assertEqual(verify(source,copy.deepcopy(source))['status'],'passed')
