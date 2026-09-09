import json
import subprocess
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from booklib import PANDOC,FORMAT,visit
from check_book import scan_math


class MarkdownInputs(unittest.TestCase):
    def parse(self,text):
        return json.loads(subprocess.check_output([PANDOC,'-f',FORMAT,'-t','json'],input=text,text=True))

    def test_pipe_token_inside_table(self):
        doc=self.parse('| Token | ID |\n|---|---|\n| `<\\|im_start\\|>` | 1 |')
        tokens=[n['c'][1] for n in visit(doc) if n['t']=='Code']
        self.assertEqual(tokens,['<|im_start|>'])
        self.assertEqual(len(next(n for n in visit(doc) if n['t']=='Table')['c'][2]),2)

    def test_math_inside_code_not_rendered(self):
        text='Inline $x_i$; code `$not_math$`.\n\n$$\ny=\\frac{1}{2}\n$$\n'
        self.assertEqual(scan_math(text),[(False,'x_i'),(True,'y=\\frac{1}{2}')])
        self.assertEqual(len([n for n in visit(self.parse(text)) if n['t']=='Math']),2)

    def test_unclosed_math_rejected(self):
        with self.assertRaises(ValueError):scan_math('Broken $formula')

    def test_footnote_is_content(self):
        d=self.parse('Text[^n].\n\n[^n]: Footnote retained.\n')
        self.assertEqual(len([n for n in visit(d) if n['t']=='Note']),1)


if __name__=='__main__':unittest.main()
