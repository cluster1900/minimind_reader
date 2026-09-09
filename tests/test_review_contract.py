import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from source_snapshot import SourceSnapshot
from check_evidence import inventory,validate_review
from check_book import check_metadata


class SourceHistory(unittest.TestCase):
    def test_optimization_cannot_disable_artifact_checks(self):
        for name in ['check_epub','check_pdf']:
            code=f'import sys; sys.path.insert(0,{str(ROOT/"scripts")!r}); from {name} import check; check(None,None)'
            result=subprocess.run([sys.executable,'-O','-c',code],text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('without -O or PYTHONOPTIMIZE',result.stderr)

    def test_cited_commit_survives_different_head_and_dirty_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            repo=Path(directory)/'minimind';repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),'-c','user.name=Book Test',
                     '-c','user.email=book-test@example.invalid',*args],stderr=subprocess.DEVNULL,text=True).strip()
            git('init');source=repo/'model.py';source.write_text('def cited(): return 1\n')
            git('add','model.py');git('commit','-m','fixed source');commit=git('rev-parse','HEAD')
            source.write_text('def renamed(): return 2\n');git('add','model.py');git('commit','-m','new head')
            source.write_text('dirty workspace, not Python')
            snap=SourceSnapshot(directory)
            self.assertEqual(snap.read('minimind',commit,'model.py'),b'def cited(): return 1\n')
            self.assertEqual(source.read_text(),'dirty workspace, not Python')
            with self.assertRaises(ValueError):snap.read('minimind',commit,'missing.py')
            with self.assertRaises(ValueError):snap.read('minimind','master','model.py')


class ReviewedContent(unittest.TestCase):
    def setUp(self):
        self.doc={'id':'ch01','path':'book/ch01.md','body':'reviewed explanation',
          'ast':{'blocks':[{'t':'Para','c':[{'t':'Math','c':[{'t':'DisplayMath'},'y=x^2']} ]},
                          {'t':'CodeBlock','c':[['', ['mermaid'],[]],'flowchart TD\nA-->B']}]}}
        self.sources=[{'repository':'minimind','commit':'a'*40,'path':'model.py','sha256':'fixed'}]
        self.review={'status':'passed','record':'review.md','reviewed_at':'2026-09-09',
                     'documents':inventory([self.doc]),'source_files':self.sources}

    def test_changed_equation_without_new_review_fails(self):
        doc=copy.deepcopy(self.doc);doc['ast']['blocks'][0]['c'][0]['c'][1]='y=x^3'
        with self.assertRaises(ValueError):validate_review(inventory([doc]),self.sources,self.review)

    def test_reversed_diagram_arrow_fails(self):
        doc=copy.deepcopy(self.doc);doc['ast']['blocks'][1]['c'][1]='flowchart TD\nB-->A'
        with self.assertRaises(ValueError):validate_review(inventory([doc]),self.sources,self.review)

    def test_changed_explanation_and_source_fail(self):
        doc=copy.deepcopy(self.doc);doc['body']='incorrect changed explanation'
        with self.assertRaises(ValueError):validate_review(inventory([doc]),self.sources,self.review)
        source=copy.deepcopy(self.sources);source[0]['sha256']='different'
        with self.assertRaises(ValueError):validate_review(inventory([self.doc]),source,self.review)

    def test_unchanged_review_passes(self):
        validate_review(inventory([self.doc]),self.sources,self.review)

    def test_unreviewed_document_added_fails(self):
        doc=copy.deepcopy(self.doc);doc['id']='ch02'
        with self.assertRaises(ValueError):validate_review(inventory([self.doc,doc]),self.sources,self.review)


class Metadata(unittest.TestCase):
    def test_dependency_cycle_rejected(self):
        cfg={'chapters':[{'id':'ch01','files':['missing01']},{'id':'ch02','files':['missing02']}]}
        docs=[]
        for cid,other in [('ch01','ch02'),('ch02','ch01')]:
            docs.append({'id':cid,'path':cid,'meta':{'chapter_id':cid,'title':cid,'status':'draft',
              'audience':'core','prerequisites':[other],'sources':[{'repository':'jingyaogong/minimind',
              'commit':'a'*40,'dirty':False}],'verified_at':None,'validation':{'static':'pending','cpu':'not_run',
              'gpu':'not_required'},'validation_records':[],'scope':'test'}})
        with self.assertRaisesRegex(ValueError,'Cyclic'):check_metadata(cfg,docs,'preview')

    def test_unknown_prerequisite_rejected(self):
        cfg={'chapters':[{'id':'ch01','files':['missing']}]}
        doc={'id':'ch01','path':'ch01','meta':{'title':'test','chapter_id':'ch01','status':'draft',
          'audience':'core','prerequisites':['ch99'],'sources':[],'verified_at':None,'validation':{},
          'validation_records':[],'scope':'test'}}
        with self.assertRaisesRegex(ValueError,'Unknown prerequisite'):check_metadata(cfg,[doc],'preview')

if __name__=='__main__':unittest.main()
