"""Mutation checks operate on the actual generated EPUB, not a mocked book."""
import contextlib
import io
import os
import re
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_epub import check


class EpubIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        edition=os.environ.get('BOOK_TEST_EDITION','preview')
        cls.stage=ROOT/'build'/edition
        cls.original=cls.stage/('minimind-source-guide.epub' if edition=='release' else 'minimind-source-guide-preview.epub')
        if not cls.original.exists():raise RuntimeError('Build the smoke EPUB before running integrity tests')

    def mutated(self,mutation):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'broken.epub'
            changed=False
            with zipfile.ZipFile(self.original) as source,zipfile.ZipFile(output,'w') as target:
                for info in source.infolist():
                    data=source.read(info.filename)
                    new=mutation(info.filename,data)
                    if new!=data:changed=True
                    if new is not None:target.writestr(info,new)
            self.assertTrue(changed,'Mutation did not touch the real artifact')
            with self.assertRaises((AssertionError,KeyError)):
                with contextlib.redirect_stdout(io.StringIO()):check(output,self.stage)

    def test_original(self):
        check(self.original,self.stage)

    def test_changed_prose(self):
        self.mutated(lambda name,data:data.replace('这些数只是词表里的编号'.encode(),'这些数就是概率'.encode()) if name.endswith('.xhtml') else data)

    def test_deleted_diagram(self):
        self.mutated(lambda name,data:None if name.endswith('.svg') and b'data-mermaid-b64' in data else data)

    def test_duplicate_chapter_in_spine(self):
        # Duplicate a real chapter in the reading order without changing its XHTML.
        # A ZIP with every resource present must still be rejected in this case.
        def duplicate(name,data):
            if not name.endswith('.opf'):return data
            return re.sub(rb'(<itemref\b[^>]*idref="ch\d+_xhtml"[^>]*/>)',
                          lambda match:match.group(1)+match.group(1),data,count=1)
        self.mutated(duplicate)

    def test_missing_image_target(self):
        self.mutated(lambda name,data:data.replace(b'src="../media/',b'src="../missing/') if name.endswith('.xhtml') else data)

    def test_changed_formula_resource(self):
        self.mutated(lambda name,data:data.replace(b'data-display="block"',b'data-display="inline"') if name.endswith('.svg') and b'data-tex-b64' in data else data)

    def test_changed_footnote(self):
        # Footnotes are exercised by the retained layout fixture, not invented
        # in the release manuscript merely to make this test run.
        stage,original=self.stage,self.original
        self.stage=ROOT/'build/preview';self.original=self.stage/'minimind-source-guide-preview.epub'
        try:self.mutated(lambda name,data:data.replace('脚注文本需要保留'.encode(),'脚注已被错误改写'.encode()) if name.endswith('.xhtml') else data)
        finally:self.stage,self.original=stage,original

    def test_changed_authored_link(self):
        self.mutated(lambda name,data:data.replace(b'https://github.com/jingyaogong/minimind/blob/',b'https://example.invalid/minimind/blob/') if name.endswith('.xhtml') else data)

    def test_changed_alt_text(self):
        self.mutated(lambda name,data:data.replace(b'alt="',b'alt="WRONG ',1) if name.endswith('.xhtml') and b'data-tex-b64' in data else data)

    def test_glyph_corruption_without_metadata_change(self):
        if self.stage.name!='release':self.skipTest('Byte manifest required: run against release candidate')
        self.mutated(lambda name,data:re.sub(b' d="',b' d="M0 0 ',data,count=1) if name.endswith('.svg') and b'data-tex-b64' in data else data)


if __name__=='__main__':unittest.main()
