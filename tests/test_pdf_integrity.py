import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import pymupdf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_pdf import check,source_probes,normalized
EDITION=os.environ.get('BOOK_TEST_EDITION','preview')
STAGE=ROOT/'build'/EDITION
PDF=STAGE/('minimind-source-guide.pdf' if EDITION=='release' else 'minimind-source-guide-preview.pdf')


class PdfIntegrity(unittest.TestCase):
    def test_original(self):
        check(PDF,STAGE)

    def test_missing_content_page(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'broken.pdf'
            with pymupdf.open(PDF) as doc:
                texts=[normalized(page.get_text()) for page in doc]
                ast=json.loads((STAGE/'source.json').read_text())
                selected=None
                for probe in source_probes(ast['blocks']):
                    hits=[i for i,text in enumerate(texts) if len(probe)>=30 and probe in text]
                    if len(hits)==1:selected=hits[0];break
                self.assertIsNotNone(selected,'Need a unique source paragraph, not a fixed page number')
                doc.delete_page(selected);doc.save(out)
            with self.assertRaises(AssertionError),contextlib.redirect_stdout(io.StringIO()):
                check(out,STAGE)

    def test_reordered_chapter_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'reordered.pdf'
            with pymupdf.open(PDF) as doc:
                chapters=[page-1 for level,title,page in doc.get_toc() if level==1]
                self.assertGreater(len(chapters),2)
                doc.move_page(chapters[2],chapters[1]);doc.save(out)
            with self.assertRaises(AssertionError),contextlib.redirect_stdout(io.StringIO()):
                check(out,STAGE)


if __name__=='__main__':unittest.main()
