#!/usr/bin/env python3
"""Record an explicitly completed content review; never invoked by the builder."""
import argparse
import datetime
import hashlib
import json
from booklib import ROOT
from check_book import check
from check_evidence import inventory


def main():
    parser=argparse.ArgumentParser(description='Only run after reviewing changed text, formulas and diagrams against sources.')
    parser.add_argument('--record',required=True,help='Existing Markdown record of the completed substantive review')
    args=parser.parse_args()
    record=(ROOT/args.record).resolve()
    if not record.is_file() or not record.is_relative_to(ROOT):
        raise ValueError('Provide an existing review record inside this project')
    result=check('release',None,require_review=False)
    review={'status':'passed','reviewed_at':datetime.datetime.now().astimezone().isoformat(),
      'record':str(record.relative_to(ROOT)),'documents':inventory(result['documents']),
      'source_files':result['source_files'],
      'math_descriptions_sha256':hashlib.sha256((ROOT/'publishing/math-descriptions.json').read_bytes()).hexdigest(),
      'limits':'Records the explicitly completed review and detects later changes; does not prove semantic correctness.'}
    (ROOT/'meta/content-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    print('Recorded reviewed content snapshot; rendering and artifact checks are still required')


if __name__=='__main__':main()
