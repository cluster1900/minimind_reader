"""Create a renamed OFL font subset for the EPUB's actual text."""
from pathlib import Path
import hashlib,json
from fontTools import subset
from fontTools.ttLib import TTFont
ROOT=Path(__file__).resolve().parents[1]
def main():
    cfg=json.loads((ROOT/'publishing/book.yaml').read_text())
    paths=[ROOT/p for e in cfg['frontmatter']+cfg['chapters']+cfg['backmatter'] for p in e['files']]
    text=''.join(p.read_text() for p in paths)+json.dumps(cfg,ensure_ascii=False)+(ROOT/'publishing/math-descriptions.json').read_text()+''.join(chr(i) for i in range(32,127))
    source=ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf';font=TTFont(source)
    options=subset.Options();options.name_IDs=['*'];options.name_languages=['*'];options.layout_features=['*']
    worker=subset.Subsetter(options=options);worker.populate(text=text);worker.subset(font)
    names={1:'MiniMind Book CJK',2:'Regular',3:'MiniMindBookCJK-20260908',4:'MiniMind Book CJK',6:'MiniMindBookCJK',16:'MiniMind Book CJK',17:'Regular'}
    for record in font['name'].names:
        if record.nameID in names:record.string=names[record.nameID].encode(record.getEncoding())
    if 'CFF ' in font:
        cff=font['CFF '].cff;cff.fontNames[0]='MiniMindBookCJK';cff.topDictIndex[0].FullName='MiniMind Book CJK';cff.topDictIndex[0].FamilyName='MiniMind Book CJK'
    output=ROOT/'assets/fonts/MiniMindBookCJK.otf';font.save(output)
    report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size,'glyphs':len(font.getGlyphOrder()),'license':'SIL OFL 1.1; renamed subset, original copyright retained'}
    (ROOT/'meta/font-subset.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
