// Inspect final EPUB XHTML in Chrome. This is not a substitute for two EPUB readers.
const fs=require('node:fs');
const crypto=require('node:crypto');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {execFileSync}=require('node:child_process');
const puppeteer=require('puppeteer');

(async()=>{
  const root=path.resolve(__dirname,'..');
  const edition=process.argv[2]||'preview';if(!['preview','release'].includes(edition))throw Error('Unknown edition');
  const stage=path.join(root,'build',edition);
  const filename=edition==='release'?'minimind-source-guide.epub':'minimind-source-guide-preview.epub';
  const digest=()=>crypto.createHash('sha256').update(fs.readFileSync(path.join(stage,filename))).digest('hex');
  const epubHash=digest();
  const output=path.join(stage,'display');fs.mkdirSync(output,{recursive:true});
  const files=JSON.parse(execFileSync(path.join(root,'.venv/bin/python'),['-c',
    'import zipfile,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); z=zipfile.ZipFile(p/sys.argv[2]); z.extractall(p/"epub-unpacked"); print(json.dumps([n for n in z.namelist() if n.startswith("EPUB/text/ch") and n.endswith(".xhtml")]))',stage,filename
  ],{cwd:root,encoding:'utf8'}));
  const config=JSON.parse(fs.readFileSync(path.join(root,'publishing/puppeteer.json')));
  const browser=await puppeteer.launch({...config,headless:true});
  const report=[];
  try{
    const folder=path.join(stage,'epub-unpacked');
    for(const relative of files){
      const name=path.basename(relative);
      for(const [width,scheme] of [[390,'light'],[900,'light'],[390,'dark']]){
        const page=await browser.newPage();await page.setViewport({width,height:1000});
        await page.emulateMediaFeatures([{name:'prefers-color-scheme',value:scheme}]);
        await page.goto(pathToFileURL(path.join(folder,relative)).href,{waitUntil:'networkidle0'});
        await page.addStyleTag({content:'body{font-size:20px;}'});
        await page.evaluate(()=>document.fonts.ready);
        const result=await page.evaluate(()=>({
          width:innerWidth,contentWidth:document.documentElement.scrollWidth,
          images:[...document.images].length,
          broken:[...document.images].filter(x=>!x.complete||x.naturalWidth===0).map(x=>x.src),
          overflow:[...document.querySelectorAll('p,pre,table,img,h1,h2,code,td')].filter(x=>{
            const r=x.getBoundingClientRect();return r.left < -1 || r.right>innerWidth+1;
          }).map(x=>({tag:x.tagName,text:x.textContent.slice(0,70)}))
        }));
        report.push({file:name,scheme,...result});
        await page.screenshot({path:path.join(output,`${name}-${width}-${scheme}-top.png`)});
        const targets=await page.$$('img.diagram,img.math-display');
        for(let i=0;i<targets.length;i++){
          await targets[i].evaluate(x=>x.scrollIntoView({block:'center'}));
          await page.screenshot({path:path.join(output,`${name}-${width}-${scheme}-visual-${i+1}.png`)});
        }
        await page.close();
      }
    }
  }finally{await browser.close()}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
  const failures=report.filter(x=>x.broken.length||x.overflow.length||x.contentWidth>x.width+1);
  if(failures.length)throw new Error(JSON.stringify(failures));
  if(digest()!==epubHash)throw new Error('EPUB changed during display checks');
  fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify({status:'passed',epub_sha256:epubHash,
    report_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(output,'report.json'))).digest('hex')},null,2));
  process.stdout.write(`Display bounds passed: ${report.length} document/viewport combinations\n`);
})().catch(error=>{console.error(error);process.exitCode=1});
