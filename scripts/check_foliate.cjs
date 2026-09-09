// Independent archived-EPUB parser and paginator, pinned by commit.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process'),puppeteer=require('puppeteer');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=path.join(root,'build/release/foliate');fs.mkdirSync(out,{recursive:true});
 const revision=execFileSync('git',['-C',path.join(root,'.tools/foliate-js'),'rev-parse','HEAD'],{encoding:'utf8'}).trim();
 if(revision!=='78914aef4466eb960965702401634c2cb348e9b1')throw Error('Reader revision changed');
 const server=http.createServer((req,res)=>{
  const file=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));
  if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end()}
  const types={'.js':'text/javascript','.html':'text/html','.epub':'application/epub+zip'};
  fs.readFile(file,(e,data)=>{if(e){res.writeHead(404);return res.end()}res.setHeader('Content-Type',types[path.extname(file)]||'application/octet-stream');res.end(data)});
 });
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const browser=await puppeteer.launch({...JSON.parse(fs.readFileSync(path.join(root,'publishing/puppeteer.json'))),headless:true});
 const report={reader:'Foliate.js',revision,sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'build/release/minimind-source-guide.epub'))).digest('hex'),checks:[]};
 try{
  for(const [width,size,theme] of [[390,16,'light'],[390,22,'dark'],[1000,16,'dark'],[1000,22,'light']]){
   const page=await browser.newPage();await page.setViewport({width,height:900});
   await page.emulateMediaFeatures([{name:'prefers-color-scheme',value:theme}]);
   await page.goto(`http://127.0.0.1:${server.address().port}/publishing/foliate-reader.html`,{waitUntil:'networkidle0'});
   const toc=await page.evaluate(()=>ready);if(toc.length!==26)throw Error('Wrong TOC');
   await page.evaluate(size=>reader.renderer.setStyles(`body{font-size:${size}px !important}`),size);
   for(const index of [0,3,12,17,18,20,25]){
    await page.evaluate(index=>reader.goTo(nav[index].href),index);
    await page.evaluate(()=>Promise.all(reader.renderer.getContents().map(async({doc})=>{
     await doc.fonts.ready;await Promise.all([...doc.images].map(x=>x.decode().catch(()=>{})));
    })));
    const state=await page.evaluate(()=>reader.renderer.getContents().map(({doc,index})=>({index,
     heading:doc.querySelector('h1')?.textContent,images:doc.images.length,
     broken:[...doc.images].filter(x=>!x.complete||!x.naturalWidth).length,
     fontSize:getComputedStyle(doc.body).fontSize,
     fonts:[...doc.fonts].map(f=>({family:f.family,status:f.status}))})));
    const normal=s=>s.replace(/\s/g,'');
    if(!state.some(x=>normal(x.heading||'')===normal(toc[index].label)))throw Error('Wrong chapter '+index+JSON.stringify(state));
    if(state.some(x=>x.broken||!x.fonts.some(f=>f.family.includes('MiniMind Book CJK')&&f.status==='loaded')))throw Error('Missing font/image');
    report.checks.push({index,width,size,theme,state});
    const stem=`ch${index}-${width}-${size}-${theme}`;
    await page.screenshot({path:path.join(out,stem+'.png')});
    await page.evaluate(()=>reader.next());await page.screenshot({path:path.join(out,stem+'-next.png')});
    const mathFound=await page.evaluate(async()=>{
     const current=reader.renderer.getContents()[0];if(!current.doc.querySelector('img.math-display'))return false;
     await reader.renderer.goTo({index:current.index,anchor:doc=>doc.querySelector('img.math-display')});return true;
    });
    if(mathFound)await page.screenshot({path:path.join(out,stem+'-formula.png')});
   }
   await page.close();
  }
  report.status='passed';fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log('Foliate reader checks passed:',report.checks.length);
 }finally{await browser.close();await new Promise(r=>server.close(r))}
})().catch(e=>{console.error(e);process.exitCode=1});
