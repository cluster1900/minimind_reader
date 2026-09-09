// A real EPUB.js reader consumes the archived EPUB; this is separate from XHTML bounds checks.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),http=require('node:http');
const puppeteer=require('puppeteer');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=path.join(root,'build/release/epubjs');fs.mkdirSync(out,{recursive:true});
 const server=http.createServer((req,res)=>{const p=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));if(!p.startsWith(root+path.sep)){res.writeHead(403);return res.end()}
 const types={'.js':'text/javascript','.html':'text/html','.epub':'application/epub+zip'};
 fs.readFile(p,(e,data)=>{if(e){res.writeHead(404);return res.end()}res.setHeader('Content-Type',types[path.extname(p)]||'application/octet-stream');res.end(data)});
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const config=JSON.parse(fs.readFileSync(path.join(root,'publishing/puppeteer.json')));const browser=await puppeteer.launch({...config,headless:true});
 const report={reader:'EPUB.js 0.3.93',sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'build/release/minimind-source-guide.epub'))).digest('hex'),checks:[]};
 try{
 const page=await browser.newPage();await page.setViewport({width:1100,height:900});
 await page.goto(`http://127.0.0.1:${server.address().port}/publishing/reader.html`,{waitUntil:'networkidle0'});
 const toc=await page.evaluate(()=>ready);if(toc.length!==26)throw Error('Unexpected TOC size '+toc.length);
 report.toc=toc.map(x=>x.label);
 for(const [index,theme,size] of [[0,'light','100%'],[3,'light','130%'],[12,'dark','130%'],[18,'light','150%'],[25,'light','100%']]){
   await page.emulateMediaFeatures([{name:'prefers-color-scheme',value:theme}]);
   await page.evaluate(async({index,size})=>{reader.themes.fontSize(size);await reader.display(nav.toc[index].href)}, {index,size});
   await page.evaluate(()=>Promise.all(reader.getContents().map(c=>c.document.fonts.ready)));
   const state=await page.evaluate(()=>reader.getContents().map(c=>({heading:c.document.querySelector('h1')?.textContent,
    images:[...c.document.images].length,broken:[...c.document.images].filter(x=>!x.complete||!x.naturalWidth).length,
    fonts:[...c.document.fonts].map(f=>({family:f.family,status:f.status})),
    text:c.document.body.textContent.slice(0,100)})));
   if(!state.length||state.some(x=>x.broken))throw Error('Missing chapter/image '+JSON.stringify(state));
   if(state.some(x=>!x.fonts.some(f=>f.family.includes('MiniMind Book CJK')&&f.status==='loaded')))throw Error('Embedded CJK font did not load');
   const normal=s=>s.replace(/\s+/g,'');
   if(!state.some(x=>normal(x.heading||'')===normal(toc[index].label)))throw Error('Wrong chapter was displayed');
   report.checks.push({index,theme,size,state});await page.screenshot({path:path.join(out,`chapter-${index}-${theme}-${size}.png`)});
   await page.evaluate(()=>reader.next());await page.screenshot({path:path.join(out,`chapter-${index}-${theme}-${size}-next.png`)});
 }
 report.status='passed';fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log('EPUB.js reader checks passed');
 }finally{await browser.close();await new Promise(resolve=>server.close(resolve))}
})().catch(e=>{console.error(e);process.exitCode=1});
