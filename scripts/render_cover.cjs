const fs=require('node:fs');const path=require('node:path');const puppeteer=require('puppeteer');
(async()=>{
 const root=path.resolve(__dirname,'..');const config=JSON.parse(fs.readFileSync(path.join(root,'publishing/puppeteer.json')));
 const browser=await puppeteer.launch({...config,headless:true});
 try{const page=await browser.newPage();await page.setViewport({width:1400,height:2000,deviceScaleFactor:1});
 const svg=fs.readFileSync(path.join(root,'assets/cover.svg'),'utf8');
 await page.setContent('<style>@page{size:148mm 210mm;margin:0}html,body{margin:0;width:100%;height:100%;background:#142b3b}svg{width:100%;height:100%;display:block}</style>'+svg);
 await page.evaluate(()=>document.fonts.ready);
 await page.screenshot({path:path.join(root,'assets/cover.png')});
 await page.pdf({path:path.join(root,'assets/cover.pdf'),width:'148mm',height:'210mm',printBackground:true,margin:{top:0,bottom:0,left:0,right:0}});
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
