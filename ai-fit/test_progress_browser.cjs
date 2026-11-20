const path=require('node:path'),fs=require('node:fs'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const {chromium}=require(process.env.AIFIT_PLAYWRIGHT||path.join(path.dirname(path.dirname(process.execPath)),'node_modules/playwright'));
const real=process.env.AIFIT_TEST_REAL_GPU==='1',base='http://127.0.0.1:5017';
(async()=>{
 if(real){const health=await fetch('http://127.0.0.1:8001/health');assert.equal(health.status,200,'GPU must be ready before live browser verification')}
 const server=spawn('.gpu-runtime/Scripts/python.exe',['-B','scripts/progress_ui_server.py'],{windowsHide:true,stdio:'ignore'});
 let browser;const errors=[],snapshots=[],updates=[];let currentSource='';
 try{
  let ready=false;for(let i=0;i<100;i++){try{if((await fetch(base+'/api/bootstrap')).ok){ready=true;break}}catch{}await new Promise(r=>setTimeout(r,150))}assert.ok(ready,'fixture server starts');
  browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1100}});page.on('pageerror',e=>errors.push(e.message));
  page.on('response',async response=>{if(response.url().includes('/api/tryon-progress/')&&response.status()===200){const source=currentSource;try{updates.push({source,...await response.json()})}catch{}}});
  let b=await (await page.request.get(base+'/api/bootstrap')).json();
  const post=async(url,data)=>{b=await(await page.request.get(base+'/api/bootstrap')).json();return page.request.post(base+'/api'+url,{headers:{'X-CSRF-Token':b.csrf},data})};
  assert.equal((await post('/auth/register',{name:'Progress test',email:'progress-ui@example.test',password:'LongPassword123!abc'})).status(),200);
  assert.equal((await post('/store',{name:'Disposable progress fixture'})).status(),200);
  const garment=await (await page.request.get(base+'/test-garment')).body();
  b=await(await page.request.get(base+'/api/bootstrap')).json();
  const upload=await page.request.post(base+'/api/uploads',{headers:{'X-CSRF-Token':b.csrf},multipart:{kind:'garment',consent:'true',file:{name:'garment.png',mimeType:'image/png',buffer:garment}}});assert.equal(upload.status(),200);
  assert.equal((await post('/products',{name:'Forest green test shirt',sku:'PROGRESS-1',category:'upper',price:100,stock:5,sizes:['M'],image:(await upload.json()).id})).status(),200);
  await page.goto(base);await page.locator('[data-product]').first().waitFor();
  await page.route('**/api/uploads',async route=>{await new Promise(resolve=>setTimeout(resolve,350));await route.continue()});
  const cases=real?(process.env.AIFIT_PROGRESS_CASES||'covered,bare').split(','):['mock'];
  if(real)assert.ok(cases.every(c=>['bare','covered'].includes(c)));
  fs.mkdirSync('instance/progress-ui-check',{recursive:true});
  for(const source of cases){
   currentSource=source;
   const person=await(await page.request.get(base+'/test-fixture'+(source==='bare'?'?bare=1':''))).body();
   const previousPhoto=await page.locator('#drop-zone img').count()?await page.locator('#drop-zone img').getAttribute('src'):null;
   await page.locator('#photo-consent').check();await page.locator('#photo-input').setInputFiles({name:'person.png',mimeType:'image/png',buffer:person});
   assert.equal(await page.locator('#generate').isDisabled(),true,'generation is disabled while the new upload is pending');
   await page.waitForFunction(previous=>{const image=document.querySelector('#drop-zone img');return image&&image.getAttribute('src')!==previous},previousPhoto);
   await page.locator('#remove-photo').waitFor();
   await page.locator('#generate').click();assert.equal(await page.locator('#generate').isDisabled(),true);
   await page.waitForFunction(()=>{const error=document.querySelector('.inline-error[role="alert"]');const b=document.querySelector('[role="progressbar"]');return error?.textContent||(b&&Number(b.getAttribute('aria-valuenow'))>=30&&Number(b.getAttribute('aria-valuenow'))<80)},{},{timeout:180000});
   assert.equal(await page.locator('.inline-error[role="alert"]').count(),0,'generation should succeed in this fixture');
   let percent=Number(await page.locator('[role="progressbar"]').getAttribute('aria-valuenow'));assert.ok(percent<100);
   await page.screenshot({path:`instance/progress-ui-check/${source}-desktop.png`,fullPage:true});
   await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   await page.screenshot({path:`instance/progress-ui-check/${source}-mobile.png`,fullPage:true});
   await page.locator('#share-result').waitFor({timeout:240000});assert.equal(await page.locator('[role="progressbar"]').getAttribute('aria-valuenow'),'100');
   await page.screenshot({path:`instance/progress-ui-check/${source}-complete.png`,fullPage:true});
   assert.match(await page.locator('#generation-progress').innerText(),/Your look is ready/);
   assert.ok(updates.filter(u=>u.source===source&&u.stage==='generating').length>=1);
   const actual=await page.evaluate(()=>({elapsed_seconds:state.progress.elapsed_seconds,request_id:state.result.request_id}));
   const backend=real?await(await fetch('http://127.0.0.1:8001/health')).json():null;
   snapshots.push({source,active_percent:percent,completed_percent:100,...actual,backend,summary:await page.locator('#generation-progress').innerText()});await page.setViewportSize({width:1440,height:1100});
  }
  if(!real){
   await page.request.get(base+'/test-failure');await page.locator('#generate').click();
   await page.locator('.inline-error[role="alert"]').waitFor({timeout:20000});
   assert.match(await page.locator('#generation-progress').innerText(),/Preview not completed/);
   assert.ok(Number(await page.locator('[role="progressbar"]').getAttribute('aria-valuenow'))<100);
   assert.equal(await page.locator('#generate').isDisabled(),false);
   await page.locator('#generate').click();await page.locator('#logout').click();await page.locator('#signin').waitFor();
   assert.equal(await page.locator('[role="progressbar"]').count(),0);
   await new Promise(resolve=>setTimeout(resolve,13000));
   assert.equal(await page.locator('[role="progressbar"]').count(),0,'late responses do not restore signed-out progress');
   assert.equal(await page.locator('#drop-zone img').count(),0);
  }
  assert.deepEqual(errors,[]);fs.writeFileSync(`instance/progress-ui-check/${real?'real':'mock'}-results.json`,JSON.stringify({real_gpu:real,snapshots,updates,browser_errors:errors},null,2));
  console.log(JSON.stringify({passed:true,real_gpu:real,snapshots,browser_errors:errors}));
 }finally{if(browser)await browser.close();server.kill()}
})().catch(e=>{console.error(e);process.exitCode=1});
