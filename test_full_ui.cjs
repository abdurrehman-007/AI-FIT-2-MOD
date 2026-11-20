const {chromium}=require('C:/Users/Night-Fury/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {spawn}=require('node:child_process');const assert=require('node:assert/strict');const fs=require('node:fs');
const python=process.env.AIFIT_TEST_PYTHON||'E:/Comsats/8th Semester/Fyp 2/Virtual-Try-On-main/.venv/Scripts/python.exe';
(async()=>{const server=spawn(python,['-B','scripts/full_ui_server.py'],{windowsHide:true,stdio:'ignore'});let browser;try{
 for(let i=0;i<100;i++){try{if((await fetch('http://127.0.0.1:5013/api/bootstrap')).ok)break}catch{}await new Promise(r=>setTimeout(r,150));}
 browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:5013');await page.locator('[data-view="size"]').click();
 const data=await (await page.request.get('http://127.0.0.1:5013/api/products?store=demo')).json();
 for(const category of ['upper','lower']){
 await page.locator('#size-category').selectOption(category);assert.equal(await page.locator('#size-category option').count(),2);const product=data.products.find(p=>p.category===category);await page.locator('#size-product').selectOption(product.id);
 await page.locator('[name="height_feet"]').fill('5');await page.locator('[name="height_inches"]').fill('8');await page.locator('[name="weight"]').fill('65');await page.locator('[name="waist"]').fill('32');
 assert.equal(await page.locator('[name="chest"]').count(),category==='lower'?0:1);assert.equal(await page.locator('[name="hips"]').count(),category==='upper'?0:1);
 if(category!=='lower')await page.locator('[name="chest"]').fill('38');if(category!=='upper')await page.locator('[name="hips"]').fill('40');
 await page.locator('[name="consent"]').check();await page.locator('#size-form button').click();await page.locator('#size-output .big').waitFor();assert.match(await page.locator('#size-output').innerText(),/Fitting confidence unavailable/);
 await page.screenshot({path:`reports/module2-ui-${category}.png`,fullPage:true});
 }
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:'reports/module2-ui-mobile.png',fullPage:true});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 // Create a disposable retailer and edit only its own charts.
 let b=await (await page.request.get('http://127.0.0.1:5013/api/bootstrap')).json();
 let r=await page.request.post('http://127.0.0.1:5013/api/auth/register',{headers:{'X-CSRF-Token':b.csrf},data:{name:'UI tester',email:'ui-module2@example.test',password:'LongPass123!abc'}});assert.equal(r.status(),200);
 b=await (await page.request.get('http://127.0.0.1:5013/api/bootstrap')).json();r=await page.request.post('http://127.0.0.1:5013/api/store',{headers:{'X-CSRF-Token':b.csrf},data:{name:'UI test store'}});assert.equal(r.status(),200);
 await page.setViewportSize({width:1440,height:1100});await page.goto('http://127.0.0.1:5013');await page.locator('[data-view="inventory"]').click();await page.locator('#edit-chart').click();
 assert.equal(await page.locator('#chart-form h3').count(),2);assert.equal(await page.locator('#chart-form tbody tr td').first().innerText(),'XS');await page.locator('[name="upper-XS-chest"]').fill('33.5');await page.locator('#chart-form button').click();await page.locator('#modal').waitFor({state:'hidden'});await page.locator('#edit-chart').click();assert.equal(await page.locator('[name="upper-XS-chest"]').inputValue(),'33.5');await page.screenshot({path:'reports/module2-ui-charts.png',fullPage:true});

 // Reference loading is opt-in and does not save until retailer submits.
 await page.locator('#load-reference').click();await page.waitForFunction(()=>document.querySelector('[name="upper-XS-chest"]').value==='32');
 assert.equal(await page.locator('[name="lower-XS-hips"]').inputValue(),'35');
 await page.locator('#chart-form button').click();await page.locator('#modal').waitFor({state:'hidden'});
 await page.locator('[data-view="analytics"]').click();await page.getByRole('heading',{name:'Size model evaluation'}).waitFor();
 assert.match(await page.locator('#content').innerText(),/Unavailable:/);
 assert.match(await page.locator('#content').innerText(),/0 signed-in customers/);
 // Synthetic images test only application flow; no visual-generation quality claim.
 const fixture=await (await page.request.get('http://127.0.0.1:5013/test-fixture')).body();
 b=await (await page.request.get('http://127.0.0.1:5013/api/bootstrap')).json();
 r=await page.request.post('http://127.0.0.1:5013/api/uploads',{headers:{'X-CSRF-Token':b.csrf},multipart:{kind:'garment',consent:'true',file:{name:'shirt.jpg',mimeType:'image/jpeg',buffer:fixture}}});assert.equal(r.status(),200);const garment=(await r.json()).id;
 r=await page.request.post('http://127.0.0.1:5013/api/products',{headers:{'X-CSRF-Token':b.csrf},data:{name:'Browser fixture shirt',sku:'UI-1',category:'upper',price:2000,stock:5,sizes:['S','M','L'],image:garment}});assert.equal(r.status(),200);
 await page.goto('http://127.0.0.1:5013');await page.locator('[data-product]').first().waitFor();
 await page.locator('#photo-consent').check();await page.locator('#photo-input').setInputFiles({name:'person.jpg',mimeType:'image/jpeg',buffer:fixture});
 await page.locator('#remove-photo').waitFor();await page.locator('#generate').click();await page.locator('#share-result').waitFor();
 const downloadPromise=page.waitForEvent('download');await page.getByText('Save image',{exact:true}).click();const download=await downloadPromise;assert.equal(await download.failure(),null);
 await page.evaluate(()=>Object.defineProperty(navigator,'canShare',{configurable:true,value:()=>false}));await page.locator('#share-result').click();await page.waitForFunction(()=>document.querySelector('#toast').textContent.includes('Use Save image'));
 await page.evaluate(()=>{Object.defineProperty(navigator,'canShare',{configurable:true,value:()=>true});Object.defineProperty(navigator,'share',{configurable:true,value:async data=>{window.sharedFile=data.files[0].name}})});
 await page.locator('#share-result').click();await page.waitForFunction(()=>window.sharedFile==='ai-fit.jpg');
 await page.locator('[data-view="analytics"]').click();await page.getByRole('heading',{name:'Customer preferences',exact:true}).waitFor();assert.match(await page.locator('#content').innerText(),/1 signed-in customers/);
 await page.locator('#record-event').click();await page.locator('[name="reference"]').fill('UI-SALE-1');await page.locator('#event-form button').click();await page.locator('#modal').waitFor({state:'hidden'});
 await page.locator('#record-event').click();await page.locator('[name="kind"]').selectOption('return');await page.locator('[name="reference"]').fill('UI-SALE-1');await page.locator('#event-form button').click();await page.locator('#modal').waitFor({state:'hidden'});
 await page.getByRole('heading',{name:'Size model evaluation'}).waitFor();await page.locator('#toast').waitFor({state:'hidden'});await page.screenshot({path:'reports/final-analytics-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:'reports/final-analytics-mobile.png',fullPage:true});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 for(const view of ['fit','inventory','qr']){await page.locator(`[data-view="${view}"]`).click();await page.waitForTimeout(300);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,view+' overflow');}
 await page.locator('[data-view="analytics"]').click();await page.getByText('Sales',{exact:true}).first().waitFor();assert.equal(await page.locator('#analytics-mode').count(),0);assert.equal(await page.getByText('Try-on to purchase',{exact:true}).count(),0);await page.getByText('32.5%',{exact:true}).waitFor();await page.getByText('7.7%',{exact:true}).waitFor();await page.getByText('Sample sales metrics',{exact:true}).waitFor();assert.equal(await page.locator('#record-event').isDisabled(),false);assert.equal(await page.locator('svg[aria-label="Sample sales chronological trend"]').count(),1);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await page.screenshot({path:'reports/analytics-demo-mobile.png',fullPage:true});await page.setViewportSize({width:1440,height:1100});await page.screenshot({path:'reports/analytics-demo-desktop.png',fullPage:true});

 assert.deepEqual(errors,[]);fs.writeFileSync('reports/final-ui-check.json',JSON.stringify({status:'passed',categories:['upper','lower'],mobile_width:390,chart_edit_persisted:true,verified_reference_opt_in:true,upload_generate_download:true,gpu_in_browser:'mocked',share_api:'stubbed supported and unsupported branches; native OS sheet not tested',analytics_sales_returns:true,browser_errors:errors},null,2));console.log('PASS: all-module browser flows, optional reference, upload/generate/save/share contracts, analytics, sales/returns and desktop/mobile layouts.');
 }finally{if(browser)await browser.close();server.kill();}})().catch(e=>{console.error(e);process.exitCode=1;});
