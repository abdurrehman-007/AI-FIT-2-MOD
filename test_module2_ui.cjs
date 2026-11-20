const {chromium}=require('C:/Users/Night-Fury/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {spawn}=require('node:child_process');const assert=require('node:assert/strict');const fs=require('node:fs');
const python=process.env.AIFIT_TEST_PYTHON||'E:/Comsats/8th Semester/Fyp 2/Virtual-Try-On-main/.venv/Scripts/python.exe';
(async()=>{const server=spawn(python,['-B','scripts/module2_ui_server.py'],{windowsHide:true,stdio:'ignore'});let browser;try{
 for(let i=0;i<100;i++){try{if((await fetch('http://127.0.0.1:5012/api/bootstrap')).ok)break}catch{}await new Promise(r=>setTimeout(r,150));}
 browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:5012');await page.locator('[data-view="size"]').click();
 const data=await (await page.request.get('http://127.0.0.1:5012/api/products?store=demo')).json();
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
 let b=await (await page.request.get('http://127.0.0.1:5012/api/bootstrap')).json();
 let r=await page.request.post('http://127.0.0.1:5012/api/auth/register',{headers:{'X-CSRF-Token':b.csrf},data:{name:'UI tester',email:'ui-module2@example.test',password:'LongPass123!abc'}});assert.equal(r.status(),200);
 b=await (await page.request.get('http://127.0.0.1:5012/api/bootstrap')).json();r=await page.request.post('http://127.0.0.1:5012/api/store',{headers:{'X-CSRF-Token':b.csrf},data:{name:'UI test store'}});assert.equal(r.status(),200);
 await page.setViewportSize({width:1440,height:1100});await page.goto('http://127.0.0.1:5012');await page.locator('[data-view="inventory"]').click();await page.locator('#edit-chart').click();
 assert.equal(await page.locator('#chart-form h3').count(),2);assert.equal(await page.locator('#chart-form tbody tr td').first().innerText(),'XS');await page.locator('[name="upper-XS-chest"]').fill('33.5');await page.locator('#chart-form button').click();await page.locator('#modal').waitFor({state:'hidden'});await page.locator('#edit-chart').click();assert.equal(await page.locator('[name="upper-XS-chest"]').inputValue(),'33.5');await page.screenshot({path:'reports/module2-ui-charts.png',fullPage:true});
 assert.deepEqual(errors,[]);fs.writeFileSync('reports/module2-ui-check.json',JSON.stringify({status:'passed',categories:['upper','lower'],mobile_width:390,chart_edit_persisted:true,browser_errors:errors},null,2));console.log('PASS: two category forms/results, mobile overflow, chart editor persistence and zero JS errors.');
 }finally{if(browser)await browser.close();server.kill();}})().catch(e=>{console.error(e);process.exitCode=1;});
