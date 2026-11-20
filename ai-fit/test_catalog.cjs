const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(__dirname+'/static/app.js','utf8');
const functions=source.slice(source.indexOf('async function refresh(){'),source.indexOf('function selected(){'));
async function check({user,store,query='',expected,previous}){
 const location={search:query,href:'http://localhost/'+query};let called;
 const c={URL,URLSearchParams,location,history:{replaceState:()=>{}},bootstrap:previous,state:{store,selected:null},render:()=>{},api:async path=>{if(path==='/bootstrap')return {user,stores:[{id:'demo'},{id:'mine'},{id:'other'}]};called=path;return {products:[{id:'shirt'}],store:{id:expected}}}};
 vm.createContext(c);vm.runInContext(functions,c);await c.refresh();assert.equal(c.state.store,expected);assert.equal(called,'/products?store='+expected);assert.equal(c.state.selected,'shirt');
 c.rememberStore('other');assert.equal(c.state.store,'other');assert.equal(c.state.filter,'all');assert.equal(c.state.search,'');
}
(async()=>{
 await check({user:{id:'u',store:'mine'},store:null,expected:'mine'});
 await check({user:null,store:null,expected:'demo'});
 await check({user:{id:'u',store:'mine'},store:'other',query:'?store=other',expected:'other'});
 await check({user:{id:'u',store:'mine'},store:'demo',previous:{user:null},expected:'mine'});
 await check({user:{id:'u',store:'mine'},store:'deleted',expected:'mine'});
 console.log('PASS: owner catalog, guest catalog, explicit store links, login and invalid-store fallback.');
})().catch(e=>{console.error(e);process.exitCode=1});
