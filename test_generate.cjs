const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(__dirname+'/static/app.js','utf8');
const generate=source.slice(source.indexOf('async function generate(){'),source.indexOf('async function shareResult'));
async function check(change,expected){
 let resolve;
 const context={state:{busy:false,photo:{id:'photo-1'},selected:'shirt-1',view:'fit',result:null},bootstrap:{user:{id:'user-1'}},selected:()=>({image:'garment'}),renderFit:()=>{},authDialog:()=>{},toast:()=>{},api:()=>new Promise(r=>resolve=r)};
 vm.createContext(context);vm.runInContext(generate,context);
 const pending=context.generate();change(context);resolve({url:'/result'});await pending;
 assert.equal(Boolean(context.state.result),expected);assert.equal(context.state.busy,false);
}
(async()=>{
 await check(()=>{},true);
 await check(c=>c.state.selected='shirt-2',false);
 await check(c=>c.state.photo={id:'photo-2'},false);
 await check(c=>c.bootstrap.user=null,false);
 console.log('PASS: valid result retained; stale garment, photo and account results rejected.');
})().catch(e=>{console.error(e);process.exitCode=1});
