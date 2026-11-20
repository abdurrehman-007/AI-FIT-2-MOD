/* Present server progress without pretending that elapsed time is completed work. */
(function(root){
  const labels={queued:'Waiting for the fitting room',starting:'Starting your preview',validating:'Checking your photo',preparing:'Preparing your outfit',generating:'Creating your look',checking:'Checking the result',refining:'Refining your preview',saving:'Finishing your preview',complete:'Your preview is ready'};
  const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=v=>typeof v==='number'&&Number.isFinite(v)?v:null;
  const duration=n=>n>=60?`${Math.floor(n/60)}m ${Math.floor(n%60)}s`:`${Math.floor(n)}s`;
  const range=n=>`${Math.max(1,Math.floor(n*.8))}–${Math.max(3,Math.ceil(n*1.2))} sec`;
  function view(p,now){
    const active=p.status==='running',delta=active?Math.max(0,(now-p.observed)/1000):0;
    const elapsed=Math.max(0,(number(p.elapsed_seconds)??0)+delta);
    const remaining=number(p.remaining_seconds)===null?null:Math.max(0,p.remaining_seconds-delta);
    const complete=p.status==='complete';
    return {elapsed,remaining,percent:complete?100:Math.max(0,Math.min(99,Math.round(number(p.percent)??0))),
      title:complete?'Your look is ready':p.status==='failed'?'Preview not completed':'Creating your look',
      stage:labels[p.stage]||labels.starting,
      eta:complete?'Complete':p.status==='failed'?'Please try again':remaining===null?'Calculating estimate…':remaining<=0?'Updating estimate…':`About ${range(remaining)} remaining`,
      total:active&&number(p.estimated_total_seconds)!==null&&remaining!==0?`Estimated total: ${range(p.estimated_total_seconds)}`:'',
      overdue:active&&remaining===0};
  }
  function markup(p,now=performance.now()){
    if(!p)return '';
    const v=view(p,now),terminal=p.status!=='running';
    const index=p.status==='complete'?5:({queued:-1,starting:-1,validating:0,preparing:1,generating:2,checking:3,refining:3,saving:3}[p.stage]??-1);
    return `<section class="generation-timeline ${terminal?'is-finished':''} ${p.status==='failed'?'has-failed':''}" aria-label="Preview progress">
      <p class="generation-eyebrow">${terminal?'GENERATION STATUS':'YOUR LOOK IS ON ITS WAY'}</p>
      <h3>${escape(v.title)}</h3><p class="generation-garment">${escape(p.garment)}</p>
      <div class="generation-percent"><strong>${v.percent}<small>%</small></strong><span>${terminal&&p.status==='complete'?'Completed':'Estimated progress'}</span></div>
      <div class="generation-track" role="progressbar" aria-label="Estimated preview progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${v.percent}"><span style="width:${v.percent}%"></span></div>
      <div class="generation-times"><span>Elapsed <b>${duration(v.elapsed)}</b></span><span>${escape(v.eta)}</span></div>
      ${v.total?`<p class="generation-total">${escape(v.total)}</p>`:''}
      <p class="generation-stage">${p.status==='failed'?'The preview could not be completed.':escape(v.stage)}</p>
      ${terminal?'':`<ol class="generation-phases">${['Photo check','Prepare outfit','Create preview','Quality check','Ready'].map((label,i)=>`<li class="${i<index?'done':i===index?'current':''}"><span>${i<index?'✓':i+1}</span>${label}</li>`).join('')}</ol>
      <p class="generation-note">${v.overdue?'This is taking longer than estimated. Processing is still in progress.':'Time and percentage are estimates. Extra refinement can extend the time.'}</p>`}
    </section>`;
  }
  const api={view,markup};root.TryOnProgress=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
