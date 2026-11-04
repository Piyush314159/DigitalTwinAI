/* Recorded UCI hydraulic cycles. No synthetic or simulated measurements. */
'use strict';
let config, state, evaluation, reference=null;
let selectedSensor='PS1',busy=false,playing=false,replayTimer=null;
const $=id=>document.getElementById(id);
const icon=name=>`<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fixed=(v,d=3)=>v===null||v===undefined?'—':Number(v).toFixed(d);
const count=v=>v===null||v===undefined?'—':Number(v).toLocaleString('en-US');
const pages={
 overview:['OPERATIONS / OVERVIEW','Hydraulic condition monitoring','Recorded cycles. Measured sensor signals. Four component estimates.'],
 sensors:['OPERATIONS / SENSORS','Measured sensor channels','Original cycle statistics and one-second signal averages.'],
 validation:['MODEL / EVALUATION','Real-data model validation','Grouped research results for the active temporal model on measured UCI cycles.'],
 methodology:['DATASET / METHOD','Dataset and methodology','Measured inputs, documented labels and a reproducible evaluation protocol.']
};
async function api(path,options){const r=await fetch(path,options);const d=await r.json();if(!r.ok)throw Error(d.error||'The local service could not complete this request.');return d;}
function selectionQuery(){return new URLSearchParams({partition:state.partition,position:state.position}).toString();}
function updateExport(){const evaluationView=['validation','methodology'].includes(location.hash.slice(1));$('export-link').href=evaluationView?'/api/evaluation.json':state?`/api/snapshot.csv?${selectionQuery()}`:'/api/snapshot.csv';$('export-label').textContent=evaluationView?'Export evaluation':'Export cycle';}
function error(message){$('error-banner').textContent=message;$('error-banner').hidden=false;}
function changeView(view){
 if(!pages[view])return;
 const [eyebrow,title,description]=pages[view];$('page-eyebrow').textContent=eyebrow;$('page-title').textContent=title;$('page-description').textContent=description;
 document.querySelectorAll('.view').forEach(e=>e.hidden=e.id!==`view-${view}`);
 document.querySelectorAll('.nav-item').forEach(e=>{const active=e.dataset.view===view;e.classList.toggle('active',active);if(active)e.setAttribute('aria-current','page');else e.removeAttribute('aria-current');});
 const evaluationView=view==='validation'||view==='methodology';$('replay-controls').hidden=evaluationView;if(evaluationView)stopReplay();
 history.replaceState(null,'',`#${view}`);updateExport();window.scrollTo(0,0);document.title=`${title} — DigitalTwin`;
}
function updateControls(){
 const ended=state&&state.position===state.totalRecords-1;
 $('btn-reset').disabled=busy||!state;$('btn-step').disabled=busy||!state||ended;$('btn-play').disabled=busy||!state||(ended&&!playing);$('step-slider').disabled=busy||!state;$('partition-select').disabled=busy||!state;
 $('btn-play').innerHTML=icon(playing?'pause':'play')+`<span>${playing?'Pause':'Play records'}</span>`;
 if(!state)return;
 $('step-slider').max=state.totalRecords-1;$('step-slider').value=state.position;$('step-label').textContent=`${state.position+1} / ${state.totalRecords}`;$('cycle-label').textContent=`Cycle ${String(state.cycle).padStart(4,'0')}`;$('partition-select').value=state.partition;
}
function stopReplay(){playing=false;clearTimeout(replayTimer);updateControls();}
async function mutate(path,body){
 if(busy)return false;busy=true;updateControls();
 try{state=await api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({partition:state.partition,position:state.position,...body})});reference=null;$('error-banner').hidden=true;if(state.position===state.totalRecords-1)stopReplay();render();if($('show-reference').checked)await loadReference();$('live-announcement').textContent=`Measured cycle ${state.cycle}, ${state.summary.flaggedComponents??'unavailable'} components to review.`;return true;}
 catch(e){stopReplay();error(e.message+' Last loaded measurements are still displayed.');return false;}
 finally{busy=false;updateControls();}
}
async function replayNext(){if(!playing)return;const success=await mutate('/api/step');if(playing&&success)replayTimer=setTimeout(replayNext,1100);}
async function loadReference(){
 if(!$('show-reference').checked){reference=null;renderComponents();return;}
 try{const result=await api(`/api/reference?${selectionQuery()}`);if(result.cycle===state.cycle&&$('show-reference').checked){reference=result;renderComponents();}}catch(e){error('Recorded labels could not be loaded.');}
}
function metric(label,value,note,unit='',alert=false){return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value ${alert?'is-alert':''}">${value}${unit?`<small>${unit}</small>`:''}</div><div class="metric-note">${note}</div></div>`;}
function renderMetrics(){
 const partitionName=state.partition==='all'?'All recorded cycles':`Original linear ${state.partition} subset`;
 $('overview-metrics').innerHTML=metric('Components to review',count(state.summary.flaggedComponents),'Predicted non-optimal condition','/ 4',state.summary.flaggedComponents>0)+metric('Physical input channels','14','Pressure, flow, temperature and more')+metric('Measured cycle window','60','Complete acquisition used for inference','s')+metric('Selected records',count(state.totalRecords),partitionName);
 $('model-state').textContent=state.modelStatus==='trained_real_data'?'Real-data model loaded':state.modelStatus==='insufficient_data'?'Insufficient sensor data':'Model unavailable';$('model-kind').textContent=config.modelKind==='tcn'?'MULTIRATE TCN':'MODEL UNAVAILABLE';
 const quality=state.dataQuality;$('data-quality-note').hidden=quality?.complete!==false;$('data-quality-note').textContent=quality?.complete===false?`Missing or invalid readings: ${quality.invalidChannels.join(', ')}. Condition estimates are unavailable for this cycle; gaps remain visible in the signal.`:'';
}
function renderComponents(){
 const symbols=['circle','stations','overview','validation'];
 $('component-grid').innerHTML=config.components.map((component,i)=>{
 const result=state.conditions.find(c=>c.id===component.id);const recorded=reference?.cycle===state.cycle?reference.conditions.find(c=>c.id===component.id):null;
 return `<article class="component-card ${result?.flagged?'flagged':''}"><div class="component-top"><h3>${esc(component.name)}</h3>${icon(symbols[i])}</div><div class="condition-name">${result?esc(result.label):'Estimate unavailable'}</div><div class="condition-support"><span>Class support</span><strong>${result?fixed(result.support):'—'}</strong></div>${recorded?`<div class="reference-label">Recorded: ${esc(recorded.label)}</div>`:''}</article>`;
 }).join('');$('reference-note').hidden=!$('show-reference').checked;
}
function renderAlerts(){
 const alerts=state.conditions.filter(c=>c.flagged);$('alert-count').textContent=state.conditions.length?alerts.length:'—';
 if(!state.conditions.length){$('alerts-list').innerHTML=`<div class="empty-alerts">${icon('circle')}<strong>Estimates unavailable</strong><p>${state.modelStatus==='insufficient_data'?'A complete set of measured inputs is required.':'Train or restore the real-data model.'}</p></div>`;return;}
 if(!alerts.length){$('alerts-list').innerHTML=`<div class="empty-alerts">${icon('validation')}<strong>No non-optimal estimates</strong><p>All four predicted classes are optimal for this recorded cycle.</p></div>`;return;}
 $('alerts-list').innerHTML=alerts.map((c,i)=>`<div class="alert-row"><span class="alert-id">0${i+1}</span><span class="alert-text"><strong>${esc(c.name)}</strong><small>${esc(c.label)}</small></span><span class="alert-score">${fixed(c.support)}</span></div>`).join('');
}
function chart(sensorId){
 const sensor=config.sensors.find(s=>s.id===sensorId),o=state.observations.find(s=>s.id===sensorId);const values=o.trace;
 const measured=values.filter(Number.isFinite);if(!measured.length)return '<div class="empty-alerts"><strong>Signal unavailable</strong><p>No valid readings for this channel in the selected cycle.</p></div>';
 let min=Math.min(...measured),max=Math.max(...measured);const padding=Math.max((max-min)*.12,Math.abs(max)*.001,.001);min-=padding;max+=padding;
 const x=i=>48+i/59*580,y=v=>160-(v-min)/(max-min)*140;
 let grid='';[min,(min+max)/2,max].forEach(v=>grid+=`<line x1="48" x2="628" y1="${y(v)}" y2="${y(v)}" stroke="#edf0e9"/><text x="40" y="${y(v)+3}" text-anchor="end" class="chart-label">${fixed(v,1)}</text>`);
 [0,15,30,45,59].forEach(v=>grid+=`<text x="${x(v)}" y="183" text-anchor="middle" class="chart-label">${v}s</text>`);
 const segments=[];let points=[];values.forEach((v,i)=>{if(Number.isFinite(v)){points.push(`${x(i)},${y(v)}`);}else if(points.length){segments.push(points);points=[];}});if(points.length)segments.push(points);
 const lines=segments.map(segment=>segment.length===1?`<circle cx="${segment[0].split(',')[0]}" cy="${segment[0].split(',')[1]}" r="2" fill="#456d48"/>`:`<polyline points="${segment.join(' ')}" fill="none" stroke="#456d48" stroke-width="1.8" stroke-linejoin="round"/>`).join('');
 return `<svg viewBox="0 0 650 196" role="img" aria-label="Measured ${esc(sensor.name)} in ${esc(sensor.unit)}, cycle ${state.cycle}, seconds 0 through 59${measured.length<values.length?', missing readings shown as gaps':''}">${grid}${lines}</svg>`;
}
function selectSensor(id){if(!config.sensors.some(s=>s.id===id))return;selectedSensor=id;renderSensorDetails();renderSensorTable();}
function renderSensorDetails(){
 const s=config.sensors.find(s=>s.id===selectedSensor),o=state.observations.find(s=>s.id===selectedSensor);
 $('trace-select').value=selectedSensor;$('trace-caption').textContent=`${s.id} / ${s.name} · ${s.unit}`;$('signal-chart').innerHTML=chart(selectedSensor);$('native-rate').textContent=`${s.rate} Hz native sampling`;
 const detail=`<div class="panel-heading"><div><div class="station-code">MEASURED CHANNEL / ${esc(s.id)}</div><h2>${esc(s.name)}</h2><p>Cycle ${state.cycle} · ${s.rate*60} original samples</p></div></div><dl class="reading-grid">${[['Mean',o.mean],['Standard deviation',o.std],['Minimum',o.min],['Maximum',o.max]].map(([name,value])=>`<div class="reading"><dt>${name}</dt><dd>${fixed(value,2)}<small>${esc(s.unit)}</small></dd></div>`).join('')}</dl><div class="selected-score"><span>Native sample rate</span><strong>${s.rate} Hz</strong></div>`;
 $('sensor-detail').innerHTML=detail;$('sensor-detail-secondary').innerHTML=detail+`<div class="station-history"><p>Recorded signal / 60 seconds</p>${chart(selectedSensor)}</div>`;
}
function renderSensorTable(){
 const query=$('sensor-search').value.toLowerCase(),group=$('group-filter').value;
 const rows=config.sensors.filter(s=>(group==='all'||s.group===group)&&`${s.id} ${s.name}`.toLowerCase().includes(query));$('sensor-count').textContent=`${rows.length} / 14 physical input channels · original measurements`;
 $('sensor-table').innerHTML=rows.length?rows.map(s=>{const o=state.observations.find(c=>c.id===s.id);return `<tr class="${s.id===selectedSensor?'is-selected':''}"><td><button class="table-station" data-sensor="${s.id}"><strong>${s.id}</strong></button></td><td>${esc(s.name)}<div class="small-note">${esc(s.unit)}</div></td><td class="mono">${fixed(o.mean,2)}</td><td class="mono">${fixed(o.std,2)}</td><td class="mono">${fixed(o.min,2)} – ${fixed(o.max,2)}</td><td class="mono">${s.rate} Hz</td><td><button class="table-inspect" data-sensor="${s.id}" aria-label="Inspect ${s.id}">${icon('arrow')}</button></td></tr>`;}).join(''):'<tr><td colspan="7" class="table-empty">No sensors match these filters.</td></tr>';
}
function renderValidation(){
 const experiment=evaluation?.tcnExperiment,active=evaluation?.activeModel;
 if(!experiment||!active||active.kind!=='tcn'){
  $('validation-content').innerHTML='<div class="panel prose-panel"><h2>Evaluation unavailable</h2><p>Prepare the frozen TCN checkpoint and matching research report to see validation results.</p></div>';
  return;
 }
 const models=experiment.models,m=models.tcn.out_of_fold,legacy=evaluation.report;
 const rows=[['linear','Linear · refitted'],['tcn','Multirate TCN · active architecture']].map(([kind,name])=>{
  const item=models[kind],oof=item.out_of_fold;
  return `<tr><td>${name}</td><td class="mono">${count(item.parameters)}</td><td class="mono">${fixed(item.fold_mean_macro_f1)}</td><td class="mono">${fixed(oof.components.accumulator.macro_f1)}</td><td class="mono">${fixed(oof.all_components_correct*100,1)}%</td></tr>`;
 }).join('');
 $('validation-content').innerHTML=`<div class="validation-intro">${icon('validation')}<p>The active TCN was trained on all ${count(active.training_cycles)} UCI cycles. Scores below come from separate fold-fitted TCNs predicting held-out condition combinations within the same rig. They are research estimates for this architecture, not a test score for the full-data checkpoint or an independent-machine result.</p></div>
 <div class="metrics-strip validation-metrics">
 ${metric('Grouped-fold macro F1',fixed(models.tcn.fold_mean_macro_f1),'Mean across three grouped folds')}
 ${metric('Out-of-fold accuracy',fixed(m.mean_accuracy*100,1),'Mean across four component tasks','%')}
 ${metric('All four conditions correct',fixed(m.all_components_correct*100,1),'Out-of-fold exact match','%')}
 ${metric('Measured cycles',count(m.cycles),'One physical UCI test rig')}</div>
 <div class="validation-grid">
 <section class="panel"><div class="panel-heading"><div><h2>Primary multirate TCN</h2><p>Active dashboard estimates · complete 60-second cycles</p></div><span class="label-pill">ACTIVE</span></div><div class="comparison-body"><div class="comparison-row"><div class="comparison-label"><span>Grouped-fold macro F1</span><strong>${fixed(models.tcn.fold_mean_macro_f1)}</strong></div><div class="comparison-track"><div class="comparison-fill" style="width:${100*models.tcn.fold_mean_macro_f1}%"></div></div></div><p class="comparison-note">${count(active.parameters)} parameters · ${active.training_epochs} training epochs · ${count(active.training_cycles)} UCI development cycles. Pressure/power and flow sequences, temperature/vibration sequences, and 140 measured-cycle features.</p></div></section>
 <section class="panel"><div class="panel-heading"><div><h2>Per-component research results</h2><p>Out-of-fold predictions · original UCI condition classes</p></div></div><div class="table-scroll"><table class="data-table"><thead><tr><th>Component</th><th>Macro F1</th><th>Accuracy</th></tr></thead><tbody>${config.components.map(c=>`<tr><td>${esc(c.name)}</td><td class="mono">${fixed(m.components[c.id].macro_f1)}</td><td class="mono">${fixed(m.components[c.id].accuracy*100,1)}%</td></tr>`).join('')}</tbody></table></div></section>
 <section class="panel robustness-panel"><div class="panel-heading"><div><h2>Class-level research evaluation</h2><p>Rows: recorded condition · columns: fold estimate</p></div><label class="sr-only" for="matrix-component">Component confusion matrix</label><select id="matrix-component" class="partition-select">${config.components.map(c=>`<option value="${c.id}">${esc(c.name)}</option>`).join('')}</select></div><div class="table-scroll" id="confusion-matrix"></div></section>
 <section class="panel robustness-panel"><div class="panel-heading"><div><h2>Matched architecture comparison</h2><p>Measured cycles · same three grouped folds</p></div><span class="label-pill">RESEARCH</span></div><div class="table-scroll"><table class="data-table"><thead><tr><th>Model</th><th>Parameters</th><th>Fold mean macro F1</th><th>Accumulator F1</th><th>All four correct</th></tr></thead><tbody>${rows}</tbody></table></div><div class="robustness-foot">The TCN led on ${experiment.tcn_better_folds} of 3 folds. These results compare model types within the previously studied single rig. Independent-machine validation is still pending.</div></section></div>
 ${legacy?`<section class="panel robustness-panel"><div class="panel-heading"><div><h2>Earlier linear test</h2><p>Original 328-cycle held-out split · different evaluation protocol</p></div><span class="label-pill">HISTORICAL</span></div><div class="robustness-foot">The original linear checkpoint scored ${fixed(legacy.models.linear.test.mean_macro_f1)} mean macro F1 on its separate held-out split. This number is not directly comparable with the grouped-fold TCN score above. The earlier linear sensor-removal diagnostic remains in the evaluation export.</div></section>`:''}`;
 $('matrix-component').addEventListener('change',renderMatrix);
 renderMatrix();
 $('footer-model').textContent=`${active.checkpoint_sha256.slice(0,8).toUpperCase()} / TCN · MEASURED UCI DATA`;
 $('model-method').textContent=`Primary multirate TCN, ${count(active.parameters)} parameters. It uses complete measured cycles; class support is uncalibrated.`;
 $('split-method').innerHTML=`<div><strong>${count(active.training_cycles)}</strong><span>UCI training cycles</span></div><div><strong>3</strong><span>grouped research folds</span></div><div><strong>0</strong><span>independent machines tested</span></div>`;
}
function renderMatrix(){
 const component=config.components.find(c=>c.id===$('matrix-component').value);
 const m=evaluation.tcnExperiment.models.tcn.out_of_fold.components[component.id];
 $('confusion-matrix').innerHTML=`<table class="data-table class-matrix"><thead><tr><th>Recorded / estimated</th>${component.labels.map(label=>`<th>${esc(label)}</th>`).join('')}<th>Recall</th></tr></thead><tbody>${m.confusion.map((row,i)=>`<tr><td>${esc(component.labels[i])}</td>${row.map((n,j)=>`<td class="mono" ${i===j?'style="background:#edf3e9"':''}>${n}</td>`).join('')}<td class="mono">${fixed(m.per_class_recall[i]*100,1)}%</td></tr>`).join('')}</tbody></table>`;
}
function render(){if(!config||!state)return;updateControls();updateExport();renderMetrics();renderComponents();renderAlerts();renderSensorDetails();renderSensorTable();}

document.addEventListener('DOMContentLoaded',async()=>{
 busy=true;updateControls();try{config=await api('/api/config');$('partition-select').innerHTML=Object.entries(config.partitions).map(([name,n])=>`<option value="${name}">${name==='all'?'All recorded cycles':`Original linear ${name} subset`} (${n})</option>`).join('');$('trace-select').innerHTML=config.sensors.map(s=>`<option value="${s.id}">${s.id} / ${s.group}</option>`).join('');$('group-filter').insertAdjacentHTML('beforeend',[...new Set(config.sensors.map(s=>s.group))].map(g=>`<option>${g}</option>`).join(''));[state,evaluation]=await Promise.all([api('/api/state'),api('/api/validation')]);render();renderValidation();changeView(pages[location.hash.slice(1)]?location.hash.slice(1):'overview');}catch(e){error(e.message+' Check the local data and trained model.');}finally{busy=false;updateControls();}
});
document.addEventListener('click',event=>{const view=event.target.closest('[data-view]');if(view){changeView(view.dataset.view);return;}const sensor=event.target.closest('[data-sensor]');if(sensor)selectSensor(sensor.dataset.sensor);});
$('btn-step').addEventListener('click',()=>{stopReplay();mutate('/api/step');});
$('btn-reset').addEventListener('click',()=>{stopReplay();mutate('/api/reset');});
$('btn-play').addEventListener('click',()=>{if(playing)stopReplay();else{playing=true;updateControls();replayNext();}});
$('step-slider').addEventListener('change',e=>{const position=Number(e.target.value);stopReplay();mutate('/api/seek',{position});});
$('partition-select').addEventListener('change',e=>{const partition=e.target.value;stopReplay();mutate('/api/partition',{partition});});
$('show-reference').addEventListener('change',loadReference);
$('trace-select').addEventListener('change',e=>selectSensor(e.target.value));
$('sensor-search').addEventListener('input',renderSensorTable);$('group-filter').addEventListener('change',renderSensorTable);
window.addEventListener('hashchange',()=>{if(pages[location.hash.slice(1)])changeView(location.hash.slice(1));});
