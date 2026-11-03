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
 validation:['MODEL / EVALUATION','Real-data model validation','Held-out condition combinations from the physical UCI hydraulic test rig.'],
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
 $('overview-metrics').innerHTML=metric('Components to review',count(state.summary.flaggedComponents),'Predicted non-optimal condition','/ 4',state.summary.flaggedComponents>0)+metric('Physical input channels','14','Pressure, flow, temperature and more')+metric('Measured cycle window','60','Complete acquisition used for inference','s')+metric('Selected records',count(state.totalRecords),`${state.partition==='test'?'Held-out test':state.partition} partition`);
 $('model-state').textContent=state.modelStatus==='trained_real_data'?'Real-data model loaded':state.modelStatus==='insufficient_data'?'Insufficient sensor data':'Model unavailable';$('model-kind').textContent=({'linear':'LINEAR CLASSIFIER','mlp':'SMALL NEURAL NETWORK','linear_knn':'LINEAR + NEIGHBORS'})[config.modelKind]||'MODEL UNAVAILABLE';
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
 const r=evaluation?.report;if(!r){$('validation-content').innerHTML='<div class="panel prose-panel"><h2>Evaluation unavailable</h2><p>A matching real-data checkpoint and report are required.</p></div>';return;}
 const selected=r.models[r.selected_model],m=selected.test;
 const bar=(name,model)=>`<div class="comparison-row"><div class="comparison-label"><span>${name}${model===r.selected_model?' · deployed':''}</span><strong>${fixed(r.models[model].test.mean_macro_f1)}</strong></div><div class="comparison-track"><div class="comparison-fill ${model==='mlp'?'baseline':''}" style="width:${100*r.models[model].test.mean_macro_f1}%"></div></div></div>`;
 const cases=evaluation.robustness?.cases;
 $('validation-content').innerHTML=`<div class="validation-intro">${icon('validation')}<p>Measured UCI hydraulic cycles. Entire joint condition combinations are held out; ${r.split_counts.test} test cycles share no joint condition combination with training or validation. This remains a same-rig experiment.</p></div><div class="metrics-strip validation-metrics">${metric('Mean component macro F1',fixed(m.mean_macro_f1),'Equal average across four component tasks')}${metric('Mean component accuracy',fixed(m.mean_accuracy*100,1),'Average classification accuracy','%')}${metric('All four conditions correct',fixed(m.all_components_correct*100,1),'Exact four-component match per cycle','%')}${metric('Held-out test cycles',count(m.cycles),'29 unseen joint condition combinations')}</div><div class="validation-grid">
 <section class="panel"><div class="panel-heading"><div><h2>Linear and neural classifiers</h2><p>Test scores · deployment selected on validation only</p></div></div><div class="comparison-body">${bar('Linear classifier','linear')}${bar('Small neural network','mlp')}<p class="comparison-note">Deployed: ${r.selected_model==='linear'?'linear classifier':'small neural network'} · ${count(selected.parameters)} parameters · 140 features.<br>Training: ${r.split_counts.train} cycles / validation: ${r.split_counts.validation} cycles. Input scaling and class weights use training only.</p></div></section>
 <section class="panel"><div class="panel-heading"><div><h2>Per-component test results</h2><p>Deployed model · original UCI condition classes</p></div></div><div class="table-scroll"><table class="data-table"><thead><tr><th>Component</th><th>Macro F1</th><th>Accuracy</th></tr></thead><tbody>${config.components.map(c=>`<tr><td>${c.name}</td><td class="mono">${fixed(m.components[c.id].macro_f1)}</td><td class="mono">${fixed(m.components[c.id].accuracy*100,1)}%</td></tr>`).join('')}</tbody></table></div></section>
 <section class="panel robustness-panel"><div class="panel-heading"><div><h2>Class-level evaluation</h2><p>Rows: recorded condition · columns: model estimate</p></div><label class="sr-only" for="matrix-component">Component confusion matrix</label><select id="matrix-component" class="partition-select">${config.components.map(c=>`<option value="${c.id}">${c.name}</option>`).join('')}</select></div><div class="table-scroll" id="confusion-matrix"></div></section>
 <section class="panel robustness-panel"><div class="panel-heading"><div><h2>Sensor-removal sensitivity</h2><p>Offline diagnostic on measured test cycles · frozen deployed model</p></div><span class="label-pill">${cases?'14 CHANNELS':'NOT EVALUATED'}</span></div>${cases?`<div class="table-scroll"><table class="data-table robustness-table"><thead><tr><th>Diagnostic condition</th><th>Mean macro F1</th><th>All four correct</th></tr></thead><tbody>${cases.map(c=>`<tr><td>${esc(c.name)}</td><td class="mono">${fixed(c.metrics.mean_macro_f1)}</td><td class="mono">${fixed(c.metrics.all_components_correct*100,1)}%</td></tr>`).join('')}</tbody></table></div><div class="robustness-foot">Artificial removal of one channel from the original measured test features, followed by replacement with training means. No synthetic readings were generated. This is a sensitivity diagnostic, not evidence from physical sensor failures.<br>The dashboard abstains on incomplete inputs; it does not silently use this imputation policy.</div>`:'<p class="prose-panel small-note">Sensitivity report not available.</p>'}</section></div>`;
 const experiment=evaluation.accumulatorExperiment;
 if(experiment){
 const names={linear:'Linear baseline',mlp:'Small neural network',linear_knn:'Linear + nearest training cycles'};
 $('validation-content').insertAdjacentHTML('beforeend',`<section class="panel robustness-panel"><div class="panel-heading"><div><h2>Accumulator upgrade experiment</h2><p>Separate evaluation · deployed model retained</p></div><span class="label-pill">RESEARCH RESULT</span></div><div class="robustness-foot">A small nearest-cycle classifier was evaluated for the accumulator. It did not improve validation performance, so the current deployment is retained. This experiment uses ${experiment.split_counts.train} training, ${experiment.split_counts.validation} validation and ${experiment.split_counts.test} separately reserved test cycles. Its scores cannot be compared directly with the original test scores above.</div><div class="table-scroll"><table class="data-table"><thead><tr><th>Experimental model</th><th>Validation accumulator F1</th><th>Test accumulator F1</th></tr></thead><tbody>${Object.entries(experiment.models).map(([kind,model])=>`<tr><td>${esc(names[kind]||kind)}</td><td class="mono">${fixed(model.validation.components.accumulator.macro_f1)}</td><td class="mono">${fixed(model.test.components.accumulator.macro_f1)}</td></tr>`).join('')}</tbody></table></div><div class="robustness-foot">Whole condition combinations remain separate. Training-only normalization and neighbor references; nine validation-only neighbor candidates. Previously inspected test combinations were excluded. These are re-reserved cycles from the same rig; independent-machine validation still requires new measurements.</div></section>`);
 }
 renderGRUComparison();
 renderTreeComparison();
 $('matrix-component').addEventListener('change',renderMatrix);renderMatrix();$('footer-model').textContent=`${r.checkpoint_sha256.slice(0,8).toUpperCase()} / MEASURED UCI DATA`;
 $('model-method').textContent=`Validation-selected deployment: ${r.selected_model==='linear'?'linear classifier':'small neural network'}, ${count(selected.parameters)} parameters. Scores express uncalibrated class support.`;
 $('split-method').innerHTML=Object.entries(r.split_counts).map(([name,n])=>`<div><strong>${count(n)}</strong><span>${name} cycles</span></div>`).join('');
}
function renderGRUComparison(){
 const experiment=evaluation.gruExperiment;if(!experiment)return;
 const names={linear:'Linear · refitted baseline',gru_1:'GRU · one layer',gru_2:'GRU · two layers'};
 const decision=experiment.layer_decision;
 $('validation-content').insertAdjacentHTML('beforeend',`<section class="panel robustness-panel"><div class="panel-heading"><div><h2>GRU layer comparison</h2><p>Matched training and evaluation cycles · active model retained</p></div><span class="label-pill">REAL SENSOR SEQUENCES</span></div><div class="robustness-foot">${esc(decision.reason)} Both architectures were freshly fitted on ${count(experiment.split_counts.train)} training cycles, with ${count(experiment.split_counts.validation)} validation and ${count(experiment.split_counts.test)} test cycles. These are different records from the original evaluation above.</div><div class="table-scroll"><table class="data-table"><thead><tr><th>Model</th><th>Parameters</th><th>Validation macro F1</th><th>Test macro F1</th><th>All four correct</th></tr></thead><tbody>${Object.entries(experiment.models).map(([kind,model])=>`<tr><td>${esc(names[kind]||kind)}${kind===experiment.selected_model?' · validation winner':''}</td><td class="mono">${count(model.parameters)}</td><td class="mono">${fixed(model.validation.mean_macro_f1)}</td><td class="mono">${fixed(model.test.mean_macro_f1)}</td><td class="mono">${fixed(model.test.all_components_correct*100,1)}%</td></tr>`).join('')}</tbody></table></div><div class="robustness-foot">GRU: 32 hidden units, 60 ordered one-second averages × 14 measured channels; memory resets for each cycle. Linear: original 140 native-sample summaries. Normalization uses training only. Two layers are attempted only when one layer beats linear on validation. Test scores are calculated after layer selection.<br>The split excludes both earlier test sets, but remains an internal comparison on previously explored, same-rig data. The deployed model and its original test results are unchanged.</div></section>`);
}
function renderTreeComparison(){
 const experiment=evaluation.treeExperiment;if(!experiment)return;
 const names={linear:'Linear · refitted baseline',boosted_trees:'Small boosted trees'};
 $('validation-content').insertAdjacentHTML('beforeend',`<section class="panel robustness-panel"><div class="panel-heading"><div><h2>Boosted-tree comparison</h2><p>Four grouped folds · current deployment retained</p></div><span class="label-pill">REAL SENSOR FEATURES</span></div><div class="robustness-foot">A compact tree model was compared with the linear architecture on ${count(experiment.development_cycles)} development cycles. All ${count(experiment.excluded_test_cycles)} cycles in previously inspected test partitions were excluded. Trees beat linear on ${experiment.tree_better_folds} of four folds.</div><div class="table-scroll"><table class="data-table"><thead><tr><th>Model</th><th>Fold mean macro F1</th><th>Out-of-fold macro F1</th><th>Accumulator F1</th><th>All four correct</th></tr></thead><tbody>${Object.entries(experiment.models).map(([kind,model])=>`<tr><td>${esc(names[kind]||kind)}</td><td class="mono">${fixed(model.fold_mean_macro_f1)}</td><td class="mono">${fixed(model.out_of_fold.mean_macro_f1)}</td><td class="mono">${fixed(model.out_of_fold.components.accumulator.macro_f1)}</td><td class="mono">${fixed(model.out_of_fold.all_components_correct*100,1)}%</td></tr>`).join('')}</tbody></table></div><div class="robustness-foot">Both models use the same 140 measured-cycle features. Every condition combination stays in one fold; linear training chooses its epoch within the training portion only. This is an internal same-rig comparison, so its scores cannot be compared directly with the deployed model's original test score. No model was replaced.</div></section>`);
}
function renderMatrix(){
 const component=config.components.find(c=>c.id===$('matrix-component').value),r=evaluation.report,m=r.models[r.selected_model].test.components[component.id];
 $('confusion-matrix').innerHTML=`<table class="data-table class-matrix"><thead><tr><th>Recorded / estimated</th>${component.labels.map(label=>`<th>${esc(label)}</th>`).join('')}<th>Recall</th></tr></thead><tbody>${m.confusion.map((row,i)=>`<tr><td>${esc(component.labels[i])}</td>${row.map((n,j)=>`<td class="mono" ${i===j?'style="background:#edf3e9"':''}>${n}</td>`).join('')}<td class="mono">${fixed(m.per_class_recall[i]*100,1)}%</td></tr>`).join('')}</tbody></table>`;
}
function render(){if(!config||!state)return;updateControls();updateExport();renderMetrics();renderComponents();renderAlerts();renderSensorDetails();renderSensorTable();}

document.addEventListener('DOMContentLoaded',async()=>{
 busy=true;updateControls();try{config=await api('/api/config');$('partition-select').innerHTML=Object.entries(config.partitions).map(([name,n])=>`<option value="${name}">${name==='test'?'Held-out test':name==='all'?'All recorded cycles':name[0].toUpperCase()+name.slice(1)} (${n})</option>`).join('');$('trace-select').innerHTML=config.sensors.map(s=>`<option value="${s.id}">${s.id} / ${s.group}</option>`).join('');$('group-filter').insertAdjacentHTML('beforeend',[...new Set(config.sensors.map(s=>s.group))].map(g=>`<option>${g}</option>`).join(''));[state,evaluation]=await Promise.all([api('/api/state'),api('/api/validation')]);render();renderValidation();changeView(pages[location.hash.slice(1)]?location.hash.slice(1):'overview');}catch(e){error(e.message+' Check the local data and trained model.');}finally{busy=false;updateControls();}
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
