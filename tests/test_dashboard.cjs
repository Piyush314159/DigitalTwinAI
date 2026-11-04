/* Browser-facing regression checks without extra packages or a browser install. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function dashboard() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {value:'',checked:false,hidden:true,addEventListener(){},
      insertAdjacentHTML(_,html){this.innerHTML=(this.innerHTML||'')+html;}});
    return elements.get(id);
  };
  const context = vm.createContext({
    document:{getElementById:element,addEventListener(){}},
    window:{addEventListener(){}},location:{hash:'#overview'},
    URLSearchParams,setTimeout,clearTimeout,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/dashboard.js'),'utf8'),context);
  return {run:code=>vm.runInContext(code,context),context,element};
}

test('missing trace samples leave gaps instead of being plotted as zero',()=>{
  const {run}=dashboard();
  run(`config={sensors:[{id:'PS1',name:'Pressure 1',unit:'bar'}]};
       state={cycle:8,observations:[{id:'PS1',trace:Array(60).fill(100)}]};
       state.observations[0].trace[10]=null;`);
  const html=run("chart('PS1')");
  assert.equal((html.match(/<polyline /g)||[]).length,2);
  assert.match(html,/missing readings shown as gaps/);
  assert.doesNotMatch(html,/NaN|Infinity/);
  run('state.observations[0].trace.fill(null)');
  assert.match(run("chart('PS1')"),/Signal unavailable/);
  assert.equal(run('fixed(null)'),'—');
});

test('replay mutations send the current tab selection with every request',async()=>{
  const {run,context}=dashboard();
  context.fetch=async(url,options)=>{
    context.sent={url,body:JSON.parse(options.body)};
    return {ok:true,json:async()=>({partition:'all',position:8,totalRecords:2205,cycle:9,summary:{flaggedComponents:1}})};
  };
  run("state={partition:'all',position:7,totalRecords:2205};render=()=>{};");
  assert.equal(await run("mutate('/api/step')"),true);
  assert.equal(context.sent.url,'/api/step');
  assert.deepEqual(context.sent.body,{partition:'all',position:7});
  assert.equal(run('state.position'),8);
});

test('missing models and incomplete channels have explicit dashboard labels',()=>{
  const {run,element}=dashboard();
  run(`config={modelKind:null};state={modelStatus:'not_trained',partition:'test',totalRecords:328,
       summary:{flaggedComponents:null},dataQuality:{complete:true,invalidChannels:[]}};renderMetrics();`);
  assert.equal(element('model-kind').textContent,'MODEL UNAVAILABLE');
  assert.equal(element('data-quality-note').hidden,true);
  run(`state.modelStatus='insufficient_data';state.dataQuality={complete:false,invalidChannels:['PS1']};renderMetrics();`);
  assert.equal(element('model-state').textContent,'Insufficient sensor data');
  assert.equal(element('data-quality-note').hidden,false);
  assert.match(element('data-quality-note').textContent,/PS1/);
});

test('reference requests and CSV exports select the displayed record explicitly',async()=>{
  const {run,context,element}=dashboard();
  run("state={partition:'all',position:7,cycle:8};renderComponents=()=>{};");
  element('show-reference').checked=true;
  context.fetch=async url=>{
    context.requested=url;
    return {ok:true,json:async()=>({cycle:8,annotationOnly:true,conditions:[]})};
  };
  await run('loadReference()');
  assert.equal(context.requested,'/api/reference?partition=all&position=7');
  run('updateExport()');
  assert.equal(element('export-link').href,'/api/snapshot.csv?partition=all&position=7');
  context.location.hash='#validation';run('updateExport()');
  assert.equal(element('export-link').href,'/api/evaluation.json');
});

test('validation identifies the primary TCN and its grouped research score',()=>{
  const {run,context,element}=dashboard();
  const output=path.join(__dirname,'../training_outputs');
  context.reportData=JSON.parse(fs.readFileSync(path.join(output,'metrics.json'),'utf8'));
  context.tcnData=JSON.parse(fs.readFileSync(path.join(output,'tcn_experiment/metrics.json'),'utf8'));
  context.active={kind:'tcn',checkpoint_sha256:'c332d3abf40d3afa',parameters:12558,training_cycles:2205,training_epochs:33};
  run(`config={components:Object.entries(tcnData.models.tcn.out_of_fold.components)
    .map(([id,m])=>({id,name:id,labels:m.confusion.map((_,i)=>String(i))}))};
    evaluation={report:reportData,tcnExperiment:tcnData,activeModel:active};
    document.getElementById('matrix-component').value='accumulator';renderValidation();`);
  const html=element('validation-content').innerHTML;
  assert.match(html,/Primary multirate TCN/);
  assert.match(html,/0\.945/);
  assert.match(html,/trained on all 2,205 UCI cycles/);
  assert.match(html,/not a test score for the full-data checkpoint/);
  assert.match(html,/Earlier linear test/);
  assert.match(html,/0\.950/);
  assert.match(element('confusion-matrix').innerHTML,/Recall/);
});

test('matched TCN and linear scores remain separate from the old split',()=>{
  const {run,context,element}=dashboard();
  const output=path.join(__dirname,'../training_outputs');
  context.tcnData=JSON.parse(fs.readFileSync(path.join(output,'tcn_experiment/metrics.json'),'utf8'));
  context.reportData=JSON.parse(fs.readFileSync(path.join(output,'metrics.json'),'utf8'));
  context.active={kind:'tcn',checkpoint_sha256:'c332d3abf40d3afa',parameters:12558,training_cycles:2205,training_epochs:33};
  run(`config={components:Object.entries(tcnData.models.tcn.out_of_fold.components)
    .map(([id,m])=>({id,name:id,labels:m.confusion.map((_,i)=>String(i))}))};
    evaluation={report:reportData,tcnExperiment:tcnData,activeModel:active};
    document.getElementById('matrix-component').value='accumulator';renderValidation();`);
  const html=element('validation-content').innerHTML;
  assert.match(html,/Matched architecture comparison/);
  for(const model of Object.values(context.tcnData.models)){
    assert.ok(html.includes(model.fold_mean_macro_f1.toFixed(3)));
  }
  assert.match(html,/Independent-machine validation is still pending/);
  assert.match(html,/not directly comparable/);
});
