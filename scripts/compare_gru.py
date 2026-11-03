"""Compare a 32-unit GRU with the current linear architecture on real cycles.

Run: python -m scripts.compare_gru
Two layers are trained only when one layer beats linear on validation macro F1.
All artifacts stay in gru_experiment/. No automatic deployment or test tuning.
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch import nn

from hydraulics import (OUTPUT as ACTIVE_OUTPUT, COMPONENTS, FEATURE_NAMES, SENSORS,
                        prepare_data, grouped_split, upgrade_split, evaluation_metrics, file_hash)
from model import HydraulicGRU, predict, load_model
from scripts.train_hydraulics import fit

OUTPUT=ACTIVE_OUTPUT/'gru_experiment'
SETTINGS={'split_seed':20261104,'training_seed':42,'hidden_size':32,
          'gru_epochs':200,'linear_epochs':250,'patience':35,'batch_size':64,
          'gru_learning_rate':.003,'weight_decay':.0001,'gradient_clip':3,
          'two_layer_dropout':.1,'minimum_improvement':.000001}


def measured_sequences(data):
    return np.ascontiguousarray(data['traces'].transpose(0,2,1),dtype=np.float32)


def experiment_split(profile):
    _,groups=np.unique(profile[:,:4],axis=0,return_inverse=True)
    excluded=np.union1d(grouped_split(profile)['test'],upgrade_split(profile)['test'])
    available=np.setdiff1d(np.unique(groups),np.unique(groups[excluded]))
    order=np.random.default_rng(SETTINGS['split_seed']).permutation(available)
    a,b=int(len(order)*.6),int(len(order)*.8)
    split={name:np.flatnonzero(np.isin(groups,selected)) for name,selected in (
        ('train',order[:a]),('validation',order[a:b]),('test',order[b:]))}
    return split,excluded


def freeze_protocol(data):
    split,excluded=experiment_split(data['profile'])
    plan={'version':'gru_v1','dataset_fingerprint':str(data['fingerprint']),
          'settings':SETTINGS,'partitions':{k:v.tolist() for k,v in split.items()},
          'excluded_previously_inspected_test_indices':excluded.tolist(),
          'inputs':{'linear':FEATURE_NAMES,'gru':'60 ordered one-second averages of 14 physical sensor channels; no annotations; reset hidden state each cycle.'},
          'layer_rule':'Train gru_2 only if gru_1 validation mean component macro F1 exceeds the refitted linear baseline by more than 0.000001. Maximum two layers.',
          'selection':'Best validation mean component macro F1; ties favor the simpler model. Freeze layer decision and checkpoints before scoring test once.',
          'deployment_policy':'Comparison only; never replace active model or reports.',
          'limitation':'Re-reserved groups from previously explored same-rig measurements, not new independent-machine evidence. Old deployed 0.950 score uses different test cycles.'}
    path=OUTPUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:
            raise ValueError('Frozen GRU protocol differs; do not tune against the inspected evaluation.')
    else:
        path.write_text(json.dumps(plan,indent=2)+'\n')
    return split,plan


def fit_gru(data,split,layers):
    torch.manual_seed(SETTINGS['training_seed'])
    sequences,targets=measured_sequences(data),data['targets']
    train,validation=split['train'],split['validation']
    mean=sequences[train].mean(axis=(0,1))
    scale=sequences[train].std(axis=(0,1)).clip(1e-4)
    model=HydraulicGRU(mean,scale,layers,SETTINGS['hidden_size'])
    optimizer=torch.optim.Adam(model.parameters(),lr=SETTINGS['gru_learning_rate'],
                               weight_decay=SETTINGS['weight_decay'])
    losses=[]
    for i,c in enumerate(COMPONENTS):
        counts=np.bincount(targets[train,i],minlength=len(c['codes']))
        if np.any(counts==0):
            raise ValueError('A training condition class is missing.')
        losses.append(nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(len(counts)*counts),dtype=torch.float32)))
    rng=np.random.default_rng(SETTINGS['training_seed'])
    best,best_score,best_epoch,stale,history=None,-1,0,0,[]
    started=time.perf_counter()
    for epoch in range(1,SETTINGS['gru_epochs']+1):
        model.train();total=0
        order=rng.permutation(train)
        for offset in range(0,len(order),SETTINGS['batch_size']):
            batch=order[offset:offset+SETTINGS['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            heads=model(torch.from_numpy(sequences[batch]))
            loss=sum(losses[i](heads[i],torch.from_numpy(targets[batch,i])) for i in range(4))/4
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite GRU training loss.')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),SETTINGS['gradient_clip'])
            optimizer.step();total+=float(loss.detach())*len(batch)
        metric=evaluation_metrics(targets[validation],predict(model,sequences[validation])[0])
        score=metric['mean_macro_f1']
        history.append({'epoch':epoch,'loss':total/len(train),'validation_mean_macro_f1':score})
        if score>best_score+SETTINGS['minimum_improvement']:
            best,best_score,best_epoch,stale=copy.deepcopy(model.state_dict()),score,epoch,0
        else:
            stale+=1
        if epoch==1 or epoch%10==0:
            print(f'gru_{layers} epoch {epoch}: validation macro F1 {score:.3f}',flush=True)
        if stale>=SETTINGS['patience']:
            break
    model.load_state_dict(best);model.eval()
    return model,{'layers':layers,'hidden_size':SETTINGS['hidden_size'],
                  'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,
                  'epochs_run':len(history),'history':history,'seconds':round(time.perf_counter()-started,3),
                  'validation':evaluation_metrics(targets[validation],predict(model,sequences[validation])[0])}


def save_checkpoint(model,data,split,path):
    checkpoint={'task':'uci_hydraulic_component_conditions_v1','kind':model.kind,
                'dataset_fingerprint':str(data['fingerprint']),'protocol_version':'gru_v1',
                'state_dict':model.state_dict(),'training_cycle_indices':split['train'].tolist()}
    if isinstance(model,HydraulicGRU):
        checkpoint.update(input_kind='sequence_1s',sequence_length=60,
                          sensor_ids=[s['id'] for s in SENSORS],
                          hidden_size=model.hidden_size,layers=model.layers)
    else:
        checkpoint.update(input_kind='summary_features',feature_names=FEATURE_NAMES)
    torch.save(checkpoint,path)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();data=prepare_data()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    split,plan=freeze_protocol(data)
    # Active artifacts are checked after this comparison to prevent accidental promotion.
    active_hashes={str(p.relative_to(ACTIVE_OUTPUT)):file_hash(p)
                   for p in ACTIVE_OUTPUT.glob('*') if p.is_file()}
    models,fits={},{}
    models['linear'],fits['linear']=fit('linear',data,split,SETTINGS['linear_epochs'],SETTINGS['training_seed'])
    models['gru_1'],fits['gru_1']=fit_gru(data,split,1)
    gain=fits['gru_1']['validation']['mean_macro_f1']-fits['linear']['validation']['mean_macro_f1']
    grow=gain>SETTINGS['minimum_improvement']
    print(f'One-layer validation gain versus linear: {gain:+.6f}. Two layers: {"train" if grow else "skip"}.',flush=True)
    if grow:
        models['gru_2'],fits['gru_2']=fit_gru(data,split,2)
    selected=max(models,key=lambda k:fits[k]['validation']['mean_macro_f1'])
    hashes={}
    for kind,model in models.items():
        save_checkpoint(model,data,split,OUTPUT/f'{kind}.pt')
        hashes[kind]=file_hash(OUTPUT/f'{kind}.pt')
    decision={'selected_model':selected,'two_layers_trained':grow,'one_layer_validation_gain':gain,
              'reason':'One-layer GRU improved validation; compare two layers.' if grow else 'One-layer GRU did not improve validation; extra layers skipped as requested.',
              'checkpoint_sha256':hashes,'protocol_sha256':file_hash(OUTPUT/'protocol.json')}
    # Write the validation-only decision BEFORE accessing the new test inputs/targets.
    (OUTPUT/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    test=split['test'];sequences=measured_sequences(data)
    reports={};saved={'cycle_indices':test,'targets':data['targets'][test]}
    for kind,model in models.items():
        inputs=sequences[test] if isinstance(model,HydraulicGRU) else data['features'][test]
        prediction,scores=predict(model,inputs)
        restored,_=load_model(OUTPUT/f'{kind}.pt')
        reloaded,reloaded_scores=predict(restored,inputs)
        np.testing.assert_array_equal(reloaded,prediction)
        for a,b in zip(scores,reloaded_scores):np.testing.assert_array_equal(a,b)
        reports[kind]={**fits[kind],'test':evaluation_metrics(data['targets'][test],prediction),
                       'checkpoint_bytes':(OUTPUT/f'{kind}.pt').stat().st_size}
        saved[f'{kind}_predictions']=prediction
        for c,values in zip(COMPONENTS,scores):saved[f"{kind}_{c['id']}_scores"]=values
    np.savez_compressed(OUTPUT/'test_predictions.npz',**saved)
    report={'source':'real_test_rig','purpose':'Conditional GRU layer comparison',
            'dataset_fingerprint':str(data['fingerprint']),'protocol_version':'gru_v1',
            'protocol_sha256':decision['protocol_sha256'],'checkpoint_sha256':hashes,
            'selected_model':selected,'layer_decision':decision,
            'split_counts':{k:len(v) for k,v in split.items()},
            'split_groups':{k:len(np.unique(data['profile'][v,:4],axis=0)) for k,v in split.items()},
            'models':reports,'seconds':round(time.perf_counter()-started,3),
            'deployed_model_unchanged':True,'limitations':[plan['limitation'],
                'Different input representations: native-sample summaries for linear; one-second averaged sequences for GRU.',
                'Full-cycle condition classification, not future failure prediction. Scores are uncalibrated class support.',
                'One training seed and one grouped split; this does not measure variation across runs.']}
    for relative,sha in active_hashes.items():
        if file_hash(ACTIVE_OUTPUT/relative)!=sha:raise RuntimeError('An active artifact changed during the experiment.')
    (OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=['| Model | Layers | Parameters | Validation macro F1 | Test macro F1 | Test accumulator F1 | All four correct | Training seconds |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind,r in reports.items():
        rows.append(f"| {kind} | {r.get('layers','—')} | {r['parameters']:,} | {r['validation']['mean_macro_f1']:.4f} | {r['test']['mean_macro_f1']:.4f} | {r['test']['components']['accumulator']['macro_f1']:.4f} | {r['test']['all_components_correct']:.2%} | {r['seconds']:.1f} |")
    text='# GRU comparison on measured hydraulic cycles\n\n'
    text+=f"Same cycle partitions for every freshly fitted candidate: {report['split_counts']}; condition groups: {report['split_groups']}. Both previously inspected test sets are excluded.\n\n"
    text+='\n'.join(rows)+'\n\n'+decision['reason']+f" Validation-selected experiment winner: **{selected}**.\n\n"
    text+='The current dashboard model and its original 0.950 test score are unchanged. That score uses different records and cannot be directly ranked against this table.\n\n'
    text+='Input: 60 ordered one-second averages × 14 measured channels. Hidden state resets per cycle. GRU normalizers and loss weights are fitted on training only. All layer choices and checkpoint selection use validation only; test is scored afterward.\n\n'
    text+='\n'.join('- '+v for v in report['limitations'])+'\n'
    (OUTPUT/'report.md').write_text(text)
    print(text,flush=True)


if __name__=='__main__':main()
