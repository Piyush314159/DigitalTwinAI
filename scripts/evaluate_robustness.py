"""Frozen real-data sensitivity check: python -m scripts.evaluate_robustness.

No generated sensor measurements or simulator. Sensor removal is an explicitly
artificial diagnostic applied only to held-out measured cycles; missing features
are replaced by training means for this offline experiment. The app abstains on
incomplete inputs rather than silently applying this imputation policy.
"""
import json
import argparse
import numpy as np
import torch
from hydraulics import OUTPUT, SENSORS, COMPONENTS, prepare_data, evaluation_metrics, file_hash
from model import load_model, predict


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(2)
    data=prepare_data()
    model, checkpoint=load_model(OUTPUT/'model.pt')
    if checkpoint['dataset_fingerprint'] != str(data['fingerprint']):
        raise ValueError('The downloaded measurements do not match the trained checkpoint.')
    manifest=json.loads((OUTPUT/'split_manifest.json').read_text())
    indices=np.array(manifest['test']['cycle_indices'])
    targets=data['targets'][indices]
    cases=[]; saved={'targets':targets,'cycle_indices':indices}
    for sensor in [None,*SENSORS]:
        x=data['features'][indices].copy()
        if sensor:
            i=SENSORS.index(sensor)
            x[:,i*10:(i+1)*10]=model.mean.numpy()[i*10:(i+1)*10]
        predicted,_=predict(model,x)
        key=sensor['id'] if sensor else 'complete'
        saved[key]=predicted
        cases.append({'id':key,'name':f"{key} removed" if sensor else 'Complete measurements',
                      'metrics':evaluation_metrics(targets,predicted)})
    directory=OUTPUT/'robustness';directory.mkdir(exist_ok=True)
    report={'source':'real_test_rig','checkpoint_sha256':file_hash(OUTPUT/'model.pt'),
            'dataset_fingerprint':str(data['fingerprint']), 'cycles':len(indices),
            'protocol':'Frozen deployed classifier; one physical sensor removed at a time; its ten features replaced by training means. All targets are original UCI annotations.',
            'deployment_policy':'Incomplete cycles produce insufficient_data, not imputed normal estimates.',
            'cases':cases}
    (directory/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(directory/'predictions.npz',**saved)
    rows=['| Condition | Mean component macro F1 | All four correct |','|---|---:|---:|']
    rows += [f"| {c['name']} | {c['metrics']['mean_macro_f1']:.3f} | {c['metrics']['all_components_correct']:.3f} |" for c in cases]
    text='# Real-data sensor-removal sensitivity\n\n'+report['protocol']+'\n\n'+'\n'.join(rows)+'\n\n'+report['deployment_policy']+'\n'
    (directory/'report.md').write_text(text)
    print(text)


if __name__=='__main__':
    main()
