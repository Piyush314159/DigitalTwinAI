"""Evaluate the saved detector under declared challenges, without retraining.

Run from the project root: .venv/bin/python -m scripts.evaluate_robustness
The saved model, training normalizer and validation-selected threshold stay fixed.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from anomaly import (DEFAULT_CHECKPOINT, LocalLinearDetector, classification_metrics,
                     load_detector, make_dataset, predict_scores)
from assembly_line import build_assembly_graph


def apply_sensor_challenge(dataset, *, outage_rate=0.0, noise_fraction=0.0, scale=None, seed=917):
    """Change only measured inputs; simulator targets remain untouched."""
    changed = copy.deepcopy(dataset)
    rng = np.random.default_rng(seed)
    if outage_rate:
        # Sustained outages, independently sampled per shift/station/feature.
        keep = rng.random((len(changed.readings), 1, 40, 12)) >= outage_rate
        changed.masks *= torch.from_numpy(keep)
    if noise_fraction:
        if scale is None:
            raise ValueError("Measurement noise requires the frozen training feature scales.")
        noise = torch.from_numpy(rng.normal(size=tuple(changed.readings.shape)).astype(np.float32))
        changed.readings += noise * scale * noise_fraction
    changed.readings *= changed.masks
    return changed


def summarize(targets, scores, threshold):
    metrics = classification_metrics(targets, scores, threshold)
    metrics['false_alarm_rate'] = metrics['false_positive'] / max(
        metrics['false_positive'] + metrics['true_negative'], 1)
    metrics['miss_rate'] = (metrics['false_negative'] / metrics['positives']
                            if metrics['positives'] else None)
    if not metrics['positives']:
        # No positive examples: recall/F1/AP are not meaningful for this case.
        metrics.update(recall=None, f1=None, average_precision=None)
    if not (metrics['true_positive'] + metrics['false_positive']):
        metrics['precision'] = None
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--shifts', type=int, default=32)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_CHECKPOINT.parent / 'robustness')
    args = parser.parse_args()
    if not 1 <= args.shifts <= 1000:
        parser.error('Use 1–1,000 complete shifts per condition.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    detector, checkpoint = load_detector(args.checkpoint)
    baseline_checkpoint = torch.load(args.checkpoint.with_name('linear_baseline.pt'),
                                     map_location='cpu', weights_only=True)
    baseline = LocalLinearDetector(torch.zeros(12), torch.ones(12))
    baseline.load_state_dict(baseline_checkpoint['state_dict'])
    baseline.eval()
    manifest_path = args.checkpoint.with_name('dataset_manifest.json')
    old_manifest = json.loads(manifest_path.read_text())
    old_seeds = {s['seed'] for split in old_manifest.values() for s in split}
    seed_sets = {name: list(range(start, start + args.shifts)) for name, start in (
        ('reference', 8_000_000), ('weaker_faults', 8_020_000),
        ('concurrent_faults', 8_040_000), ('healthy_shifts', 8_060_000))}
    if old_seeds & {seed for seeds in seed_sets.values() for seed in seeds}:
        raise ValueError('Robustness seeds overlap an original dataset split.')
    reference = make_dataset(seed_sets['reference'])
    cases = [
        ('reference', 'Fresh simulated shifts', 'Original data recipe, new seeds.', reference),
        ('sensor_outages', 'Additional sensor outages',
         '20% of otherwise observable sensors unavailable for the full shift; paired with reference.',
         apply_sensor_challenge(reference, outage_rate=0.20)),
        ('measurement_noise', 'Noisy measurements',
         'Independent Gaussian measurement noise: standard deviation 20% of frozen training feature scales; paired with reference.',
         apply_sensor_challenge(reference, noise_fraction=0.20, scale=detector.normalizer.scale)),
        ('weaker_faults', 'Weaker disturbances', 'Fault severity sampled from 0.32–0.55.',
         make_dataset(seed_sets['weaker_faults'], severity_range=(0.32, 0.55))),
        ('concurrent_faults', 'Concurrent disturbances', '4–6 injected faults in each non-healthy shift.',
         make_dataset(seed_sets['concurrent_faults'], fault_count_range=(4, 7))),
        ('healthy_shifts', 'Healthy shifts', 'No injected disturbances; evaluate false alarms.',
         make_dataset(seed_sets['healthy_shifts'], healthy_probability=1.0)),
    ]
    _, edges, _ = build_assembly_graph()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, manifests, saved = [], {}, {}
    for key, title, description, dataset in cases:
        scores = predict_scores(detector, dataset, edges)
        linear_scores = predict_scores(baseline, dataset, edges)
        result = {
            'id': key, 'title': title, 'description': description,
            'shifts': len(dataset.readings),
            'detector': summarize(dataset.targets.numpy(), scores, checkpoint['threshold']),
            'baseline': summarize(dataset.targets.numpy(), linear_scores, baseline_checkpoint['threshold']),
            'data_sha256': hashlib.sha256(dataset.readings.numpy().tobytes() +
                                          dataset.masks.numpy().tobytes() +
                                          dataset.targets.numpy().tobytes()).hexdigest(),
        }
        manifests[key] = dataset.manifests
        saved[f'{key}_targets'] = dataset.targets.numpy()
        saved[f'{key}_scores'] = scores
        saved[f'{key}_baseline_scores'] = linear_scores
        results.append(result)
        m = result['detector']
        print(f"{title}: F1={m['f1']} | false alarms={m['false_positive']} | misses={m['false_negative']}", flush=True)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during frozen-model evaluation.')
    report = {
        'format_version': 1, 'data_source': 'synthetic',
        'checkpoint_sha256': checkpoint_hash, 'threshold': checkpoint['threshold'],
        'model_and_threshold_frozen': True, 'original_split_seeds_disjoint': True,
        'protocol': 'Predeclared six conditions. Sensor challenges share the reference shifts for paired comparisons. No fitting or threshold selection.',
        'limitations': [
            'Same simulator, fault categories and fixed graph as training. No real-factory validation.',
            'Station-time observations are correlated; counts are not independent trials.',
            'F1, recall and average precision are undefined for all-healthy shifts.',
            'These challenge results are now inspected; do not tune against them as a final holdout.',
        ],
        'cases': results, 'seconds': round(time.perf_counter() - started, 3),
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output_dir / 'dataset_manifest.json').write_text(json.dumps(manifests, indent=2) + '\n')
    np.savez_compressed(args.output_dir / 'predictions.npz', **saved)
    rows = []
    for case in results:
        m = case['detector']
        f1 = f"{m['f1']:.3f}" if m['f1'] is not None else '—'
        recall = f"{m['recall']:.3f}" if m['recall'] is not None else '—'
        rows.append(f"| {case['title']} | {f1} | {recall} | {m['false_positive']} | {m['false_negative']} |")
    summary = (
        '# DigitalTwin — frozen-model robustness evaluation\n\n'
        f"{args.shifts} complete simulated shifts per condition. Model and threshold "
        f"({checkpoint['threshold']:.2f}) remain fixed. No retraining.\n\n"
        '| Condition | F1 | Recall | False alarms | Misses |\n'
        '|---|---:|---:|---:|---:|\n' + '\n'.join(rows) + '\n\n'
        + '\n'.join(f"- {case['title']}: {case['description']}" for case in results) + '\n\n'
        + '\n'.join(f'- {note}' for note in report['limitations']) + '\n'
    )
    (args.output_dir / 'report.md').write_text(summary)
    print(summary)


if __name__ == '__main__':
    main()
