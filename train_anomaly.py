"""Train ONE DigitalTwin capability on a laptop: current-station anomalies.

Run: .venv/bin/python train_anomaly.py
No LLM, GPU, new service, or downloaded dataset is required.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from anomaly import (
    DEFAULT_CHECKPOINT, LocalLinearDetector, StationAnomalyDetector, SEVERITY_CUTOFF,
    classification_metrics, fit_normalizer, load_detector,
    make_dataset, predict_scores, select_threshold,
)
from assembly_line import FEATURE_NAMES, build_assembly_graph


def train_model(model, train, validation, edges, *, epochs, batch_size, seed, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    positives = float(train.targets.sum())
    negatives = train.targets.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Training requires normal and anomalous observations.")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    order_rng = np.random.default_rng(seed)
    best, best_ap, best_epoch, stale, history = None, -1.0, 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_rng.permutation(len(train.readings))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train.readings[batch], edges, train.masks[batch])
            loss = loss_fn(logits, train.targets[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        scores = predict_scores(model, validation, edges, batch_size)
        metrics = classification_metrics(validation.targets.numpy(), scores, 0.5)
        ap = metrics["average_precision"]
        history.append({"epoch": epoch, "training_loss": total_loss / len(order),
                        "validation_average_precision": ap})
        if ap > best_ap + 1e-5:
            best_ap, best_epoch = ap, epoch
            best, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch:2d} | loss {history[-1]['training_loss']:.4f} | "
                  f"validation AP {ap:.4f}", flush=True)
        if stale >= 8:
            break
    model.load_state_dict(best)
    model.eval()
    scores = predict_scores(model, validation, edges, batch_size)
    threshold = select_threshold(validation.targets.numpy(), scores)
    return {
        "best_epoch": best_epoch, "epochs_run": len(history),
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameters": sum(p.numel() for p in model.parameters()),
        "threshold": threshold,
        "validation": classification_metrics(validation.targets.numpy(), scores, threshold),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CHECKPOINT.parent)
    parser.add_argument("--train-shifts", type=int, default=128)
    parser.add_argument("--validation-shifts", type=int, default=32)
    parser.add_argument("--test-shifts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_shifts, args.validation_shifts, args.test_shifts, args.epochs,
           args.batch_size, args.threads) < 1:
        parser.error("Shift counts, epochs, batch size and threads must be positive.")
    if max(args.train_shifts, args.validation_shifts, args.test_shifts) > 10000:
        parser.error("Use at most 10,000 shifts per split to keep seed ranges disjoint.")
    if args.steps < 12:
        parser.error("Use at least 12 time steps per shift.")
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_base = args.seed * 100000
    seeds = {
        "train": list(range(seed_base, seed_base + args.train_shifts)),
        "validation": list(range(seed_base + 20000, seed_base + 20000 + args.validation_shifts)),
        "test": list(range(seed_base + 40000, seed_base + 40000 + args.test_shifts)),
    }
    print("Generating independent synthetic training and validation shifts...", flush=True)
    train = make_dataset(seeds["train"], args.steps)
    validation = make_dataset(seeds["validation"], args.steps)
    mean, scale = fit_normalizer(train)
    _, edges, _ = build_assembly_graph()
    baseline = LocalLinearDetector(mean, scale)
    detector = StationAnomalyDetector(mean, scale, hidden_channels=args.hidden_channels)
    print("Training local linear baseline...", flush=True)
    baseline_fit = train_model(baseline, train, validation, edges, epochs=args.epochs,
                               batch_size=args.batch_size, seed=args.seed, learning_rate=0.01)
    print("Training compact GAT + GRU anomaly detector...", flush=True)
    detector_fit = train_model(detector, train, validation, edges, epochs=args.epochs,
                               batch_size=args.batch_size, seed=args.seed, learning_rate=0.003)

    # The test split is first materialized AFTER every training/selection decision.
    # Do not tune the model using results from this fixed final evaluation.
    print("Final held-out evaluation (no training or threshold selection)...", flush=True)
    test = make_dataset(seeds["test"], args.steps)
    test_scores = predict_scores(detector, test, edges, args.batch_size)
    baseline_scores = predict_scores(baseline, test, edges, args.batch_size)
    detector_test = classification_metrics(test.targets.numpy(), test_scores, detector_fit["threshold"])
    baseline_test = classification_metrics(test.targets.numpy(), baseline_scores, baseline_fit["threshold"])
    manifest = {"train": train.manifests, "validation": validation.manifests, "test": test.manifests}
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_hash = hashlib.sha256(manifest_text.encode()).hexdigest()
    checkpoint = {
        "format_version": 1, "task": "current_station_anomaly",
        "data_source": "synthetic", "trained_heads": ["anomaly"],
        "model_config": {"hidden_channels": args.hidden_channels, "dropout": 0.05},
        "state_dict": detector.state_dict(), "threshold": detector_fit["threshold"],
        "feature_names": FEATURE_NAMES, "severity_cutoff": SEVERITY_CUTOFF,
        "manifest_sha256": manifest_hash,
    }
    checkpoint_path = args.output_dir / "anomaly_detector.pt"
    torch.save(checkpoint, checkpoint_path)
    # Verify that the saved artifact itself reproduces the evaluated scores.
    restored, _ = load_detector(checkpoint_path)
    restored_scores = predict_scores(restored, test, edges, args.batch_size)
    if not np.array_equal(test_scores, restored_scores):
        raise RuntimeError("Reloaded checkpoint changed predictions.")
    torch.save({"state_dict": baseline.state_dict(), "threshold": baseline_fit["threshold"],
                "format_version": 1, "task": "local_linear_anomaly_baseline"},
               args.output_dir / "linear_baseline.pt")
    np.savez_compressed(args.output_dir / "heldout_predictions.npz",
                        targets=test.targets.numpy(), detector_scores=test_scores,
                        baseline_scores=baseline_scores, seeds=np.array(seeds["test"]))
    report = {
        "task": "current_station_anomaly", "data_source": "synthetic",
        "target": f"Simulator disturbance severity >= {SEVERITY_CUTOFF}; current detection, not future warning.",
        "split_protocol": "Independent complete shifts with disjoint seeds; no overlapping windows across splits.",
        "selection": "Feature scales/positive weight: train only. Checkpoint: validation AP. Threshold: validation F1.",
        "seed": args.seed, "cpu_threads": args.threads,
        "split_counts": {name: len(items) for name, items in manifest.items()},
        "steps_per_shift": args.steps, "stations": 40,
        "normal_shifts": {name: sum(not s["scenarios"] for s in items) for name, items in manifest.items()},
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "detector": {**detector_fit, "test": detector_test},
        "linear_baseline": {**baseline_fit, "test": baseline_test},
        "always_normal_baseline": classification_metrics(test.targets.numpy(),
                                                          np.zeros_like(test_scores), 0.5),
        "limitations": [
            "All splits use the same synthetic generator and fixed assembly graph.",
            "Failure categories are shared across splits; only shifts/configurations are unseen.",
            "Station-step observations within a shift are correlated; they are not independent trials.",
            "Scores are not calibrated probabilities. No measured predictive lead time.",
            "Queue forecasting, defect risk, virtual sensing and confidence heads remain untrained.",
            "No real-factory validation, causal attribution, intervention benefit or measured savings.",
        ],
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
        "total_seconds": round(time.perf_counter() - started, 3),
        "checkpoint_reload_matches": True,
    }
    (args.output_dir / "dataset_manifest.json").write_text(manifest_text)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for name, metrics in (("GAT + GRU", detector_test), ("Local linear", baseline_test)):
        rows.append(f"| {name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                    f"{metrics['f1']:.3f} | {metrics['average_precision']:.3f} |")
    summary = (
        "# DigitalTwin — first anomaly training run\n\n"
        "A compact GAT + GRU detects current simulated station disturbances. "
        "Only this capability is trained; the original multi-head dashboard remains a prototype.\n\n"
        f"CPU run: {report['total_seconds']:.1f} seconds, {args.threads} threads, "
        f"{detector_fit['parameters']:,} parameters.\n\n"
        f"Complete shifts: {args.train_shifts} train / {args.validation_shifts} validation / "
        f"{args.test_shifts} test, each {args.steps} steps with 40 stations.\n\n"
        "Held-out results with validation-selected thresholds:\n\n"
        "| Model | Precision | Recall | F1 | Average precision |\n"
        "|---|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
        "Average precision is a ranking metric, not classification accuracy. "
        "A graph model advantage is established only if the measured comparison supports it.\n\n"
        + "\n".join(f"- {limitation}" for limitation in report["limitations"]) + "\n\n"
        "See metrics.json for thresholds, confusion counts and epoch history; "
        "dataset_manifest.json for seeds/scenario configurations; "
        "heldout_predictions.npz for the exact evaluated outputs.\n"
    )
    (args.output_dir / "training_report.md").write_text(summary)
    print(summary, flush=True)
    print(f"Saved checkpoint: {checkpoint_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
