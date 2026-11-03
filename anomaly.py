"""Small, trainable anomaly detector for the simulated assembly line.

This is a separate, single-task model. It reuses the existing GAT + GRU block,
but does not claim to train the prototype's queue, risk or confidence heads.
Only observed readings, their availability masks and the fixed graph are inputs.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch import nn

from assembly_line import NUM_FEATURES, build_assembly_graph
from model import SpatioTemporalGATBlock
from simulator import AnomalyScenario, ProductionSimulator


SEVERITY_CUTOFF = 0.30
DEFAULT_CHECKPOINT = Path(__file__).parent / "training_outputs" / "anomaly_detector.pt"


@dataclass
class ShiftDataset:
    readings: torch.Tensor       # shifts, time, stations, features (unobserved = 0)
    masks: torch.Tensor
    targets: torch.Tensor        # shifts, time, stations
    manifests: list


def make_dataset(seeds, steps=30):
    """Generate independent complete shifts; labels never enter model features.

    These are new configurations of the SAME simulator, not evidence of
    generalization to real factories or to unseen fault mechanisms.
    """
    if steps < 12:
        raise ValueError("At least 12 steps are needed for varied fault onsets.")
    stations, _, _ = build_assembly_graph()
    readings, masks, targets, manifests = [], [], [], []
    categories = ("equipment_wear", "environmental", "operator", "upstream_quality")
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        scenarios = []
        count = 0 if rng.random() < 0.25 else int(rng.integers(1, 4))
        for index in range(count):
            category = str(rng.choice(categories))
            pool = range(13, 24) if category == "environmental" else (
                range(24, 40) if category == "operator" else range(40)
            )
            scenarios.append(AnomalyScenario(
                name=f"{category}_{index}", category=category,
                affected_stations=[int(rng.choice(list(pool)))],
                onset_step=int(rng.integers(2, steps - 8)),
                severity=float(rng.uniform(0.35, 1.0)),
            ))
        sim = ProductionSimulator(stations, scenarios=scenarios, seed=int(seed),
                                  shift_duration_steps=steps)
        shift = sim.generate_shift()
        # A few additional sensor outages per shift, independent of fault labels.
        observed = np.stack([s["sensor_mask"] for s in shift])
        outages = rng.random((len(stations), NUM_FEATURES)) >= 0.03
        observed = observed * outages[None, :, :]
        raw = np.stack([s["features"] for s in shift])
        x = np.where(observed > 0, raw, 0).astype(np.float32)
        severity = np.stack([s["ground_truth_anomaly"] for s in shift])
        y = (severity >= SEVERITY_CUTOFF).astype(np.float32)
        descriptor = {
            "seed": int(seed), "steps": steps,
            "scenarios": [vars(s) for s in scenarios],
        }
        descriptor["data_sha256"] = hashlib.sha256(
            x.tobytes() + observed.astype(np.float32).tobytes() + y.tobytes()
        ).hexdigest()
        readings.append(x)
        masks.append(observed.astype(np.float32))
        targets.append(y)
        manifests.append(descriptor)
    if not readings:
        raise ValueError("At least one shift is required.")
    return ShiftDataset(torch.from_numpy(np.stack(readings)),
                        torch.from_numpy(np.stack(masks)),
                        torch.from_numpy(np.stack(targets)), manifests)


def fit_normalizer(dataset):
    """Fit feature scales using TRAINING observations only."""
    values, observed = dataset.readings, dataset.masks
    axes = (0, 1, 2)
    count = observed.sum(dim=axes).clamp_min(1)
    mean = (values * observed).sum(dim=axes) / count
    variance = (((values - mean) ** 2) * observed).sum(dim=axes) / count
    return mean, variance.sqrt().clamp_min(1e-3)


class SensorNormalizer(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())

    def forward(self, readings, masks):
        available = (masks > 0) & torch.isfinite(readings)
        safe = torch.where(available, readings, self.mean)
        standardized = ((safe - self.mean) / self.scale).clamp(-12, 12)
        return torch.cat((standardized * available, available.float()), dim=-1)


def batch_edges(edge_index, batch_size, stations):
    """Make disjoint copies: shifts cannot pass messages to one another."""
    offsets = torch.arange(batch_size, device=edge_index.device) * stations
    return (edge_index[None, :, :] + offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)


class StationAnomalyDetector(nn.Module):
    def __init__(self, mean, scale, hidden_channels=32, dropout=0.05):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.normalizer = SensorNormalizer(mean, scale)
        self.input_proj = nn.Sequential(nn.Linear(NUM_FEATURES * 2, hidden_channels),
                                        nn.ELU(), nn.LayerNorm(hidden_channels))
        self.block = SpatioTemporalGATBlock(hidden_channels, hidden_channels,
                                          heads=4, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_channels + NUM_FEATURES * 2, 16),
                                  nn.ELU(), nn.Linear(16, 1))

    def forward_step(self, readings, edge_index, masks, hidden=None):
        prepared = self.normalizer(readings, masks)
        hidden = self.block(self.input_proj(prepared), edge_index, hidden)
        logits = self.head(torch.cat((hidden, prepared), dim=-1)).squeeze(-1)
        return logits, hidden

    def forward(self, readings, edge_index, masks):
        batch_size, steps, stations, _ = readings.shape
        edges = batch_edges(edge_index, batch_size, stations)
        hidden, outputs = None, []  # Reset recurrence at every new batch of shifts.
        for t in range(steps):
            logits, hidden = self.forward_step(
                readings[:, t].reshape(-1, NUM_FEATURES), edges,
                masks[:, t].reshape(-1, NUM_FEATURES), hidden)
            outputs.append(logits.reshape(batch_size, stations))
        return torch.stack(outputs, dim=1)


class LocalLinearDetector(nn.Module):
    """Baseline using the same readings/masks, without graph or time context."""
    def __init__(self, mean, scale):
        super().__init__()
        self.normalizer = SensorNormalizer(mean, scale)
        self.head = nn.Linear(NUM_FEATURES * 2, 1)

    def forward(self, readings, edge_index, masks):
        return self.head(self.normalizer(readings, masks)).squeeze(-1)


def classification_metrics(targets, scores, threshold):
    y = np.asarray(targets).reshape(-1).astype(bool)
    scores = np.asarray(scores).reshape(-1)
    if not y.size or y.size != scores.size or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match the targets.")
    predictions = scores >= threshold
    tp = int(np.sum(y & predictions))
    fp = int(np.sum(~y & predictions))
    fn = int(np.sum(y & ~predictions))
    tn = int(np.sum(~y & ~predictions))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    # Average precision at distinct score cutoffs, including ties as a group.
    order = np.argsort(-scores, kind="stable")
    sorted_y, sorted_scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), y.size - 1]
    cumulative_tp = np.cumsum(sorted_y)[ends]
    ranked_precision = cumulative_tp / (ends + 1)
    ranked_recall = cumulative_tp / max(int(y.sum()), 1)
    ap = float(np.sum(np.diff(np.r_[0.0, ranked_recall]) * ranked_precision))
    return {
        "threshold": float(threshold), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "average_precision": ap, "accuracy": (tp + tn) / max(y.size, 1),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "true_negative": tn, "observations": int(y.size), "positives": int(y.sum()),
    }


def select_threshold(targets, scores):
    """Choose once on validation data, never on held-out test labels."""
    results = [classification_metrics(targets, scores, t)
               for t in np.linspace(0.01, 0.99, 99)]
    return max(results, key=lambda m: (m["f1"], m["precision"], m["threshold"]))["threshold"]


@torch.no_grad()
def predict_scores(model, dataset, edge_index, batch_size=16):
    model.eval()
    predictions = []
    for start in range(0, len(dataset.readings), batch_size):
        predictions.append(model(dataset.readings[start:start + batch_size], edge_index,
                                 dataset.masks[start:start + batch_size]).sigmoid().cpu())
    return torch.cat(predictions).numpy()


def load_detector(checkpoint_path=DEFAULT_CHECKPOINT):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("task") != "current_station_anomaly":
        raise ValueError("Unsupported anomaly checkpoint.")
    model = StationAnomalyDetector(torch.zeros(NUM_FEATURES), torch.ones(NUM_FEATURES),
                                   **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
