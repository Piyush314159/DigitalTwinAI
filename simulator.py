"""
DigitalTwin.ai — Production Data Simulator
============================================
Generates realistic synthetic telemetry for a 40-station vehicle assembly line.

Features per station (12 dimensions):
    queue_length, cycle_time, defect_rate, temperature, vibration,
    torque_deviation, utilization, throughput, humidity, power_draw,
    hrs_since_maintenance, operator_skill_index

Supports:
    - Sensor masking for under-instrumented stations
    - Multi-causal anomaly injection (equipment wear, upstream quality,
      environmental, operator variation)
    - 8-hour shift temporal dynamics with ramp-up, breaks, and degradation
    - Configurable anomaly scenarios

Author : Piyush Maji
Project: Accenture Innovation Challenge 2026
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from assembly_line import (
    Station, Zone, InstrumentationLevel, EquipmentVintage,
    NUM_FEATURES, get_sensor_mask_tensor,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyScenario:
    """Defines a multi-causal anomaly injected into the simulation."""
    name: str
    category: str           # "equipment_wear", "upstream_quality", "environmental", "operator"
    affected_stations: List[int]
    onset_step: int         # Time step when anomaly begins
    severity: float         # 0.0 to 1.0 — scales the anomaly magnitude
    description: str = ""


# Pre-built anomaly scenarios that demonstrate key capabilities
DEFAULT_SCENARIOS = [
    AnomalyScenario(
        name="Weld Gun Electrode Wear",
        category="equipment_wear",
        affected_stations=[7],  # Spot Weld Cell 3 (legacy)
        onset_step=5,
        severity=0.7,
        description="Progressive electrode wear causes weld strength degradation, "
                    "detected first at Geometry Check (station 9) downstream."
    ),
    AnomalyScenario(
        name="Paint Booth Humidity Excursion",
        category="environmental",
        affected_stations=[19, 21],  # Color Booth and Clear Coat
        onset_step=8,
        severity=0.6,
        description="HVAC fault causes humidity spike in paint zone, leading to "
                    "orange peel and fish-eye defects in color and clear coat."
    ),
    AnomalyScenario(
        name="Stamping Die Wear",
        category="equipment_wear",
        affected_stations=[2],  # Stamping Press B (legacy)
        onset_step=3,
        severity=0.5,
        description="Progressive die wear on legacy press causes dimensional "
                    "drift in fender panels, propagating through welding."
    ),
    AnomalyScenario(
        name="Shift Change Operator Drop",
        category="operator",
        affected_stations=[24, 25, 26, 27],  # Manual final assembly stations
        onset_step=15,
        severity=0.4,
        description="B-shift has less experienced operators at manual trim stations, "
                    "causing higher defect rates and longer cycle times."
    ),
    AnomalyScenario(
        name="Upstream Part Quality Issue",
        category="upstream_quality",
        affected_stations=[8],  # Adhesive Bonding
        onset_step=10,
        severity=0.6,
        description="Batch of adhesive from new supplier has lower viscosity, "
                    "affecting bond strength at multiple joints."
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# NORMAL OPERATING RANGES PER FEATURE
# ═══════════════════════════════════════════════════════════════════════════════

# [min, max] for each feature under normal operating conditions
NORMAL_RANGES = {
    0:  (1.0, 6.0),       # queue_length
    1:  (0.85, 1.15),     # cycle_time (multiplier of takt)
    2:  (0.1, 1.5),       # defect_rate_%
    3:  (60.0, 78.0),     # temperature_C
    4:  (0.5, 3.5),       # vibration_mm/s
    5:  (0.0, 2.0),       # torque_deviation_%
    6:  (65.0, 88.0),     # utilization_%
    7:  (0.85, 1.0),      # throughput (fraction of target)
    8:  (40.0, 55.0),     # humidity_%
    9:  (10.0, 35.0),     # power_draw_kW
    10: (5.0, 150.0),     # hrs_since_maintenance
    11: (0.6, 1.0),       # operator_skill_index
}

# Paint zone has different temperature/humidity normals
PAINT_ZONE_OVERRIDES = {
    3: (22.0, 25.0),      # Tight temperature control in paint
    8: (45.0, 55.0),      # Controlled humidity in paint
}


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ProductionSimulator:
    """
    Simulates realistic production telemetry for the assembly line.

    The simulator generates temporal sequences of node feature matrices,
    with configurable anomaly injection and sensor masking.
    """

    def __init__(
        self,
        stations: List[Station],
        scenarios: Optional[List[AnomalyScenario]] = None,
        seed: int = 42,
        shift_duration_steps: int = 30,  # Total time steps in simulation
    ):
        self.stations = stations
        self.num_stations = len(stations)
        self.scenarios = scenarios or DEFAULT_SCENARIOS
        self.rng = np.random.RandomState(seed)
        self.shift_duration = shift_duration_steps
        self.sensor_mask = get_sensor_mask_tensor(stations)

        # Pre-compute per-station maintenance hours (randomized starting point)
        self._maint_hours = self.rng.uniform(5, 300, self.num_stations)
        for s in stations:
            if s.vintage == EquipmentVintage.LEGACY:
                self._maint_hours[s.id] = self.rng.uniform(100, 400)

    def generate_step(self, step: int) -> Dict:
        """
        Generate telemetry for a single time step.

        Returns dict with:
            'features'        : np.ndarray [num_stations, 12] — raw feature values
            'masked_features' : np.ndarray [num_stations, 12] — with sensor gaps as NaN
            'sensor_mask'     : np.ndarray [num_stations, 12] — boolean observability mask
            'ground_truth_anomaly' : np.ndarray [num_stations] — 0/1 anomaly labels
            'active_scenarios'     : list of active AnomalyScenario names
        """
        features = np.zeros((self.num_stations, NUM_FEATURES), dtype=np.float32)
        anomaly_labels = np.zeros(self.num_stations, dtype=np.float32)
        active_scenarios = []

        # ── Step 1: Generate normal baseline telemetry ──
        for s in self.stations:
            features[s.id] = self._generate_normal(s, step)

        # ── Step 2: Apply shift temporal dynamics ──
        features = self._apply_temporal_dynamics(features, step)

        # ── Step 3: Inject anomalies ──
        for scenario in self.scenarios:
            if step >= scenario.onset_step:
                progression = min(1.0, (step - scenario.onset_step) / 8.0)
                self._inject_anomaly(features, anomaly_labels, scenario, progression)
                active_scenarios.append(scenario.name)

        # ── Step 4: Apply cascade effects (downstream of anomalous stations) ──
        features = self._apply_cascade(features, anomaly_labels, step)

        # ── Step 5: Apply sensor masking ──
        mask = self.sensor_mask.numpy().astype(np.float32)
        masked_features = features.copy()
        masked_features[~self.sensor_mask.numpy()] = 0.0  # Zero-fill unobserved

        return {
            'features': features,
            'masked_features': masked_features,
            'sensor_mask': mask,
            'ground_truth_anomaly': anomaly_labels,
            'active_scenarios': active_scenarios,
        }

    def generate_shift(self) -> List[Dict]:
        """Generate telemetry for an entire shift (all time steps)."""
        return [self.generate_step(t) for t in range(self.shift_duration)]

    # ─── Private helpers ──────────────────────────────────────────────────

    def _generate_normal(self, station: Station, step: int) -> np.ndarray:
        """Generate normal-state features for a single station."""
        features = np.zeros(NUM_FEATURES, dtype=np.float32)

        for feat_idx in range(NUM_FEATURES):
            lo, hi = NORMAL_RANGES[feat_idx]

            # Override for paint zone
            if station.zone == Zone.PAINT and feat_idx in PAINT_ZONE_OVERRIDES:
                lo, hi = PAINT_ZONE_OVERRIDES[feat_idx]

            if feat_idx == 1:  # cycle_time — multiply by station's takt
                base = self.rng.uniform(lo, hi) * station.takt_time_sec
            elif feat_idx == 10:  # hrs_since_maint — use tracked value
                base = self._maint_hours[station.id] + step * 0.5
            else:
                base = self.rng.uniform(lo, hi)

            # Legacy equipment has slightly noisier readings
            noise_scale = 0.02 if station.vintage == EquipmentVintage.MODERN else 0.05
            noise = self.rng.normal(0, noise_scale * (hi - lo))

            features[feat_idx] = base + noise

        return features

    def _apply_temporal_dynamics(self, features: np.ndarray, step: int) -> np.ndarray:
        """Apply shift-level temporal patterns."""
        total = self.shift_duration

        # Ramp-up in first 10% of shift (higher queue, lower throughput)
        if step < total * 0.1:
            ramp_factor = step / (total * 0.1)
            features[:, 0] *= (1.5 - 0.5 * ramp_factor)   # Queue higher initially
            features[:, 7] *= (0.7 + 0.3 * ramp_factor)   # Throughput lower

        # Break period at ~40-45% of shift
        if 0.40 * total <= step <= 0.45 * total:
            features[:, 6] *= 0.3   # Utilization drops during break
            features[:, 7] *= 0.2   # Throughput near zero
            features[:, 0] *= 0.5   # Queues drain

        # End-of-shift fatigue (last 15%)
        if step > total * 0.85:
            fatigue = (step - total * 0.85) / (total * 0.15)
            features[:, 2] += fatigue * 0.5    # Defect rate creeps up
            features[:, 11] -= fatigue * 0.1   # Operator focus drops

        return features

    def _inject_anomaly(
        self, features: np.ndarray, labels: np.ndarray,
        scenario: AnomalyScenario, progression: float
    ):
        """Inject an anomaly scenario into the feature matrix."""
        severity = scenario.severity * progression

        for sid in scenario.affected_stations:
            labels[sid] = min(1.0, labels[sid] + severity)

            if scenario.category == "equipment_wear":
                features[sid, 4] += severity * 8.0   # Vibration spike
                features[sid, 1] *= (1 + severity * 0.3)  # Cycle time increase
                features[sid, 2] += severity * 4.0   # Defect rate up
                features[sid, 5] += severity * 5.0   # Torque deviation
                features[sid, 9] += severity * 10.0  # Power draw up

            elif scenario.category == "environmental":
                features[sid, 3] += severity * 8.0   # Temperature excursion
                features[sid, 8] += severity * 15.0  # Humidity spike
                features[sid, 2] += severity * 3.0   # Defect rate up

            elif scenario.category == "operator":
                features[sid, 11] -= severity * 0.3  # Skill index drop
                features[sid, 1] *= (1 + severity * 0.25)  # Slower cycle
                features[sid, 2] += severity * 2.0   # More defects
                features[sid, 0] += severity * 3.0   # Queue build-up

            elif scenario.category == "upstream_quality":
                features[sid, 2] += severity * 5.0   # Defect rate spike
                features[sid, 0] += severity * 4.0   # Queue builds (rework)
                features[sid, 5] += severity * 3.0   # Torque deviation

    def _apply_cascade(
        self, features: np.ndarray, labels: np.ndarray, step: int
    ) -> np.ndarray:
        """
        Simulate cascade effects: anomalous stations affect downstream stations
        with a time-delayed, attenuated signal.
        """
        # Simple propagation: for each anomalous station, increase queue and
        # defect rate at stations within 3 hops downstream
        from assembly_line import build_edges
        edges = build_edges()

        # Build adjacency for forward edges only
        adj = {}
        for src, dst in edges:
            if dst > src:  # forward only
                adj.setdefault(src, []).append(dst)

        for sid in range(self.num_stations):
            if labels[sid] > 0.3:
                # BFS up to 3 hops downstream
                cascade_strength = labels[sid] * 0.4
                visited = {sid}
                frontier = [sid]
                for hop in range(3):
                    next_frontier = []
                    for node in frontier:
                        for neighbor in adj.get(node, []):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                next_frontier.append(neighbor)
                                # Attenuate by hop distance
                                attenuation = cascade_strength * (0.6 ** hop)
                                features[neighbor, 0] += attenuation * 2.0  # Queue
                                features[neighbor, 2] += attenuation * 1.0  # Defect
                                labels[neighbor] = min(1.0, labels[neighbor] + attenuation * 0.3)
                    frontier = next_frontier

        return features


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: Generate demo data
# ═══════════════════════════════════════════════════════════════════════════════

def generate_demo_shift(stations: List[Station], num_steps: int = 30) -> List[Dict]:
    """Quick helper to generate a full shift of demo data with default anomalies."""
    sim = ProductionSimulator(stations, shift_duration_steps=num_steps)
    return sim.generate_shift()


def to_torch_tensors(step_data: Dict, device: str = "cpu") -> Dict:
    """Convert a simulator step output to PyTorch tensors."""
    return {
        'features': torch.tensor(step_data['features'], dtype=torch.float32).to(device),
        'masked_features': torch.tensor(step_data['masked_features'], dtype=torch.float32).to(device),
        'sensor_mask': torch.tensor(step_data['sensor_mask'], dtype=torch.float32).to(device),
        'ground_truth_anomaly': torch.tensor(step_data['ground_truth_anomaly'], dtype=torch.float32).to(device),
        'active_scenarios': step_data['active_scenarios'],
    }
