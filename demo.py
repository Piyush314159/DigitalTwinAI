#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           DigitalTwin.ai — Demo                            ║
║       Spatio-Temporal Graph Intelligence for Vehicle Assembly Lines        ║
║                   Accenture Innovation Challenge 2026                      ║
║                           Author: Piyush Maji                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This CLI demo demonstrates the core GNN inference pipeline on a 40-station
vehicle assembly line with:

  1. Predictive Propagation — forecasting queue build-ups & defect risk
  2. Virtual Sensing       — inferring state at under-instrumented stations
  3. What-If Simulation    — testing interventions before applying them
  4. Anomaly Detection     — multi-causal scenario detection with cascade
  5. Validation Status     — unavailable without trained weights

For the interactive dashboard:  python app.py

Runtime: ~3 seconds (demo-friendly for live presentations)
Usage :  python demo.py
"""

import time
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

# ── Local imports ──
from model import DigitalTwinGNN, load_model, prediction_arrays
from assembly_line import (
    build_assembly_graph, build_stations, get_sensor_mask_tensor,
    get_station_positions, get_zone_assignment,
    Zone, InstrumentationLevel, EquipmentVintage,
    FEATURE_NAMES, NUM_FEATURES, RISK_LABELS, RISK_COLORS, ZONE_COLORS,
)
from simulator import ProductionSimulator, DEFAULT_SCENARIOS, to_torch_tensors

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "demo_outputs"
NUM_STEPS = 20


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def banner(text, width=76):
    print(f"\n{'━' * width}")
    print(f"  ▶  {text}")
    print(f"{'━' * width}")


def sub_banner(text):
    print(f"\n  ┌─ {text}")
    print(f"  └{'─' * 72}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    set_seeds()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Build Assembly Line ──
    banner("STEP 1 / 6 — Building 40-Station Assembly Line Graph")

    stations, edge_index, G = build_assembly_graph()
    num_stations = len(stations)

    full = sum(1 for s in stations if s.instrumentation == InstrumentationLevel.FULL)
    partial = sum(1 for s in stations if s.instrumentation == InstrumentationLevel.PARTIAL)
    manual = sum(1 for s in stations if s.instrumentation == InstrumentationLevel.MANUAL)

    print(f"    ✓ {num_stations} stations across 3 zones (Body={sum(1 for s in stations if s.zone==Zone.BODY)}, "
          f"Paint={sum(1 for s in stations if s.zone==Zone.PAINT)}, "
          f"Final={sum(1 for s in stations if s.zone==Zone.FINAL)})")
    print(f"    ✓ {G.number_of_edges()} edges (forward flow + QC feedback + rework)")
    print(f"    ✓ Instrumentation: FULL={full} ({full/num_stations*100:.0f}%) "
          f"PARTIAL={partial} ({partial/num_stations*100:.0f}%) "
          f"MANUAL={manual} ({manual/num_stations*100:.0f}%)")

    # ── Step 2: Initialize Model ──
    banner("STEP 2 / 6 — Loading Spatio-Temporal GNN Model")

    model, total_params, trainable = load_model(DEVICE)
    print('UNTRAINED DEMO: outputs are not validated predictions or calibrated confidence.')
    edge_index_dev = edge_index.to(DEVICE)
    sensor_mask = get_sensor_mask_tensor(stations).float().to(DEVICE)

    print(f"    ✓ Architecture: GATv2 (4-head) + GRU, 3 ST-blocks")
    print(f"    ✓ Parameters: {total_params:,} ({trainable:,} trainable)")
    print(f"    ✓ Input: {NUM_FEATURES} features × {num_stations} stations")
    print(f"    ✓ Prediction heads: Queue, Defect Risk, Virtual Sensor, Anomaly, Root-Cause")

    # ── Step 3: Generate Simulation Data ──
    banner("STEP 3 / 6 — Generating Production Telemetry")

    simulator = ProductionSimulator(stations, shift_duration_steps=NUM_STEPS)
    shift_data = simulator.generate_shift()

    print(f"    ✓ {NUM_STEPS} time steps × {num_stations} stations × {NUM_FEATURES} features")
    print(f"    ✓ {len(DEFAULT_SCENARIOS)} anomaly scenarios injected:")
    for sc in DEFAULT_SCENARIOS:
        print(f"      • {sc.name} ({sc.category}) — onset at step {sc.onset_step}")

    # ── Step 4: Run Inference Loop ──
    banner("STEP 4 / 6 — Spatio-Temporal Inference")

    h_states = None
    all_predictions = []
    all_gt = []

    for step in range(NUM_STEPS):
        step_data = shift_data[step]
        tensors = to_torch_tensors(step_data, DEVICE)

        with torch.no_grad():
            output = model(
                tensors['masked_features'], edge_index_dev,
                h_states=h_states, mask=sensor_mask
            )
            h_states = output['h_states']

        decoded = prediction_arrays(output)
        queue_pred = decoded['queue']
        anomaly_score = decoded['anomaly_score']
        risk_classes = decoded['risk_class']
        confidence = decoded['confidence']
        gt_anomaly = step_data['ground_truth_anomaly']  # evaluation-only metadata
        features = step_data['features']

        pred = {
            'queue': queue_pred,
            'anomaly_score': anomaly_score,
            'risk_class': risk_classes,
            'confidence': confidence,
            'features': features,
            'active_scenarios': step_data['active_scenarios'],
        }
        all_predictions.append(pred)
        all_gt.append(gt_anomaly)

        # Print step summary
        high = [stations[i].name for i in range(num_stations) if risk_classes[i] == 2]
        med = [stations[i].name for i in range(num_stations) if risk_classes[i] == 1]

        sub_banner(f"Time Step {step}: Inference Complete")
        print(f"    ✓ High-risk stations : {', '.join(high[:3]) or 'None'}")
        print(f"    ✓ Medium-risk        : {', '.join(med[:4]) or 'None'}")

        top3 = sorted(range(num_stations), key=lambda i: queue_pred[i], reverse=True)[:3]
        print(f"    ✓ Top queue build-up : {', '.join(f'{stations[i].short_name}({queue_pred[i]:.0f})' for i in top3)}")

        if step_data['active_scenarios']:
            print(f"    ⚡ Active scenarios  : {', '.join(step_data['active_scenarios'])}")

    # ── Step 5: What-If Simulation ──
    banner("STEP 5 / 6 — What-If Simulation")

    print('    Unavailable: no trained checkpoint or validated intervention model.')

    # ── Step 6: Visualizations ──
    banner("STEP 6 / 6 — Generating Visualizations")

    generate_assembly_graph_viz(stations, G, all_predictions[-1])
    print(f"    ✓ Assembly line graph → {OUTPUT_DIR}/01_assembly_graph_predictions.png")

    generate_temporal_dashboard(stations, all_predictions, all_gt)
    print(f"    ✓ Temporal dashboard  → {OUTPUT_DIR}/02_temporal_dashboard.png")

    generate_virtual_sensing_viz(stations, all_predictions[-1])
    print(f"    ✓ Virtual sensing     → {OUTPUT_DIR}/03_virtual_sensing.png")

    generate_validation_viz(all_predictions, all_gt, stations)
    print(f"    ✓ Validation metrics  → {OUTPUT_DIR}/04_validation_metrics.png")

    t_end = time.time()

    # ── Summary ──
    banner("DEMO COMPLETE")
    print(f"""
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Untrained Architecture Demonstration                          │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  • 40-station assembly line across 3 zones                      │
    │    (Body Construction, Paint Shop, Final Assembly)               │
    │                                                                  │
    │  • 5 simulated anomaly scenarios injected:                   │
    │    - Weld electrode wear (equipment degradation)                │
    │    - Paint booth humidity excursion (environmental)              │
    │    - Stamping die wear (progressive dimensional drift)          │
    │    - Shift-change operator skill drop                           │
    │    - Upstream adhesive quality issue                             │
    │                                                                  │
    │  • Virtual sensing covers {manual} MANUAL + {partial} PARTIAL stations      │
    │    ({(manual+partial)/num_stations*100:.0f}% of line requiring inferred readings)            │
    │                                                                  │
    │  • What-if benefits and model validation are unavailable         │
    │    until a trained checkpoint is independently evaluated            │
    │                                                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │  Runtime    : {t_end - t_start:.1f}s                                          │
    │  Output dir : ./{OUTPUT_DIR}/                                      │
    │  Plots saved: 4 high-resolution PNG files                        │
    │  Dashboard  : python app.py → http://localhost:5001              │
    └──────────────────────────────────────────────────────────────────┘

    DigitalTwin.ai — From reactive firefighting to proactive optimization.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 1: Assembly Graph
# ═══════════════════════════════════════════════════════════════════════════════

def generate_assembly_graph_viz(stations, G, predictions):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), facecolor='#f8fafc')
    ax.set_facecolor('#f8fafc')

    pos = get_station_positions(stations)
    risk_colors_map = {0: '#16a34a', 1: '#d97706', 2: '#dc2626'}

    # Draw edges
    for u, v, d in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if d.get('type') == 'forward':
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='#0284c7', lw=1.2, alpha=0.5))
        else:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='#db2777', lw=0.8,
                                        alpha=0.3, linestyle='dashed',
                                        connectionstyle='arc3,rad=0.2'))

    # Draw nodes
    for s in stations:
        x, y = pos[s.id]
        risk = predictions['risk_class'][s.id]
        queue = predictions['queue'][s.id]
        color = risk_colors_map[risk]
        size = max(200, min(800, 150 + queue * 40))

        ax.scatter(x, y, s=size, c=color, alpha=0.85, edgecolors='#f8fafc', linewidth=1.5, zorder=5)
        ax.text(x, y, f"Q:{queue:.0f}", ha='center', va='center', fontsize=6,
                fontweight='bold', color='#1e293b', zorder=6)
        ax.text(x, y - 0.35, s.short_name, ha='center', va='top', fontsize=6,
                color='#64748b', zorder=6)

    # Zone labels
    zone_groups = get_zone_assignment(stations)
    for zone, zone_color in ZONE_COLORS.items():
        ids = zone_groups[zone]
        xs = [pos[i][0] for i in ids]
        ys = [pos[i][1] for i in ids]
        cx, cy = np.mean(xs), max(ys) + 0.6
        ax.text(cx, cy, zone.value.upper(), ha='center', fontsize=10, fontweight='bold',
                color=zone_color, alpha=0.6)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#16a34a', label='Low Risk'),
        mpatches.Patch(facecolor='#d97706', label='Medium Risk'),
        mpatches.Patch(facecolor='#dc2626', label='High Risk'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              facecolor='#f1f5f9', edgecolor='#cbd5e1', labelcolor='#1e293b')

    ax.set_xlim(-1, max(p[0] for p in pos.values()) + 1)
    ax.set_ylim(min(p[1] for p in pos.values()) - 1, max(p[1] for p in pos.values()) + 1.5)
    ax.axis('off')
    ax.set_title("DigitalTwin.ai — 40-Station Assembly Line with Predictive Overlay",
                 fontsize=14, fontweight='bold', color='#0f172a', pad=15)

    fig.text(0.5, 0.02,
             "Node color = Defect Risk  |  Node size = Predicted Queue  |  "
             "Blue arrows = Material flow  |  Pink dashed = QC feedback",
             ha='center', fontsize=9, color='#64748b')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.gcf().text(0.5, 0.005, 'UNTRAINED DEMO - SIMULATED DATA - NOT VALIDATED PERFORMANCE', ha='center', fontsize=9, color='#b45309')
    plt.savefig(f"{OUTPUT_DIR}/01_assembly_graph_predictions.png", dpi=150,
                facecolor='#f8fafc', bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 2: Temporal Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def generate_temporal_dashboard(stations, all_predictions, all_gt):
    fig = plt.figure(figsize=(20, 12), facecolor='#f8fafc')
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    num_steps = len(all_predictions)
    num_stations = len(stations)

    # ── Panel 1: Queue Heatmap ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#ffffff')
    queue_matrix = np.array([p['queue'] for p in all_predictions]).T
    cmap = LinearSegmentedColormap.from_list('queue', ['#064e3b', '#22c55e', '#fbbf24', '#ef4444'])
    im = ax1.imshow(queue_matrix, aspect='auto', cmap=cmap, interpolation='bilinear')
    ax1.set_xlabel("Time Step", color='#94a3b8', fontsize=9)
    ax1.set_ylabel("Station", color='#94a3b8', fontsize=9)
    ax1.set_title("Queue Build-up Forecast", fontsize=11, fontweight='bold', color='#0f172a')
    ax1.set_yticks(range(0, num_stations, 3))
    ax1.set_yticklabels([stations[i].short_name for i in range(0, num_stations, 3)],
                        fontsize=6, color='#64748b')
    ax1.tick_params(colors='#94a3b8', labelsize=7)
    plt.colorbar(im, ax=ax1, label='Predicted Queue', shrink=0.8)

    # ── Panel 2: Risk Evolution ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#ffffff')

    # Track interesting stations (those that reach HIGH risk)
    interesting = set()
    for p in all_predictions:
        for i in range(num_stations):
            if p['risk_class'][i] >= 1:
                interesting.add(i)

    colors_cycle = ['#f472b6', '#38bdf8', '#a78bfa', '#34d399', '#fbbf24',
                    '#fb923c', '#22d3ee', '#c084fc', '#f87171', '#4ade80']

    for idx, sid in enumerate(sorted(interesting)[:8]):
        risk_over_time = [p['risk_class'][sid] for p in all_predictions]
        ax2.plot(range(num_steps), risk_over_time, '-o', markersize=4,
                 label=stations[sid].short_name, color=colors_cycle[idx % len(colors_cycle)],
                 linewidth=1.5, alpha=0.8)

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['LOW', 'MEDIUM', 'HIGH'], fontsize=9, color='#64748b')
    ax2.set_xlabel("Time Step", color='#94a3b8', fontsize=9)
    ax2.set_ylabel("Risk Class", color='#94a3b8', fontsize=9)
    ax2.set_title("Defect Risk Propagation", fontsize=11, fontweight='bold', color='#0f172a')
    ax2.legend(fontsize=6, ncol=2, loc='lower right',
               facecolor='#f1f5f9', edgecolor='#cbd5e1', labelcolor='#1e293b')
    ax2.tick_params(colors='#94a3b8', labelsize=7)
    ax2.set_ylim(-0.2, 2.5)
    ax2.grid(True, alpha=0.1)

    # ── Panel 3: Anomaly Score Evolution ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#ffffff')

    anomaly_matrix = np.array([p['anomaly_score'] for p in all_predictions]).T
    cmap2 = LinearSegmentedColormap.from_list('anomaly', ['#064e3b', '#22c55e', '#fbbf24', '#ef4444'])
    im2 = ax3.imshow(anomaly_matrix, aspect='auto', cmap=cmap2, interpolation='bilinear',
                     vmin=0, vmax=1)
    ax3.set_xlabel("Time Step", color='#94a3b8', fontsize=9)
    ax3.set_ylabel("Station", color='#94a3b8', fontsize=9)
    ax3.set_title("Anomaly Score Heatmap", fontsize=11, fontweight='bold', color='#0f172a')
    ax3.set_yticks(range(0, num_stations, 3))
    ax3.set_yticklabels([stations[i].short_name for i in range(0, num_stations, 3)],
                        fontsize=6, color='#64748b')
    ax3.tick_params(colors='#94a3b8', labelsize=7)
    plt.colorbar(im2, ax=ax3, label='Anomaly Score', shrink=0.8)

    # ── Panel 4: Scenario Activation Timeline ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#ffffff')

    scenario_names = [s.name for s in DEFAULT_SCENARIOS]
    for idx, sc in enumerate(DEFAULT_SCENARIOS):
        active_steps = [t for t in range(num_steps) if t >= sc.onset_step]
        if active_steps:
            ax4.barh(idx, len(active_steps), left=active_steps[0],
                     height=0.6, color=colors_cycle[idx % len(colors_cycle)], alpha=0.7)
            ax4.plot(sc.onset_step, idx, 'k*', markersize=8, zorder=5)

    ax4.set_yticks(range(len(scenario_names)))
    ax4.set_yticklabels([n[:25] for n in scenario_names], fontsize=7, color='#64748b')
    ax4.set_xlabel("Time Step", color='#94a3b8', fontsize=9)
    ax4.set_title("Anomaly Scenario Timeline", fontsize=11, fontweight='bold', color='#0f172a')
    ax4.tick_params(colors='#94a3b8', labelsize=7)
    ax4.set_xlim(-0.5, num_steps)
    ax4.grid(True, alpha=0.1, axis='x')

    fig.suptitle("DigitalTwin.ai — Spatio-Temporal Prediction Dashboard",
                 fontsize=14, fontweight='bold', color='#0f172a', y=0.98)

    plt.gcf().text(0.5, 0.005, 'UNTRAINED DEMO - SIMULATED DATA - NOT VALIDATED PERFORMANCE', ha='center', fontsize=9, color='#b45309')
    plt.savefig(f"{OUTPUT_DIR}/02_temporal_dashboard.png", dpi=150,
                facecolor='#f8fafc', bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 3: Virtual Sensing
# ═══════════════════════════════════════════════════════════════════════════════

def generate_virtual_sensing_viz(stations, predictions):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor='#f8fafc',
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    # Sort stations by confidence
    sorted_stations = sorted(range(len(stations)),
                             key=lambda i: predictions['confidence'][i], reverse=True)

    names = [stations[i].name for i in sorted_stations]
    confs = [predictions['confidence'][i] for i in sorted_stations]
    instr = [stations[i].instrumentation for i in sorted_stations]

    colors = []
    for c, inst in zip(confs, instr):
        if inst == InstrumentationLevel.FULL:
            colors.append('#22c55e')
        elif inst == InstrumentationLevel.PARTIAL:
            colors.append('#f59e0b')
        else:
            colors.append('#ef4444')

    ax1.set_facecolor('#ffffff')
    bars = ax1.barh(range(len(names)), confs, color=colors, alpha=0.8, height=0.7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7, color='#64748b')
    ax1.set_xlabel("Confidence Score", fontsize=10, color='#64748b')
    ax1.set_title("Virtual Sensor Confidence Ranking", fontsize=12, fontweight='bold', color='#0f172a')
    ax1.axvline(x=0.5, color='#ef4444', linestyle='--', linewidth=1, alpha=0.7)
    ax1.tick_params(colors='#94a3b8')
    ax1.invert_yaxis()

    # Add confidence values as text
    for i, (c, bar) in enumerate(zip(confs, bars)):
        ax1.text(c + 0.02, i, f"{c:.2f}", va='center', fontsize=7, color='#1e293b', fontweight='bold')

    # Legend
    legend_els = [
        mpatches.Patch(facecolor='#16a34a', label='FULL Instrumentation'),
        mpatches.Patch(facecolor='#d97706', label='PARTIAL Instrumentation'),
        mpatches.Patch(facecolor='#dc2626', label='MANUAL (Checklist Only)'),
    ]
    ax1.legend(handles=legend_els, loc='lower right', fontsize=7,
               facecolor='#f1f5f9', edgecolor='#cbd5e1', labelcolor='#1e293b')

    # ── Right panel: Feature reconstruction comparison ──
    ax2.set_facecolor('#ffffff')
    lowest = sorted_stations[-1]
    highest = sorted_stations[0]

    feat_low = predictions['features'][lowest]
    feat_high = predictions['features'][highest]

    x = np.arange(NUM_FEATURES)
    width = 0.35

    feat_low_norm = feat_low / (np.max(np.abs(feat_low)) + 1e-6)
    feat_high_norm = feat_high / (np.max(np.abs(feat_high)) + 1e-6)

    ax2.bar(x - width/2, np.abs(feat_low_norm), width, color='#ef4444', alpha=0.8,
            label=f'{stations[lowest].name} (Low Conf)')
    ax2.bar(x + width/2, np.abs(feat_high_norm), width, color='#22c55e', alpha=0.8,
            label=f'{stations[highest].name} (High Conf)')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace('_', '\n') for f in FEATURE_NAMES], fontsize=6,
                        color='#94a3b8', rotation=45, ha='right')
    ax2.set_ylabel("Normalized Value", fontsize=10, color='#64748b')
    ax2.set_title("Reconstructed Feature Comparison", fontsize=12, fontweight='bold', color='#0f172a')
    ax2.legend(fontsize=8, facecolor='#f1f5f9', edgecolor='#cbd5e1', labelcolor='#1e293b')
    ax2.tick_params(colors='#94a3b8')

    fig.suptitle("DigitalTwin.ai — Virtual Sensing for Legacy Equipment",
                 fontsize=14, fontweight='bold', color='#0f172a', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.gcf().text(0.5, 0.005, 'UNTRAINED DEMO - SIMULATED DATA - NOT VALIDATED PERFORMANCE', ha='center', fontsize=9, color='#b45309')
    plt.savefig(f"{OUTPUT_DIR}/03_virtual_sensing.png", dpi=150,
                facecolor='#f8fafc', bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 4: Validation Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def generate_validation_viz(all_predictions, all_gt, stations):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.text(0.5, 0.5, 'Validation unavailable\nNo trained checkpoint is supplied.\nSimulator outcomes are excluded from model predictions.',
            ha='center', va='center', fontsize=16, transform=ax.transAxes)
    fig.savefig(f'{OUTPUT_DIR}/04_validation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
