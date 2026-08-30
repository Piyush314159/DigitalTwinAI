# DigitalTwin.ai

### Spatio-Temporal Graph Intelligence for Vehicle Assembly Lines

> **Accenture Innovation Challenge 2026 — Prototype Submission**
> 
> **Author:** Piyush Maji

---

## 🚀 Problem Statement

Modern vehicle assembly lines are tightly coupled networks where small disruptions cascade rapidly. A brief stoppage upstream ripples downstream into bottlenecks and buffer overflows. Defects go undetected until end-of-line QC, causing expensive rework. Current plant monitoring is **static and retrospective** — dashboards and siloed PLC telemetry only alert *after* thresholds are breached, and sparse sensor coverage on legacy equipment leaves blind spots.

**There's no live, interconnected model that captures spatial *and* temporal dependencies across stations to predict problems before they cascade.**

## 💡 Solution

**DigitalTwin.ai** models the entire assembly line as a **dynamic directed graph** — stations as nodes, material flow as edges — powered by a **Spatio-Temporal Graph Neural Network (ST-GNN)**.

### Three Core Capabilities

| Capability | How It Works | Impact |
|---|---|---|
| **Predictive Propagation** | GATv2 + GRU layers learn spatial and temporal dependencies, forecasting queue build-ups minutes in advance and classifying defect-propagation risk downstream | Prevents cascading bottlenecks |
| **Virtual Sensing** | Masked-node training lets the GNN infer state of under-instrumented legacy stations via neighbor aggregation | Extends monitoring to legacy equipment without new hardware |
| **What-If Simulation** | Learned GNN dynamics overlaid on discrete-event line logic for testing interventions virtually | De-risks operational decisions before execution |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DigitalTwin.ai Architecture                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Sensor Data ──→ Graph Construction ──→ Spatio-Temporal GNN  │
│                                             │                 │
│                                    ┌────────┼────────┐       │
│                                    ▼        ▼        ▼       │
│                               Queue    Defect    Virtual     │
│                              Forecast   Risk     Sensor      │
│                                             │                │
│                                    ┌────────┴────────┐       │
│                                    ▼                 ▼       │
│                              What-If           Dashboard     │
│                             Simulation        & Alerts       │
└──────────────────────────────────────────────────────────────┘
```

### Model Details

| Component | Specification |
|---|---|
| **Graph Type** | Dynamic directed graph (12 stations, 14 edges) |
| **Spatial Layer** | GATv2Conv (4 attention heads) |
| **Temporal Layer** | GRU cell per node |
| **Stacked Blocks** | 2 × Spatio-Temporal GAT blocks |
| **Input Features** | 8 per station (queue, cycle time, defect rate, temp, vibration, utilization, throughput, maintenance hours) |
| **Hidden Dimension** | 64 |
| **Prediction Heads** | Queue forecast (regression), Defect risk (3-class), Virtual sensor (reconstruction), Confidence (sigmoid) |
| **Framework** | PyTorch + PyTorch Geometric |

## 📁 Repository Structure

```
accentureInnovation/
├── demo.py                 # Main demo script — run this!
├── model.py                # DigitalTwinGNN model architecture
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── demo_outputs/           # Generated after running demo
    ├── 01_assembly_graph_predictions.png
    ├── 02_temporal_dashboard.png
    └── 03_virtual_sensing.png
```

## ⚡ Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/DigitalTwin.ai.git
cd DigitalTwin.ai

# Install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch Geometric:** If `torch-geometric` installation fails, follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your specific PyTorch + CUDA version.

### Run the Demo

```bash
python demo.py
```

The demo runs in **~15 seconds** and outputs:
1. **Terminal output** — Step-by-step inference with formatted tables
2. **3 visualization PNGs** — Saved to `demo_outputs/`

## 📊 Demo Walkthrough

### Step 1: Graph Construction
Builds a 12-station directed graph representing a vehicle assembly line with forward material flow edges and QC feedback loops.

### Step 2: Synthetic Telemetry
Generates realistic sensor data with **injected anomalies** at Paint Booth (Station 3) and Chassis Marry (Station 7) — high queue lengths, elevated defect rates, overdue maintenance.

### Step 3: Model Loading
Initializes the Spatio-Temporal GNN with ~27K parameters, featuring GATv2 attention layers and GRU temporal recurrence.

### Step 4: Temporal Inference
Runs 10 time steps of inference, demonstrating:
- **Predictive Propagation:** Detects escalating queue build-ups and rising defect risk
- **Virtual Sensing:** Assigns confidence scores to each station's predicted state
- **What-If Simulation:** Tests the effect of resetting maintenance on problem stations

### Step 5: Visualization
Generates three publication-quality dark-themed visualizations:
1. Assembly graph with prediction overlays
2. Temporal evolution dashboard
3. Virtual sensing confidence analysis

## 🎯 Key Features Demonstrated

### 1. Predictive Propagation
The model detects that Paint Booth and Chassis Marry conditions are worsening and predicts downstream impact on Final Assembly and Quality Control — **before** defects cascade.

### 2. Virtual Sensing for Legacy Equipment
Stations with low confidence scores are flagged for manual inspection rather than trusting unreliable sensor data. This extends monitoring to under-instrumented legacy equipment **without** deploying new hardware.

### 3. Interactive What-If Simulation
The demo tests: *"What happens if we perform maintenance on stations 3 and 7?"* — showing predicted queue reduction and risk improvement across the entire line, enabling data-driven maintenance scheduling.

## 🔮 Scalability & Future Work

- **Scale to 100+ stations** with hierarchical graph pooling
- **Real-time streaming** via Apache Kafka + GNN inference server
- **Multi-plant federation** with graph-of-graphs architecture
- **Reinforcement learning** for automated intervention recommendations
- **Integration with MES/ERP** systems for closed-loop optimization

## 📈 Business Impact

| Metric | Expected Improvement |
|---|---|
| Unplanned Downtime | 30-45% reduction |
| Rework Costs | 25-35% reduction |
| Bottleneck Detection Lead Time | From reactive to 10-15 min advance warning |
| Sensor Hardware Costs | 40-60% reduction via virtual sensing |

## 🛠️ Technology Stack

- **Python 3.9+**
- **PyTorch 2.0+** — Deep learning framework
- **PyTorch Geometric** — Graph neural network library
- **NetworkX** — Graph construction and analysis
- **Matplotlib** — Visualization
- **NumPy** — Numerical computing

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**DigitalTwin.ai** — *From reactive firefighting to proactive optimization.*

*Built with ❤️ for the Accenture Innovation Challenge 2026*

</div>
