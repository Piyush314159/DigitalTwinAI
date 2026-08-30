"""
DigitalTwin.ai — Spatio-Temporal Graph Neural Network for Assembly Lines
=========================================================================
Model Architecture (v2 — Round 2):
    - Graph Attention Network (GATv2) layers capture spatial dependencies
    - GRU-based temporal recurrence captures time-series dynamics
    - Masked-node training enables virtual sensing for under-instrumented stations
    - Root-cause attribution via attention weight extraction
    - Per-node anomaly scoring for real-time alerting
    - What-If simulation layer for intervention testing

Prediction Heads (5):
    1. Queue Forecast        — regression, per-node
    2. Defect Risk           — 3-class classification per-node
    3. Virtual Sensor        — feature reconstruction for masked nodes
    4. Anomaly Score         — binary anomaly probability per-node
    5. Root-Cause Attribution — attention-weighted upstream influence

Author : Piyush Maji
Project: Accenture Innovation Challenge 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class SpatioTemporalGATBlock(nn.Module):
    """
    A single Spatio-Temporal block combining:
      1. GATv2 (Graph Attention v2) for spatial message passing across assembly stations
      2. GRU cell for temporal recurrence across time steps
      3. Layer normalization + residual connection + dropout for stable training
    """

    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels,
            hidden_channels // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            concat=True,
        )
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions differ
        self.residual = (
            nn.Linear(in_channels, hidden_channels)
            if in_channels != hidden_channels
            else nn.Identity()
        )

        # Store attention weights for explainability
        self._attention_weights = None

    def forward(self, x, edge_index, h_prev=None):
        """
        Args:
            x          : Node features [num_nodes, in_channels]
            edge_index : Edge connectivity [2, num_edges]
            h_prev     : Previous GRU hidden state [num_nodes, hidden_channels]
        Returns:
            h_new      : Updated hidden state [num_nodes, hidden_channels]
        """
        # Spatial: Graph Attention with attention weight storage
        x_spatial, attn = self.gat(x, edge_index, return_attention_weights=True)
        self._attention_weights = attn  # (edge_index, attention_coefficients)
        x_spatial = F.elu(x_spatial)
        x_spatial = self.dropout(x_spatial)

        # Temporal: GRU recurrence
        if h_prev is None:
            h_prev = torch.zeros_like(x_spatial)
        h_new = self.gru(x_spatial, h_prev)

        # Residual + Norm
        h_new = self.norm(h_new + self.residual(x))

        return h_new


class DigitalTwinGNN(nn.Module):
    """
    Full DigitalTwin.ai Model (v2)
    ================================
    Architecture:
        Input projection → 3× SpatioTemporalGATBlocks → 5 prediction heads

    Prediction Heads:
        1. Queue Forecast    — Predicts queue build-up at each station (regression)
        2. Defect Risk       — Classifies defect-propagation risk per station (3-class)
        3. Virtual Sensor    — Infers sensor readings for under-instrumented stations
        4. Anomaly Score     — Per-node anomaly probability [0, 1]
        5. Root-Cause Attn   — Upstream influence weights for explainability

    Nodes  = Assembly stations (40 stations across 3 zones)
    Edges  = Material/information flow between stations (directed)
    """

    INPUT_DIM = 12       # 12 features per station (expanded from 8)
    HIDDEN_DIM = 128     # Larger hidden dim for 40-station graph
    NUM_DEFECT_CLASSES = 3

    def __init__(self, in_channels=None, hidden_channels=None, heads=4,
                 num_layers=3, dropout=0.1):
        super().__init__()
        in_ch = in_channels or self.INPUT_DIM
        hid_ch = hidden_channels or self.HIDDEN_DIM

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_ch, hid_ch),
            nn.ELU(),
            nn.LayerNorm(hid_ch),
        )

        # Stacked Spatio-Temporal GAT blocks
        self.st_blocks = nn.ModuleList([
            SpatioTemporalGATBlock(
                in_channels=hid_ch,
                hidden_channels=hid_ch,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Prediction Heads ──

        # Head 1: Queue forecast (regression)
        self.queue_head = nn.Sequential(
            nn.Linear(hid_ch, hid_ch // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_ch // 2, 1),
        )

        # Head 2: Defect-propagation risk (3-class)
        self.defect_head = nn.Sequential(
            nn.Linear(hid_ch, hid_ch // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_ch // 2, self.NUM_DEFECT_CLASSES),
        )

        # Head 3: Virtual sensor reconstruction
        self.virtual_sensor_head = nn.Sequential(
            nn.Linear(hid_ch, hid_ch // 2),
            nn.ELU(),
            nn.Linear(hid_ch // 2, in_ch),
        )

        # Head 3b: Confidence for virtual sensing
        self.confidence_head = nn.Sequential(
            nn.Linear(hid_ch, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Head 4: Anomaly score (binary probability)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hid_ch, hid_ch // 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_ch // 4, 1),
            nn.Sigmoid(),
        )

        # Head 5: Root-cause attribution (attention over neighbors)
        self.root_cause_proj = nn.Sequential(
            nn.Linear(hid_ch, hid_ch // 4),
            nn.ELU(),
        )

    def forward(self, x, edge_index, h_states=None, mask=None):
        """
        Args:
            x          : Node features [num_nodes, in_channels]
            edge_index : Edge connectivity [2, num_edges]
            h_states   : List of hidden states for each ST-block (or None)
            mask       : Sensor mask [num_nodes, in_channels] — 1 = observed, 0 = masked

        Returns dict with:
            'queue_forecast'    : [num_nodes, 1]
            'defect_risk'       : [num_nodes, 3] (logits)
            'virtual_sensor'    : [num_nodes, in_channels]
            'confidence'        : [num_nodes, 1]
            'anomaly_score'     : [num_nodes, 1]
            'node_embeddings'   : [num_nodes, hidden_channels]
            'h_states'          : list of updated hidden states
        """
        if mask is not None:
            x = x * mask

        h = self.input_proj(x)

        if h_states is None:
            h_states = [None] * len(self.st_blocks)

        new_h_states = []
        for block, h_prev in zip(self.st_blocks, h_states):
            h = block(h, edge_index, h_prev)
            new_h_states.append(h)

        return {
            'queue_forecast': self.queue_head(h),
            'defect_risk': self.defect_head(h),
            'virtual_sensor': self.virtual_sensor_head(h),
            'confidence': self.confidence_head(h),
            'anomaly_score': self.anomaly_head(h),
            'node_embeddings': h,
            'h_states': new_h_states,
        }

    def get_attention_weights(self) -> list:
        """
        Extract GAT attention weights from all ST-blocks for explainability.

        Returns:
            List of (edge_index, attention_coefficients) tuples, one per block.
        """
        weights = []
        for block in self.st_blocks:
            if block._attention_weights is not None:
                ei, alpha = block._attention_weights
                weights.append((ei.detach().cpu(), alpha.detach().cpu()))
        return weights

    def simulate_intervention(self, x, edge_index, intervention_dict, h_states=None):
        """
        What-If Simulation: Apply hypothetical changes to specific stations,
        then run forward pass to see predicted downstream effects.

        Args:
            x                 : Current node features
            edge_index        : Graph connectivity
            intervention_dict : {node_idx: {feature_idx: new_value, ...}, ...}
            h_states          : Optional hidden states

        Returns:
            Predictions after intervention (same format as forward())
        """
        x_modified = x.clone()
        for node_idx, changes in intervention_dict.items():
            for feat_idx, new_val in changes.items():
                x_modified[node_idx, feat_idx] = new_val

        return self.forward(x_modified, edge_index, h_states)


def load_model(device="cpu"):
    """
    Initialize the DigitalTwinGNN and load weights.

    In production: loads a trained checkpoint.
    For this demo: initializes with seeded random weights.
    """
    torch.manual_seed(42)
    model = DigitalTwinGNN(
        in_channels=12,
        hidden_channels=128,
        heads=4,
        num_layers=3,
        dropout=0.1,
    )
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, total_params, trainable
