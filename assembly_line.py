"""
DigitalTwin.ai — Assembly Line Definition (40 Stations)
=========================================================
Defines the complete assembly line topology for a mixed-model vehicle
production facility with ~40 stations across three zones:
  Zone A: Body Construction (13 stations)
  Zone B: Paint Shop (11 stations)
  Zone C: Final Assembly (16 stations)

Each station has metadata for:
  - Zone membership
  - Instrumentation level (FULL / PARTIAL / MANUAL)
  - Equipment vintage (MODERN / LEGACY)
  - Nominal takt time (seconds)
  - Maintenance window frequency

Author : Piyush Maji
Project: Accenture Innovation Challenge 2026
"""

import torch
import networkx as nx
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class Zone(Enum):
    BODY = "Body Construction"
    PAINT = "Paint Shop"
    FINAL = "Final Assembly"


class InstrumentationLevel(Enum):
    FULL = "full"         # Rich sensor array: all 12 features available
    PARTIAL = "partial"   # Basic PLC data: ~7-8 features, rest inferred
    MANUAL = "manual"     # Checklist-only: ~3-4 features, rest must be inferred


class EquipmentVintage(Enum):
    MODERN = "modern"     # < 5 years, IoT-enabled, OPC-UA ready
    LEGACY = "legacy"     # > 10 years, basic PLC, serial/analog interfaces


# ═══════════════════════════════════════════════════════════════════════════════
# STATION DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Station:
    id: int
    name: str
    short_name: str
    zone: Zone
    instrumentation: InstrumentationLevel
    vintage: EquipmentVintage
    takt_time_sec: float       # Nominal cycle time in seconds
    maint_windows_per_year: int  # Scheduled maintenance windows
    description: str = ""

    @property
    def sensor_mask(self) -> List[bool]:
        """
        Returns a boolean mask indicating which of the 12 features are
        directly observable at this station (vs. must be inferred).

        Features:
          0: queue_length       — always observable (visual count)
          1: cycle_time         — always observable (PLC timer or manual)
          2: defect_rate        — always observable (inspection records)
          3: temperature        — FULL/PARTIAL only
          4: vibration          — FULL only
          5: torque_deviation   — FULL only
          6: utilization        — always observable (uptime/downtime)
          7: throughput         — always observable (unit count)
          8: humidity           — FULL/PARTIAL only
          9: power_draw         — FULL only
         10: hrs_since_maint    — always observable (maintenance logs)
         11: operator_skill_idx — always observable (HR training records)
        """
        if self.instrumentation == InstrumentationLevel.FULL:
            return [True] * 12
        elif self.instrumentation == InstrumentationLevel.PARTIAL:
            # Missing: vibration, torque_deviation, power_draw
            return [True, True, True, True, False, False, True, True, True, False, True, True]
        else:  # MANUAL
            # Only: queue, cycle_time, defect_rate, utilization, throughput, maint, operator
            return [True, True, True, False, False, False, True, True, False, False, True, True]


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "queue_length",
    "cycle_time_sec",
    "defect_rate_%",
    "temperature_C",
    "vibration_mm/s",
    "torque_dev_%",
    "utilization_%",
    "throughput_uph",
    "humidity_%",
    "power_draw_kW",
    "hrs_since_maint",
    "operator_skill",
]

NUM_FEATURES = len(FEATURE_NAMES)

# ═══════════════════════════════════════════════════════════════════════════════
# RISK / DEFECT LABELS
# ═══════════════════════════════════════════════════════════════════════════════

RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]
RISK_COLORS = ["#16a34a", "#d97706", "#dc2626"]

ZONE_COLORS = {
    Zone.BODY: "#0284c7",   # Blue
    Zone.PAINT: "#7c3aed",  # Violet
    Zone.FINAL: "#059669",  # Teal
}

# ═══════════════════════════════════════════════════════════════════════════════
# STATION DEFINITIONS — 40 STATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_stations() -> List[Station]:
    """
    Define all 40 stations across 3 production zones.

    Instrumentation distribution:
      FULL:    24 stations (60%)
      PARTIAL: 10 stations (25%)
      MANUAL:   6 stations (15%)
    """
    stations = [
        # ──────────────────────────────────────────────────────────────
        # ZONE A: BODY CONSTRUCTION (13 stations, IDs 0-12)
        # ──────────────────────────────────────────────────────────────
        Station(0,  "Blanking Press",       "BLANK",   Zone.BODY, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 45,  4, "Sheet metal blanking from coils"),
        Station(1,  "Stamping Press A",     "STMP-A",  Zone.BODY, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 52,  4, "Major body panel stamping (doors, hood)"),
        Station(2,  "Stamping Press B",     "STMP-B",  Zone.BODY, InstrumentationLevel.FULL,    EquipmentVintage.LEGACY, 55,  3, "Secondary panel stamping (fenders, roof)"),
        Station(3,  "Laser Cutting",        "LASER",   Zone.BODY, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 38,  4, "Precision laser trimming of blanks"),
        Station(4,  "Hemming Station",      "HEM",     Zone.BODY, InstrumentationLevel.PARTIAL, EquipmentVintage.LEGACY, 60,  2, "Panel edge hemming and folding"),
        Station(5,  "Spot Weld Cell 1",     "WELD-1",  Zone.BODY, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 48,  4, "Robotic spot welding — underbody"),
        Station(6,  "Spot Weld Cell 2",     "WELD-2",  Zone.BODY, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 48,  4, "Robotic spot welding — body sides"),
        Station(7,  "Spot Weld Cell 3",     "WELD-3",  Zone.BODY, InstrumentationLevel.PARTIAL, EquipmentVintage.LEGACY, 55,  2, "Manual-assist welding — closures"),
        Station(8,  "Adhesive Bonding",     "BOND",    Zone.BODY, InstrumentationLevel.PARTIAL, EquipmentVintage.LEGACY, 42,  2, "Structural adhesive application"),
        Station(9,  "Geometry Check",       "GEO",     Zone.BODY, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 35,  4, "CMM-based body geometry measurement"),
        Station(10, "BiW Audit",            "BIW-AUD", Zone.BODY, InstrumentationLevel.MANUAL,  EquipmentVintage.LEGACY, 90,  2, "Body-in-White manual quality audit"),
        Station(11, "Sealer Application",   "SEAL",    Zone.BODY, InstrumentationLevel.PARTIAL, EquipmentVintage.LEGACY, 50,  2, "Underbody and joint sealer application"),
        Station(12, "Underbody Coating",    "UCOAT",   Zone.BODY, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 40,  4, "Anti-corrosion underbody spray"),

        # ──────────────────────────────────────────────────────────────
        # ZONE B: PAINT SHOP (11 stations, IDs 13-23)
        # ──────────────────────────────────────────────────────────────
        Station(13, "Pretreatment Bath",    "PRETR",   Zone.PAINT, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 120, 4, "Chemical pretreatment & phosphate dip"),
        Station(14, "E-Coat Dip Tank",      "ECOAT",   Zone.PAINT, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 180, 4, "Electrodeposition primer coating"),
        Station(15, "E-Coat Cure Oven",     "EC-CURE", Zone.PAINT, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 200, 3, "E-coat baking/curing oven"),
        Station(16, "Seam Seal Booth",      "SEAM",    Zone.PAINT, InstrumentationLevel.PARTIAL, EquipmentVintage.LEGACY, 65,  2, "Robotic + manual seam sealing"),
        Station(17, "Primer Spray Booth",   "PRIMER",  Zone.PAINT, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 55,  4, "Automated primer application"),
        Station(18, "Primer Cure Oven",     "PR-CURE", Zone.PAINT, InstrumentationLevel.PARTIAL, EquipmentVintage.LEGACY, 180, 2, "Primer baking oven"),
        Station(19, "Color Spray Booth",    "COLOR",   Zone.PAINT, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 60,  4, "Base coat color application"),
        Station(20, "Color Cure Oven",      "CL-CURE", Zone.PAINT, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 180, 3, "Base coat curing"),
        Station(21, "Clear Coat Booth",     "CLEAR",   Zone.PAINT, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 55,  4, "Clear coat application"),
        Station(22, "Clear Cure Oven",      "CC-CURE", Zone.PAINT, InstrumentationLevel.PARTIAL, EquipmentVintage.LEGACY, 180, 2, "Clear coat final cure"),
        Station(23, "Paint Inspection",     "P-INSP",  Zone.PAINT, InstrumentationLevel.MANUAL,  EquipmentVintage.LEGACY, 120, 2, "Visual paint quality inspection"),

        # ──────────────────────────────────────────────────────────────
        # ZONE C: FINAL ASSEMBLY (16 stations, IDs 24-39)
        # ──────────────────────────────────────────────────────────────
        Station(24, "Wiring Harness",       "WIRE",    Zone.FINAL, InstrumentationLevel.MANUAL,  EquipmentVintage.LEGACY, 95,  2, "Main wiring harness installation"),
        Station(25, "Dashboard Install",    "DASH",    Zone.FINAL, InstrumentationLevel.PARTIAL, EquipmentVintage.LEGACY, 110, 2, "Instrument panel & cockpit module"),
        Station(26, "Headliner Install",    "HEAD",    Zone.FINAL, InstrumentationLevel.MANUAL,  EquipmentVintage.LEGACY, 75,  2, "Roof headliner installation"),
        Station(27, "Door Trim Assembly",   "DTRIM",   Zone.FINAL, InstrumentationLevel.MANUAL,  EquipmentVintage.LEGACY, 85,  2, "Door panel trim and hardware"),
        Station(28, "Glass Installation",   "GLASS",   Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 70,  4, "Windshield & window bonding (robotic)"),
        Station(29, "Seat Installation",    "SEAT",    Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 65,  4, "Front & rear seat drop and bolt"),
        Station(30, "Chassis Marriage",     "MARRY",   Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 80,  4, "Body-to-chassis mating"),
        Station(31, "Engine/Motor Drop",    "ENGN",    Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 90,  4, "Powertrain installation"),
        Station(32, "Suspension Torque",    "SUSP",    Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 70,  4, "Suspension bolt torquing"),
        Station(33, "Brake Line Connect",   "BRAKE",   Zone.FINAL, InstrumentationLevel.PARTIAL, EquipmentVintage.LEGACY, 55,  3, "Brake line routing & connection"),
        Station(34, "Fluid Fill Station",   "FLUID",   Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 50,  4, "All fluids fill (coolant, oil, brake, washer)"),
        Station(35, "Wheel Mounting",       "WHEEL",   Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 45,  4, "Automated wheel mounting & torque"),
        Station(36, "Battery/Fuel System",  "BATT",    Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 60,  4, "HV battery or fuel system install"),
        Station(37, "Final Torque Audit",   "TORQ-A",  Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 55,  4, "Critical torque verification (electronic)"),
        Station(38, "ADAS Calibration",     "ADAS",    Zone.FINAL, InstrumentationLevel.FULL,    EquipmentVintage.MODERN, 120, 4, "Camera & radar alignment calibration"),
        Station(39, "Roll Test & Dispatch", "ROLL",    Zone.FINAL, InstrumentationLevel.MANUAL,  EquipmentVintage.LEGACY, 180, 2, "Dynamic test, water test, final sign-off"),
    ]
    return stations


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

def build_edges() -> List[Tuple[int, int]]:
    """
    Define directed edges representing material flow, information flow,
    and quality feedback loops across the 40-station assembly line.

    Edge types:
      - Forward flow: material moves from upstream to downstream
      - Parallel merge: multiple feeder stations converge
      - QC feedback: inspection results flow back upstream
      - Rework loops: defective units return to earlier stations
    """
    edges = [
        # ── Zone A: Body Construction ──
        # Main flow: Blanking → Stamping → Processing → Welding → Assembly
        (0, 1),    # Blanking → Stamping A
        (0, 2),    # Blanking → Stamping B (parallel press)
        (1, 3),    # Stamping A → Laser Cutting
        (2, 3),    # Stamping B → Laser Cutting (merge)
        (3, 4),    # Laser → Hemming
        (3, 5),    # Laser → Spot Weld 1 (some parts skip hemming)
        (4, 5),    # Hemming → Spot Weld 1
        (4, 6),    # Hemming → Spot Weld 2 (parallel weld cells)
        (5, 7),    # Spot Weld 1 → Spot Weld 3
        (6, 7),    # Spot Weld 2 → Spot Weld 3 (merge)
        (7, 8),    # Spot Weld 3 → Adhesive Bonding
        (8, 9),    # Adhesive → Geometry Check
        (9, 10),   # Geometry → BiW Audit
        (10, 11),  # BiW Audit → Sealer
        (11, 12),  # Sealer → Underbody Coating

        # ── Zone A → Zone B transition ──
        (12, 13),  # Underbody Coating → Pretreatment Bath

        # ── Zone B: Paint Shop ──
        (13, 14),  # Pretreatment → E-Coat
        (14, 15),  # E-Coat → E-Coat Cure
        (15, 16),  # E-Coat Cure → Seam Seal
        (16, 17),  # Seam Seal → Primer
        (17, 18),  # Primer → Primer Cure
        (18, 19),  # Primer Cure → Color Booth
        (19, 20),  # Color → Color Cure
        (20, 21),  # Color Cure → Clear Coat
        (21, 22),  # Clear Coat → Clear Cure
        (22, 23),  # Clear Cure → Paint Inspection

        # ── Zone B → Zone C transition ──
        (23, 24),  # Paint Inspection → Wiring Harness

        # ── Zone C: Final Assembly ──
        # Interior trim sub-line
        (24, 25),  # Wiring → Dashboard
        (25, 26),  # Dashboard → Headliner
        (26, 27),  # Headliner → Door Trim
        (27, 28),  # Door Trim → Glass
        (28, 29),  # Glass → Seats

        # Mechanical sub-line (merges at chassis marriage)
        (29, 30),  # Seats → Chassis Marriage
        (30, 31),  # Chassis Marriage → Engine Drop
        (31, 32),  # Engine → Suspension
        (32, 33),  # Suspension → Brake Lines
        (33, 34),  # Brake → Fluid Fill
        (34, 35),  # Fluid Fill → Wheels
        (35, 36),  # Wheels → Battery/Fuel
        (36, 37),  # Battery → Final Torque Audit
        (37, 38),  # Torque Audit → ADAS Calibration
        (38, 39),  # ADAS → Roll Test & Dispatch

        # ── QC Feedback Loops (information flow upstream) ──
        (9, 5),    # Geometry Check → Spot Weld 1 (weld quality feedback)
        (9, 6),    # Geometry Check → Spot Weld 2
        (10, 7),   # BiW Audit → Spot Weld 3 (closure fit feedback)
        (10, 8),   # BiW Audit → Adhesive Bonding
        (23, 17),  # Paint Inspection → Primer Booth (paint defect feedback)
        (23, 19),  # Paint Inspection → Color Booth
        (23, 21),  # Paint Inspection → Clear Coat
        (37, 30),  # Torque Audit → Chassis Marriage (torque spec feedback)
        (37, 32),  # Torque Audit → Suspension
        (39, 37),  # Roll Test → Torque Audit (final pass/fail → audit station)
        (39, 38),  # Roll Test → ADAS Cal (alignment/calibration redo)
        (39, 23),  # Roll Test → Paint Inspection (cosmetic issues spotted at end)

        # ── Rework / bypass edges ──
        (10, 5),   # BiW Audit fail → rework at Spot Weld 1
        (23, 16),  # Paint fail → rework at Seam Seal
    ]
    return edges


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_assembly_graph() -> Tuple[List[Station], torch.LongTensor, nx.DiGraph]:
    """
    Build the complete assembly line graph.

    Returns:
        stations   : List of Station objects (length 40)
        edge_index : torch.LongTensor [2, num_edges] for PyTorch Geometric
        G          : networkx.DiGraph for visualization
    """
    stations = build_stations()
    edges = build_edges()

    # Build PyTorch Geometric edge_index
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Build NetworkX graph
    G = nx.DiGraph()
    for s in stations:
        G.add_node(s.id, label=s.name, short=s.short_name,
                   zone=s.zone.value, instrumentation=s.instrumentation.value,
                   vintage=s.vintage.value, takt=s.takt_time_sec)

    # Tag edges as forward, feedback, or rework
    for src_id, dst_id in edges:
        if dst_id < src_id:
            edge_type = "feedback"
        else:
            edge_type = "forward"
        G.add_edge(src_id, dst_id, type=edge_type)

    return stations, edge_index, G


def get_sensor_mask_tensor(stations: List[Station]) -> torch.BoolTensor:
    """
    Build a [num_stations, num_features] boolean mask tensor.
    True = feature is directly observed, False = must be inferred.
    """
    masks = [s.sensor_mask for s in stations]
    return torch.tensor(masks, dtype=torch.bool)


def get_zone_assignment(stations: List[Station]) -> Dict[Zone, List[int]]:
    """Group station IDs by zone."""
    zones = {z: [] for z in Zone}
    for s in stations:
        zones[s.zone].append(s.id)
    return zones


def get_station_positions(stations: List[Station]) -> Dict[int, Tuple[float, float]]:
    """
    Generate layout positions for graph visualization.
    Stations are arranged in a serpentine layout by zone.
    """
    pos = {}
    zone_groups = get_zone_assignment(stations)

    y_offset = 0
    for zone in [Zone.BODY, Zone.PAINT, Zone.FINAL]:
        ids = zone_groups[zone]
        n = len(ids)

        # Two-row serpentine within each zone
        for local_idx, sid in enumerate(ids):
            row = local_idx % 2
            col = local_idx // 2
            x = col * 1.8
            y = y_offset - row * 1.2
            pos[sid] = (x, y)

        y_offset -= 3.5  # Space between zones

    return pos


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY / STATS
# ═══════════════════════════════════════════════════════════════════════════════

def print_assembly_stats(stations: List[Station], G: nx.DiGraph):
    """Print summary statistics about the assembly line."""
    total = len(stations)
    full = sum(1 for s in stations if s.instrumentation == InstrumentationLevel.FULL)
    partial = sum(1 for s in stations if s.instrumentation == InstrumentationLevel.PARTIAL)
    manual = sum(1 for s in stations if s.instrumentation == InstrumentationLevel.MANUAL)
    modern = sum(1 for s in stations if s.vintage == EquipmentVintage.MODERN)
    legacy = sum(1 for s in stations if s.vintage == EquipmentVintage.LEGACY)

    forward = sum(1 for u, v, d in G.edges(data=True) if d.get('type') == 'forward')
    feedback = sum(1 for u, v, d in G.edges(data=True) if d.get('type') == 'feedback')

    print(f"""
    Assembly Line Configuration
    ───────────────────────────────────────
    Total Stations    : {total}
    ├─ Body Construction : {sum(1 for s in stations if s.zone == Zone.BODY)}
    ├─ Paint Shop        : {sum(1 for s in stations if s.zone == Zone.PAINT)}
    └─ Final Assembly    : {sum(1 for s in stations if s.zone == Zone.FINAL)}

    Instrumentation   :
    ├─ FULL    : {full:2d} ({full/total*100:.0f}%)
    ├─ PARTIAL : {partial:2d} ({partial/total*100:.0f}%)
    └─ MANUAL  : {manual:2d} ({manual/total*100:.0f}%)

    Equipment Vintage :
    ├─ Modern  : {modern:2d} ({modern/total*100:.0f}%)
    └─ Legacy  : {legacy:2d} ({legacy/total*100:.0f}%)

    Graph Topology    :
    ├─ Forward flow edges  : {forward}
    ├─ Feedback/rework     : {feedback}
    └─ Total edges         : {G.number_of_edges()}
    """)
