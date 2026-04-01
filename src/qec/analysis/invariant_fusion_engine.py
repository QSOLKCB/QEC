"""v127.0.0 — Deterministic invariant fusion engine.

Fuses logical, topology, and control-state invariant risks into a stable,
bounded risk score with deterministic label thresholds.
"""

from __future__ import annotations

import math
from typing import Any, Dict, TypedDict


class FusedInvariantRisks(TypedDict):
    logical_risk: float
    topology_risk: float
    control_risk: float
    fused_risk_score: float
    fused_risk_label: str


def normalize_risk_signal(value: float) -> float:
    """Normalize a numeric risk signal to [0.0, 1.0] deterministically."""
    numeric = float(value)

    if math.isnan(numeric):
        return 0.0
    if math.isinf(numeric):
        return 1.0 if numeric > 0.0 else 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def compute_logical_risk(proof_result: Dict[str, Any]) -> float:
    """Map proof status to deterministic logical risk."""
    proof_status = str(proof_result.get("proof_status", "violated"))

    if proof_status == "valid":
        return 0.0
    if proof_status == "partial":
        return 0.5
    return 1.0


def compute_topology_risk(topology_result: Dict[str, Any]) -> float:
    """Map topology risk status to deterministic topology risk."""
    topology_risk = str(topology_result.get("topology_risk", "critical"))

    if topology_risk == "safe":
        return 0.0
    if topology_risk == "warning":
        return 0.5
    return 1.0


def compute_control_risk(controller_result: Dict[str, Any]) -> float:
    """Map controller fail-safe/hysteresis state to deterministic control risk."""
    fail_safe_triggered = bool(controller_result.get("fail_safe_triggered", False))
    hysteresis_active = bool(controller_result.get("hysteresis_active", False))

    if fail_safe_triggered:
        return 1.0
    if hysteresis_active:
        return 0.5
    return 0.0


def fuse_invariants(
    proof_result: Dict[str, Any],
    topology_result: Dict[str, Any],
    controller_result: Dict[str, Any],
) -> FusedInvariantRisks:
    """Fuse logical/topology/control invariant risks into a stable schema."""
    logical_risk = normalize_risk_signal(compute_logical_risk(proof_result))
    topology_risk = normalize_risk_signal(compute_topology_risk(topology_result))
    control_risk = normalize_risk_signal(compute_control_risk(controller_result))

    fused_risk_score = normalize_risk_signal(
        (0.4 * logical_risk) + (0.3 * topology_risk) + (0.3 * control_risk)
    )

    if fused_risk_score < 0.35:
        fused_risk_label = "safe"
    elif fused_risk_score < 0.70:
        fused_risk_label = "warning"
    else:
        fused_risk_label = "critical"

    return {
        "logical_risk": logical_risk,
        "topology_risk": topology_risk,
        "control_risk": control_risk,
        "fused_risk_score": fused_risk_score,
        "fused_risk_label": fused_risk_label,
    }
