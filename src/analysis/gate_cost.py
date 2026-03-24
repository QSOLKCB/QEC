"""
Deterministic analytical gate-cost estimation helpers (v3.0.1).

All functions are pure — no randomness, no simulation, no side effects.
Outputs are JSON-safe dicts with deterministic content.

Supported models
----------------
* ``"qubit_decomp_v1"``
    Placeholder qubit decomposition cost model.
    Estimates the number of two-qubit (CNOT-equivalent) gates needed to
    implement a native d-dimensional gate via qubit decomposition.
    Formula: ceil(log2(d)) qubits per qudit, with O(d^2) two-qubit gates.

* ``"native_placeholder_v1"``
    Placeholder native-qudit gate cost model.
    Estimates native d-dimensional gate counts assuming direct access
    to qudit hardware primitives.  Costs are intentionally optimistic
    placeholders.

Both models are labelled as placeholders and are not intended for
publication-grade analysis.
"""

from __future__ import annotations

import math
from typing import Any

from utils.canonicalize import canonicalize


# ── Supported model identifiers ──────────────────────────────────────

_KNOWN_MODELS = frozenset({"qubit_decomp_v1", "native_placeholder_v1"})


# ── Internal cost helpers ────────────────────────────────────────────

def _qubit_decomp_v1(dimension: int, assumptions: dict[str, Any]) -> dict[str, Any]:
    """Qubit decomposition cost estimates for a d-dimensional system.

    Model (placeholder):
    - qubits_per_qudit = ceil(log2(d))
    - single_qudit_gates (decomposed) = (d - 1)^2 two-qubit gates
    - two_qudit_gates (decomposed) = (d - 1)^2 * 2 two-qubit gates
    - total_ancilla_qubits = 0 (no ancilla assumed in placeholder)
    """
    d = dimension
    qubits_per_qudit = math.ceil(math.log2(d)) if d > 1 else 1
    single_qudit_cost = (d - 1) ** 2
    two_qudit_cost = (d - 1) ** 2 * 2

    return {
        "model": "qubit_decomp_v1",
        "dimension": d,
        "qubits_per_qudit": qubits_per_qudit,
        "single_qudit_gate_cost": single_qudit_cost,
        "two_qudit_gate_cost": two_qudit_cost,
        "ancilla_qubits": 0,
        "notes": "Placeholder qubit decomposition model",
    }


def _native_placeholder_v1(dimension: int, assumptions: dict[str, Any]) -> dict[str, Any]:
    """Native qudit gate cost estimates (optimistic placeholder).

    Model (placeholder):
    - single_qudit_gate_cost = 1  (native single-qudit gate)
    - two_qudit_gate_cost = 1     (native two-qudit gate)
    - qubits_per_qudit = N/A (native)
    - ancilla_qubits = 0
    """
    return {
        "model": "native_placeholder_v1",
        "dimension": dimension,
        "qubits_per_qudit": 0,
        "single_qudit_gate_cost": 1,
        "two_qudit_gate_cost": 1,
        "ancilla_qubits": 0,
        "notes": "Placeholder native-qudit gate cost model (optimistic)",
    }


_MODEL_DISPATCH = {
    "qubit_decomp_v1": _qubit_decomp_v1,
    "native_placeholder_v1": _native_placeholder_v1,
}


# ── Public API ───────────────────────────────────────────────────────

def estimate_gate_costs(
    dimension: int,
    model: str,
    *,
    assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate gate costs for a given dimension using the specified model.

    Parameters
    ----------
    dimension:
        Local Hilbert-space dimension (>= 2).
    model:
        One of the supported model identifiers (see module docstring).
    assumptions:
        Optional dict of model-specific assumptions.  Passed through to
        the model function for future extensibility.

    Returns
    -------
    dict
        JSON-safe dict with cost estimates.  Keys depend on the model
        but always include ``"model"``, ``"dimension"``, and ``"notes"``.

    Raises
    ------
    ValueError
        If *dimension* < 2 or *model* is unknown.
    """
    if dimension < 2:
        raise ValueError(f"dimension must be >= 2, got {dimension}")
    if model not in _MODEL_DISPATCH:
        raise ValueError(
            f"Unknown model {model!r}. Available: {sorted(_MODEL_DISPATCH)}"
        )
    assumptions = assumptions if assumptions is not None else {}
    result = _MODEL_DISPATCH[model](dimension, assumptions)
    result["assumptions"] = canonicalize(assumptions)
    return result


def compare_costs(
    qubit_decomp: dict[str, Any],
    native_qudit: dict[str, Any],
) -> dict[str, Any]:
    """Compare two gate-cost estimates and return a deterministic summary.

    Parameters
    ----------
    qubit_decomp:
        Output of :func:`estimate_gate_costs` with a qubit decomposition model.
    native_qudit:
        Output of :func:`estimate_gate_costs` with a native qudit model.

    Returns
    -------
    dict
        JSON-safe comparison dict with ratio and advantage fields.
    """
    qd_single = qubit_decomp.get("single_qudit_gate_cost", 0)
    nq_single = native_qudit.get("single_qudit_gate_cost", 0)
    qd_two = qubit_decomp.get("two_qudit_gate_cost", 0)
    nq_two = native_qudit.get("two_qudit_gate_cost", 0)

    def _ratio(a: int, b: int) -> float | None:
        if b == 0:
            return None
        return round(a / b, 4)

    return {
        "single_gate_ratio": _ratio(qd_single, nq_single),
        "two_gate_ratio": _ratio(qd_two, nq_two),
        "qubit_decomp_model": qubit_decomp.get("model", "unknown"),
        "native_model": native_qudit.get("model", "unknown"),
        "dimension": qubit_decomp.get("dimension", native_qudit.get("dimension")),
        "notes": "Ratio = qubit_decomp / native; higher means native is more efficient",
    }
