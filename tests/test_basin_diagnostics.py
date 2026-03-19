from __future__ import annotations

import numpy as np

from qec.analysis.basin_diagnostics import BasinDiagnostics, BasinDiagnosticsConfig


def test_basin_state_classification_sequence() -> None:
    diag = BasinDiagnostics(
        config=BasinDiagnosticsConfig(
            energy_window=3,
            slope_threshold=1e-4,
            ipr_threshold=0.2,
            trap_ipr_threshold=0.4,
            edge_reuse_threshold=0.5,
        ),
    )

    free = diag.update(
        energy=10.0,
        unstable_modes=2,
        flow=np.array([0.6, 0.4], dtype=np.float64),
        hot_edges=[(0, 0), (1, 1)],
    )
    assert free["basin_state"] == "free_descent"

    plateau = diag.update(
        energy=9.99995,
        unstable_modes=2,
        flow=np.array([1.0] * 8, dtype=np.float64),
        hot_edges=[(0, 1), (1, 0)],
    )
    assert plateau["basin_state"] == "metastable_plateau"

    trap = diag.update(
        energy=9.99994,
        unstable_modes=1,
        flow=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        hot_edges=[(0, 1)] * 8,
    )
    assert trap["basin_state"] == "localized_trap"
    assert trap["edge_reuse_rate"] > 0.5

    converged = diag.update(
        energy=9.99994,
        unstable_modes=0,
        flow=np.array([0.2, 0.2], dtype=np.float64),
        hot_edges=[(0, 0)],
    )
    assert converged["basin_state"] == "converged"


def test_basin_diagnostics_deterministic() -> None:
    cfg = BasinDiagnosticsConfig(energy_window=4)
    sequence = [
        (5.0, 3, np.array([0.4, 0.6], dtype=np.float64), [(0, 0)]),
        (4.8, 2, np.array([0.5, 0.5], dtype=np.float64), [(0, 0), (1, 1)]),
        (4.7, 2, np.array([0.7, 0.3], dtype=np.float64), [(0, 0), (1, 1)]),
    ]

    d1 = BasinDiagnostics(config=cfg)
    d2 = BasinDiagnostics(config=cfg)
    out1 = [
        d1.update(energy=e, unstable_modes=u, flow=f, hot_edges=h)
        for e, u, f, h in sequence
    ]
    out2 = [
        d2.update(energy=e, unstable_modes=u, flow=f, hot_edges=h)
        for e, u, f, h in sequence
    ]

    assert out1 == out2
