"""v13.0.0 — Eigenmode signature robustness experiment (deterministic)."""

from __future__ import annotations

import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.analysis.nb_eigenmode_flow import NBEigenmodeFlowAnalyzer
from qec.discovery.swap_helpers import deterministic_two_edge_swap


_FIELDS = [
    "spectral_radius",
    "mode_ipr",
    "support_fraction",
    "topk_mass_fraction",
]


def _graphs() -> list[np.ndarray]:
    return [
        np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], dtype=np.float64),
        np.array([
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1],
        ], dtype=np.float64),
        np.array([
            [1, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 1],
        ], dtype=np.float64),
    ]


def _bounded_deterministic_perturbation(H: np.ndarray) -> np.ndarray:
    """Apply one lexicographically first valid degree-preserving 2-edge swap."""
    return deterministic_two_edge_swap(H)


def run() -> dict[str, dict[str, float]]:
    analyzer = NBEigenmodeFlowAnalyzer()

    residuals: dict[str, list[float]] = {k: [] for k in _FIELDS}

    for H in _graphs():
        base = analyzer.analyze(H)["signature"]
        H_mut = _bounded_deterministic_perturbation(H)
        pert = analyzer.analyze(H_mut)["signature"]
        for field in _FIELDS:
            residuals[field].append(abs(pert[field] - base[field]))

    summary: dict[str, dict[str, float]] = {}
    for field in _FIELDS:
        values = np.asarray(residuals[field], dtype=np.float64)
        summary[field] = {
            "mean_abs_residual": round(float(values.mean()) if len(values) else 0.0, 12),
            "max_residual": round(float(values.max()) if len(values) else 0.0, 12),
        }
    return summary


def _bar(value: float, scale: int = 30) -> str:
    n = max(0, min(scale, int(round(value * scale))))
    return "#" * n


def main() -> None:
    summary = run()
    print("Eigenmode Signature Robustness (deterministic perturbations)")
    print("field                 mean_abs_residual   max_residual   bar")
    print("-" * 72)
    for field in _FIELDS:
        mean_v = summary[field]["mean_abs_residual"]
        max_v = summary[field]["max_residual"]
        print(f"{field:20s} {mean_v:18.12f} {max_v:14.12f}   {_bar(mean_v)}")

    stable_field = min(_FIELDS, key=lambda f: (summary[f]["mean_abs_residual"], f))
    unstable_field = max(_FIELDS, key=lambda f: (summary[f]["mean_abs_residual"], f))
    print(
        f"Interpretation: most stable={stable_field}, most sensitive={unstable_field}."
    )


if __name__ == "__main__":
    main()
