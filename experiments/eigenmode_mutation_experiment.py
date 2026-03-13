"""v13.0.0 — Baseline vs NB-gradient vs NB-eigenmode mutation experiment."""

from __future__ import annotations

import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.nb_eigenmode_flow import NBEigenmodeFlowAnalyzer
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator
from src.qec.discovery.mutation_nb_eigenmode import NBEigenmodeMutation
from src.qec.discovery.objectives import compute_discovery_objectives


_FIELDS = ["spectral_radius", "mode_ipr", "support_fraction", "topk_mass_fraction"]


def _base_graph() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def run() -> dict[str, dict[str, float]]:
    H = _base_graph()
    analyzer = NBEigenmodeFlowAnalyzer()

    baseline_sig = analyzer.analyze(H)["signature"]
    baseline_obj = compute_discovery_objectives(H, seed=0)

    grad = NBGradientMutator(enabled=True)
    H_grad, _ = grad.mutate(H, steps=1)
    grad_sig = analyzer.analyze(H_grad)["signature"]
    grad_obj = compute_discovery_objectives(H_grad, seed=0)

    eig = NBEigenmodeMutation(enabled=True)
    H_eig, _ = eig.mutate(H)
    eig_sig = analyzer.analyze(H_eig)["signature"]
    eig_obj = compute_discovery_objectives(H_eig, seed=0)

    return {
        "baseline": {**baseline_sig, "failure_proxy": round(float(baseline_obj.get("instability_score", 0.0)), 12)},
        "nb_gradient": {**grad_sig, "failure_proxy": round(float(grad_obj.get("instability_score", 0.0)), 12)},
        "nb_eigenmode": {**eig_sig, "failure_proxy": round(float(eig_obj.get("instability_score", 0.0)), 12)},
    }


def main() -> None:
    result = run()
    print("Baseline vs Eigenmode Mutation (deterministic)")
    headers = ["strategy", "failure_proxy", *_FIELDS]
    print(" ".join(f"{h:>18s}" for h in headers))
    print("-" * (19 * len(headers)))
    for name in ["baseline", "nb_gradient", "nb_eigenmode"]:
        row = result[name]
        vals = [name, *[f"{row[k]:.12f}" for k in ["failure_proxy", *_FIELDS]]]
        print(f"{vals[0]:>18s} " + " ".join(f"{v:>18s}" for v in vals[1:]))


if __name__ == "__main__":
    main()
