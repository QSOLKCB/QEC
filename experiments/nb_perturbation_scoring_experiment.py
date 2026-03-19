"""v13.1.0 — Deterministic NB perturbation scoring experiment."""

from __future__ import annotations

import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.analysis.api import NBEigenmodeFlowAnalyzer, NBPerturbationScorer, compute_nb_spectrum, enumerate_candidate_swaps
from qec.discovery.mutation_nb_eigenmode import NBEigenmodeMutation


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def _exact_total_delta(
    H: np.ndarray,
    swap: tuple[int, int, int, int],
    analyzer: NBEigenmodeFlowAnalyzer,
    baseline_signature: dict[str, float],
    precision: int,
) -> float:
    ci, vi, cj, vj = swap
    H_candidate = H.copy()
    H_candidate[ci, vi] = 0.0
    H_candidate[cj, vj] = 0.0
    H_candidate[ci, vj] = 1.0
    H_candidate[cj, vi] = 1.0
    cand_sig = analyzer.analyze(H_candidate)["signature"]
    dr = round(cand_sig["spectral_radius"] - baseline_signature["spectral_radius"], precision)
    di = round(cand_sig["mode_ipr"] - baseline_signature["mode_ipr"], precision)
    ds = round(cand_sig["support_fraction"] - baseline_signature["support_fraction"], precision)
    dt = round(cand_sig["topk_mass_fraction"] - baseline_signature["topk_mass_fraction"], precision)
    return round(dr + di + ds + dt, precision)


def main() -> None:
    H = _H()
    mutator = NBEigenmodeMutation(enabled=True)
    baseline = mutator._analyzer.analyze(H)
    swaps = enumerate_candidate_swaps(H, baseline.get("hot_edges", []))

    scorer = NBPerturbationScorer()
    spectrum = compute_nb_spectrum(H)

    predicted_rank: list[tuple[tuple[float, int, int, int, int], tuple[int, int, int, int]]] = []
    exact_rank: list[tuple[tuple[float, int, int, int, int], tuple[int, int, int, int]]] = []

    for swap in swaps:
        p = scorer.predict_swap_delta(H, swap, spectrum)
        if p is None or not p["valid_first_order"]:
            continue

        ci, vi, cj, vj = swap
        pred_delta = float(p["predicted_delta"])
        predicted_rank.append(((pred_delta, ci, vi, cj, vj), swap))

        exact_total = _exact_total_delta(H, swap, mutator._analyzer, baseline["signature"], mutator.precision)
        exact_rank.append(((exact_total, ci, vi, cj, vj), swap))

    predicted_rank.sort(key=lambda item: item[0])
    exact_rank.sort(key=lambda item: item[0])

    print("NB Perturbation Scoring (deterministic)")
    print(f"valid_first_order={spectrum.get('valid_first_order', False)}")
    print(f"num_swaps={len(swaps)}")
    print(f"num_predicted={len(predicted_rank)}")
    print(f"predicted_top3={[swap for _, swap in predicted_rank[:3]]}")
    print(f"exact_top3={[swap for _, swap in exact_rank[:3]]}")


if __name__ == "__main__":
    main()
