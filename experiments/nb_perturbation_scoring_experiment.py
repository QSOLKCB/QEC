"""v13.1.0 — Deterministic NB perturbation scoring sanity experiment."""

from __future__ import annotations

import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.nb_perturbation_scorer import NBPerturbationScorer


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def main() -> None:
    H = _H()
    scorer = NBPerturbationScorer()
    spectrum = scorer.compute_nb_spectrum(H)
    pred = scorer.predict_swap_delta(H, (0, 0, 1, 1), spectrum)

    print("NB Perturbation Scoring (deterministic)")
    print(f"valid_first_order={spectrum.get('valid_first_order', False)}")
    print(f"predicted_delta={pred.get('predicted_delta', 0.0):.12f}")
    print(f"pressure={pred.get('pressure', 0.0):.12f}")


if __name__ == "__main__":
    main()
