from __future__ import annotations

import numpy as np

from qec.discovery.mutation_nb_eigenmode import NBEigenmodeMutation


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_hybrid_mode_is_optional() -> None:
    H = _H()
    base = NBEigenmodeMutation(enabled=True, use_nb_perturbation_scoring=False)
    hybrid = NBEigenmodeMutation(enabled=True, use_nb_perturbation_scoring=True)
    assert base.use_nb_perturbation_scoring is False
    assert hybrid.use_nb_perturbation_scoring is True


def test_hybrid_shortlist_clamps_and_skips_invalid_predictions() -> None:
    H = _H()
    mut = NBEigenmodeMutation(enabled=True, use_nb_perturbation_scoring=True, top_k_exact_recheck=10)

    swaps = [(0, 0, 1, 2), (0, 3, 1, 2)]
    mut._enumerate_swaps = lambda *_args: swaps  # type: ignore[assignment]
    mut._perturbation_scorer.compute_nb_spectrum = lambda _H: {  # type: ignore[assignment]
        "valid_first_order": True,
        "u": np.array([1.0], dtype=np.float64),
        "v": np.array([1.0], dtype=np.float64),
        "index": {(0, 3): 0},
        "fohpe_denominator": 1.0,
    }

    def _pred(_H: np.ndarray, swap: tuple[int, int, int, int], _spectrum: dict) -> dict | None:
        if swap == swaps[0]:
            return None
        return {"valid_first_order": True, "predicted_delta": -0.1, "pressure": 0.0}

    mut._perturbation_scorer.predict_swap_delta = _pred  # type: ignore[assignment]

    calls: list[tuple[int, int, int, int]] = []

    def _fake_analyze(Hcand: np.ndarray) -> dict:
        if np.array_equal(Hcand, H):
            return {"signature": {"spectral_radius": 1.0, "mode_ipr": 1.0, "support_fraction": 1.0, "topk_mass_fraction": 1.0}, "hot_edges": [(0, 0)]}
        calls.append((0, 3, 1, 2))
        return {"signature": {"spectral_radius": 0.9, "mode_ipr": 0.9, "support_fraction": 0.9, "topk_mass_fraction": 0.9}, "hot_edges": [(0, 0)]}

    mut._analyzer.analyze = _fake_analyze  # type: ignore[assignment]

    _, log = mut.mutate(H)
    assert len(calls) == 1
    assert len(log) == 1


def test_hybrid_deterministic_ranking_with_ties() -> None:
    H = _H()
    mut = NBEigenmodeMutation(enabled=True, use_nb_perturbation_scoring=True, top_k_exact_recheck=2)
    swaps = [(0, 0, 1, 2), (0, 3, 1, 2)]
    mut._enumerate_swaps = lambda *_args: swaps  # type: ignore[assignment]
    mut._perturbation_scorer.compute_nb_spectrum = lambda _H: {"valid_first_order": True, "u": np.array([1.0]), "v": np.array([1.0]), "index": {}, "fohpe_denominator": 1.0}  # type: ignore[assignment]
    mut._perturbation_scorer.predict_swap_delta = lambda _H, swap, _s: {"valid_first_order": True, "predicted_delta": -1.0, "pressure": 0.0}  # type: ignore[assignment]

    def _fake_analyze(Hcand: np.ndarray) -> dict:
        if np.array_equal(Hcand, H):
            return {"signature": {"spectral_radius": 1.0, "mode_ipr": 1.0, "support_fraction": 1.0, "topk_mass_fraction": 1.0}, "hot_edges": [(0, 0)]}
        return {"signature": {"spectral_radius": 0.9, "mode_ipr": 1.0, "support_fraction": 1.0, "topk_mass_fraction": 1.0}, "hot_edges": [(0, 0)]}

    mut._analyzer.analyze = _fake_analyze  # type: ignore[assignment]
    out1, log1 = mut.mutate(H)
    out2, log2 = mut.mutate(H)
    np.testing.assert_array_equal(out1, out2)
    assert log1 == log2
