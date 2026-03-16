from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_frustration import SpectralFrustrationAnalyzer, SpectralFrustrationResult
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator


class _StubFrustration:
    def __init__(self, precision: int = 12) -> None:
        self.precision = precision

    def compute_frustration(self, H: np.ndarray) -> SpectralFrustrationResult:
        signature = tuple(int(x) for x in np.argwhere(H > 0.5).ravel())
        score_map = {
            (0, 0, 0, 2, 1, 1, 1, 2): 2.0,
            (0, 1, 0, 2, 1, 0, 1, 2): 1.0,
            (0, 0, 0, 1, 1, 1, 1, 2): 3.0,
        }
        score = float(score_map.get(signature, 2.5))
        return SpectralFrustrationResult(
            frustration_score=round(score, self.precision),
            negative_modes=1,
            max_ipr=round(0.25 + 0.01 * score, self.precision),
            transport_imbalance=round(0.1 * score, self.precision),
            trap_modes=(np.array([1.0, 0.0, 0.0], dtype=np.float64),),
        )



def _make_gradient() -> dict[str, object]:
    return {
        "edge_scores": {(0, 0): 2.0, (1, 1): 2.0},
        "node_instability": {0: 5.0, 1: 5.0, 2: 0.0, 3: 0.0, 4: 0.0},
        "gradient_direction": {(0, 0): 7.0, (1, 1): 7.0},
    }


def test_frustration_guided_scoring_prefers_reduction() -> None:
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
    mut = NBGradientMutator(
        enabled=True,
        avoid_4cycles=False,
        frustration_guided=True,
        eta_frustration=1.0,
        frustration_eval_limit=2,
    )
    mut._frustration = _StubFrustration(precision=mut.precision)
    mut._find_partner_check = lambda ci, vi, vj, *_: 1 if ci == 0 else 0  # type: ignore[assignment]

    candidates = mut._enumerate_swap_candidates(H, _make_gradient())
    mut._apply_frustration_guidance(H, candidates)

    best = min(candidates, key=lambda c: (float(c["score"]), int(c["swap_index"])))
    assert best["delta_frustration"] < 0.0
    assert best["frustration_after"] < best["frustration_before"]


def test_deterministic_swap_sequence_with_frustration_guidance() -> None:
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
    mut = NBGradientMutator(
        enabled=True,
        avoid_4cycles=False,
        frustration_guided=True,
        eta_frustration=0.5,
        frustration_eval_limit=2,
        track_trap_modes=True,
    )
    mut._analyzer.compute_gradient = lambda _H: _make_gradient()  # type: ignore[assignment]
    mut._frustration = _StubFrustration(precision=mut.precision)
    mut._find_partner_check = lambda ci, vi, vj, *_: 1 if ci == 0 else 0  # type: ignore[assignment]

    H1, log1 = mut.mutate(H, steps=1)
    H2, log2 = mut.mutate(H, steps=1)

    np.testing.assert_array_equal(H1, H2)
    assert log1 == log2
    assert len(mut._trap_modes) == 1


def test_stable_frustration_score_across_repeated_calls() -> None:
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
    analyzer = SpectralFrustrationAnalyzer()

    r1 = analyzer.compute_frustration(H)
    r2 = analyzer.compute_frustration(H)
    assert r1.frustration_score == r2.frustration_score
    assert r1.negative_modes == r2.negative_modes
    assert r1.max_ipr == r2.max_ipr


def test_trap_node_detection_from_localized_mode() -> None:
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
    analyzer = SpectralFrustrationAnalyzer()
    modes = analyzer.extract_trap_modes(H)

    assert len(modes) >= 1
    mode = np.abs(modes[0][: H.shape[1]])
    top = np.argsort(-mode, kind="mergesort")[:2]
    assert tuple(int(i) for i in sorted(top.tolist())) == (0, 2)


def test_entropy_bonus_is_opt_in_and_deterministic() -> None:
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
    gradient = _make_gradient()

    base = NBGradientMutator(enabled=True, avoid_4cycles=False)
    base._find_partner_check = lambda ci, vi, vj, *_: 1 if ci == 0 else 0  # type: ignore[assignment]
    c0 = base._enumerate_swap_candidates(H, gradient)

    ent = NBGradientMutator(
        enabled=True,
        avoid_4cycles=False,
        enable_spectral_entropy=True,
        eta_entropy=0.3,
    )
    ent._find_partner_check = lambda ci, vi, vj, *_: 1 if ci == 0 else 0  # type: ignore[assignment]
    c1 = ent._enumerate_swap_candidates(H, gradient)
    ent._apply_entropy_guidance(H, c1)
    c2 = ent._enumerate_swap_candidates(H, gradient)
    ent._apply_entropy_guidance(H, c2)

    assert len(c1) == len(c2)
    for left, right in zip(c1, c2):
        assert left.keys() == right.keys()
        assert np.isfinite(float(left["score"]))
        assert np.isfinite(float(right["score"]))
    for cand in c0:
        assert "delta_entropy" not in cand
    for cand in c1:
        assert "entropy_temperature" in cand
        assert "delta_entropy" in cand
