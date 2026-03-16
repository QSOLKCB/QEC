"""Tests for v12.6.0 NB instability-gradient mutation."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from src.qec.analysis.nb_instability_gradient import NBInstabilityGradientAnalyzer
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator


def _matrix() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=np.float64)


def _assert_binary(H: np.ndarray) -> None:
    assert set(np.unique(H)).issubset({0.0, 1.0})


class TestGradientAnalyzer:
    def test_deterministic_rounding_and_keys(self) -> None:
        H = _matrix()
        analyzer = NBInstabilityGradientAnalyzer()
        g1 = analyzer.compute_gradient(H)
        g2 = analyzer.compute_gradient(H)

        assert g1 == g2
        assert set(g1.keys()) == {
            "edge_scores", "node_instability", "gradient_direction",
        }
        for value in g1["edge_scores"].values():
            assert round(value, 12) == value

    def test_sparse_matches_dense(self) -> None:
        H = _matrix()
        H_sp = scipy.sparse.csr_matrix(H)
        analyzer = NBInstabilityGradientAnalyzer()

        dense = analyzer.compute_gradient(H)
        sparse = analyzer.compute_gradient(H_sp)
        assert dense == sparse


class TestGradientMutator:
    def test_flow_damping_alpha_validation(self) -> None:
        with pytest.raises(ValueError, match="flow_damping_alpha must be between 0 and 1"):
            NBGradientMutator(enabled=True, flow_damping=True, flow_damping_alpha=-0.1)
        with pytest.raises(ValueError, match="flow_damping_alpha must be between 0 and 1"):
            NBGradientMutator(enabled=True, flow_damping=True, flow_damping_alpha=1.1)

    def test_disabled_is_noop(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=False)
        H_new, log = mut.mutate(H, steps=3)
        np.testing.assert_array_equal(H_new, H)
        assert log == []

    def test_deterministic_results(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=True, avoid_4cycles=True)

        H1, log1 = mut.mutate(H, steps=4)
        H2, log2 = mut.mutate(H, steps=4)
        np.testing.assert_array_equal(H1, H2)
        assert log1 == log2

    def test_degree_preservation_and_no_duplicates(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=True, avoid_4cycles=True)
        H_new, _ = mut.mutate(H, steps=5)

        np.testing.assert_array_equal(H.sum(axis=0), H_new.sum(axis=0))
        np.testing.assert_array_equal(H.sum(axis=1), H_new.sum(axis=1))
        _assert_binary(H_new)

    def test_gradient_monotonicity_per_step(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=True, avoid_4cycles=False)
        _, log = mut.mutate(H, steps=5)

        for step in log:
            assert step["source_gradient"] > step["target_gradient"]

    def test_mutate_flow_stability(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=True, avoid_4cycles=True)
        H_flow_a, log_a = mut.mutate_flow(H, iterations=6)
        H_flow_b, log_b = mut.mutate_flow(H, iterations=6)

        np.testing.assert_array_equal(H_flow_a, H_flow_b)
        assert log_a == log_b

    def test_mutate_flow_zero_iterations_is_noop(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=True, avoid_4cycles=True)
        H_new, log = mut.mutate_flow(H, iterations=0)

        np.testing.assert_array_equal(H_new, H)
        assert log == []

    def test_mutate_flow_disabled_is_noop(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=False, avoid_4cycles=True)
        H_new, log = mut.mutate_flow(H, iterations=3)

        np.testing.assert_array_equal(H_new, H)
        assert log == []

    def test_mutate_flow_default_matches_undamped_path(self) -> None:
        H = _matrix()
        mut_default = NBGradientMutator(enabled=True, avoid_4cycles=True)
        mut_undamped = NBGradientMutator(
            enabled=True,
            avoid_4cycles=True,
            flow_damping=False,
        )

        H_default, log_default = mut_default.mutate_flow(H, iterations=5)
        H_undamped, log_undamped = mut_undamped.mutate_flow(H, iterations=5)

        np.testing.assert_array_equal(H_default, H_undamped)
        assert log_default == log_undamped

    def test_mutate_flow_damped_is_deterministic(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(
            enabled=True,
            avoid_4cycles=True,
            flow_damping=True,
        )

        H_flow_a, log_a = mut.mutate_flow(H, iterations=6)
        H_flow_b, log_b = mut.mutate_flow(H, iterations=6)

        np.testing.assert_array_equal(H_flow_a, H_flow_b)
        assert log_a == log_b

    def test_mutate_flow_damping_alpha_uses_zero_baseline_for_new_edges(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(
            enabled=True,
            avoid_4cycles=True,
            flow_damping=True,
            flow_damping_alpha=0.5,
        )

        gradients = [
            {
                "edge_scores": {(0, 0): 1.0},
                "node_instability": {0: 1.0, H.shape[0]: 0.0},
                "gradient_direction": {(0, 0): 1.0},
            },
            {
                "edge_scores": {(0, 0): 1.0},
                "node_instability": {0: 1.0, H.shape[0]: 0.0},
                "gradient_direction": {(0, 0): 1.0, (1, 1): 1.0},
            },
        ]

        call_idx = {"i": 0}

        def _gradient(_H: np.ndarray) -> dict[str, dict]:
            idx = call_idx["i"]
            call_idx["i"] += 1
            return gradients[idx]

        captured: list[dict[tuple[int, int], float]] = []

        step_count = {"i": 0}

        def _step(_H: np.ndarray, gradient: dict[str, dict]) -> dict | None:
            captured.append(dict(gradient["gradient_direction"]))
            step_count["i"] += 1
            if step_count["i"] == 1:
                return {
                    "removed_edge": (0, 0),
                    "added_edge": (0, 1),
                    "partner_removed": (1, 1),
                    "partner_added": (1, 0),
                    "source_gradient": 1.0,
                    "target_gradient": 0.0,
                }
            return None

        mut._analyzer.compute_gradient = _gradient  # type: ignore[assignment]
        mut._apply_single_gradient_step = _step  # type: ignore[assignment]

        _, log = mut.mutate_flow(H, iterations=2)
        assert len(log) == 1
        assert len(captured) == 2
        assert captured[1][(0, 0)] == 1.0
        assert captured[1][(1, 1)] == 0.5



    def test_avoid_predicted_trapping_sets_default_unchanged(self) -> None:
        H = _matrix()
        mut_default = NBGradientMutator(enabled=True, avoid_4cycles=True)
        mut_explicit = NBGradientMutator(
            enabled=True,
            avoid_4cycles=True,
            avoid_predicted_trapping_sets=False,
        )

        H_a, log_a = mut_default.mutate(H, steps=4)
        H_b, log_b = mut_explicit.mutate(H, steps=4)
        np.testing.assert_array_equal(H_a, H_b)
        assert log_a == log_b

    def test_avoid_predicted_trapping_sets_enabled_deterministic(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(
            enabled=True,
            avoid_4cycles=True,
            avoid_predicted_trapping_sets=True,
        )

        H1, log1 = mut.mutate(H, steps=4)
        H2, log2 = mut.mutate(H, steps=4)
        np.testing.assert_array_equal(H1, H2)
        assert log1 == log2

    def test_sparse_input(self) -> None:
        H = _matrix()
        H_sp = scipy.sparse.csr_matrix(H)
        mut = NBGradientMutator(enabled=True, avoid_4cycles=True)

        H_dense, log_dense = mut.mutate(H, steps=3)
        H_sparse, log_sparse = mut.mutate(H_sp, steps=3)

        np.testing.assert_array_equal(H_dense, H_sparse)
        assert log_dense == log_sparse


def test_basin_depth_energy_delta_is_finite_for_empty_variable_flow() -> None:
    H = _matrix()
    mut = NBGradientMutator(enabled=True)
    mut._trapping_predictor.predict_trapping_regions = lambda _H: {"ipr": 0.0, "risk_score": 0.0}  # type: ignore[assignment]
    mut._nb_flow.compute_flow = lambda _H: {"edge_flow": np.zeros(0, dtype=np.float64), "variable_flow": []}  # type: ignore[assignment]

    depth = mut._compute_current_basin_depth(H, prediction=None, flow_for_bias=None)
    assert np.isfinite(depth)
    assert mut._energy_deltas
    assert np.isfinite(mut._energy_deltas[-1])
