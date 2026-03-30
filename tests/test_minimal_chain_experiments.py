"""Deterministic tests for v108.1.0 minimal chain experiments."""

from __future__ import annotations

from qec.analysis.minimal_chain_experiments import run_minimal_chain_experiment


REQUIRED_KEYS = {
    "chain_length",
    "perturbation_index",
    "is_boundary_perturbation",
    "initial_chain",
    "final_chain",
    "endpoint_signal_strength",
    "interior_signal_strength",
    "signal_asymmetry",
    "coherence_response",
    "parity_response",
    "protection_hint_score",
}


def test_boundary_perturbation() -> None:
    out = run_minimal_chain_experiment(chain_length=7, perturbation_index=0, perturbation_magnitude=1.0)

    assert set(out.keys()) == REQUIRED_KEYS
    assert out["is_boundary_perturbation"] is True
    assert out["chain_length"] == 7
    assert out["perturbation_index"] == 0
    assert out["endpoint_signal_strength"] >= out["interior_signal_strength"]


def test_center_perturbation() -> None:
    out = run_minimal_chain_experiment(chain_length=7, perturbation_index=3, perturbation_magnitude=1.0)

    assert out["is_boundary_perturbation"] is False
    assert out["chain_length"] == 7
    assert out["perturbation_index"] == 3
    assert out["interior_signal_strength"] >= out["endpoint_signal_strength"]


def test_boundary_and_center_produce_distinct_asymmetry() -> None:
    boundary = run_minimal_chain_experiment(chain_length=7, perturbation_index=0, perturbation_magnitude=1.0)
    center = run_minimal_chain_experiment(chain_length=7, perturbation_index=3, perturbation_magnitude=1.0)

    assert boundary["signal_asymmetry"] != center["signal_asymmetry"]


def test_short_chain_length_three_edge_case() -> None:
    out = run_minimal_chain_experiment(chain_length=3, perturbation_index=1, perturbation_magnitude=1.0)

    assert out["chain_length"] == 3
    assert len(out["initial_chain"]) == 3
    assert len(out["final_chain"]) == 3


def test_bounded_outputs() -> None:
    out = run_minimal_chain_experiment(chain_length=5, perturbation_index=0, perturbation_magnitude=1.0)

    for key in ("endpoint_signal_strength", "interior_signal_strength", "signal_asymmetry", "parity_response", "protection_hint_score"):
        assert 0.0 <= out[key] <= 1.0


def test_exact_determinism() -> None:
    r1 = run_minimal_chain_experiment(chain_length=9, perturbation_index=4, perturbation_magnitude=1.0)
    r2 = run_minimal_chain_experiment(chain_length=9, perturbation_index=4, perturbation_magnitude=1.0)
    assert r1 == r2
