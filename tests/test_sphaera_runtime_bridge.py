"""Tests for v143.4 Sphaera integration bridge.

Attribution:
This module incorporates concepts from:
Marc Brendecke (2026)
Quantum Sphaera Companion v3.30.0
DOI: https://doi.org/10.5281/zenodo.19682951
License: CC-BY-4.0
"""

from __future__ import annotations

import pytest

from qec.analysis.iterative_system_abstraction_layer import IterativeExecutionReceipt, IterativeStateSnapshot, evaluate_iterative_system_abstraction
from qec.analysis.self_determination_kernel import SelfDeterminationReceipt
from qec.analysis.spectral_structure_kernel import SpectralStructureReceipt
from qec.analysis.sphaera_runtime_bridge import evaluate_sphaera_runtime_bridge


def _snapshot(step_index: int, state_id: str, convergence_metric: float) -> IterativeStateSnapshot:
    return IterativeStateSnapshot(
        step_index=step_index,
        state_id=state_id,
        state_payload={"id": state_id, "step": step_index},
        convergence_metric=convergence_metric,
        active=True,
    )


def _execution_state() -> IterativeExecutionReceipt:
    return evaluate_iterative_system_abstraction(
        (
            _snapshot(0, "A", 0.10),
            _snapshot(1, "B", 0.20),
            _snapshot(2, "A", 0.40),
            _snapshot(3, "B", 0.75),
            _snapshot(4, "A", 0.97),
        )
    )


def _full_pipeline(execution_state: IterativeExecutionReceipt):
    computed = evaluate_sphaera_runtime_bridge(execution_state)
    return (
        computed,
        computed.invariant_receipt,
        computed.geometry_receipt,
        computed.ensemble_receipt,
        computed.spectral_receipt,
        computed.self_determination_receipt,
    )


def test_full_pipeline_deterministic_replay() -> None:
    execution_state = _execution_state()

    first = evaluate_sphaera_runtime_bridge(execution_state)
    second = evaluate_sphaera_runtime_bridge(execution_state)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_precomputed_receipts_match_computed_output() -> None:
    execution_state = _execution_state()
    computed, invariant, geometry, ensemble, spectral, self_det = _full_pipeline(execution_state)

    provided = evaluate_sphaera_runtime_bridge(
        execution_state,
        invariant_receipt=invariant,
        geometry_receipt=geometry,
        ensemble_receipt=ensemble,
        spectral_receipt=spectral,
        self_determination_receipt=self_det,
    )

    assert provided.to_canonical_json() == computed.to_canonical_json()
    assert provided.stable_hash == computed.stable_hash


def test_lineage_mismatch_rejected() -> None:
    execution_state = _execution_state()
    _, invariant, geometry, ensemble, spectral, self_det = _full_pipeline(execution_state)

    mismatch_invariant = evaluate_sphaera_runtime_bridge(
        evaluate_iterative_system_abstraction(
            (
                _snapshot(0, "X", 0.1),
                _snapshot(1, "X", 0.2),
                _snapshot(2, "X", 0.5),
            )
        )
    ).invariant_receipt

    with pytest.raises(ValueError, match="lineage mismatch"):
        evaluate_sphaera_runtime_bridge(
            execution_state,
            invariant_receipt=mismatch_invariant,
            geometry_receipt=geometry,
            ensemble_receipt=ensemble,
            spectral_receipt=spectral,
            self_determination_receipt=self_det,
        )


def test_stable_hash_depends_on_upstream_receipts() -> None:
    execution_state = _execution_state()
    base = evaluate_sphaera_runtime_bridge(execution_state)

    varied_state = evaluate_iterative_system_abstraction(
        (
            _snapshot(0, "A", 0.10),
            _snapshot(1, "A", 0.30),
            _snapshot(2, "A", 0.60),
            _snapshot(3, "A", 0.95),
        )
    )
    varied = evaluate_sphaera_runtime_bridge(varied_state)

    assert base.invariant_hash != varied.invariant_hash
    assert base.stable_hash != varied.stable_hash


def test_global_state_label_mapping_correctness() -> None:
    execution_state = _execution_state()
    base, invariant, geometry, ensemble, spectral, self_det = _full_pipeline(execution_state)

    terminal_self = SelfDeterminationReceipt(
        allowed_transitions=self_det.allowed_transitions,
        selected_transition_id="no_admissible_transition",
        selected_transition_score=0.0,
        selection_confidence=0.0,
        posture_label=self_det.posture_label,
        transition_count=self_det.transition_count,
        admissible_count=0,
        spectral_receipt_stable_hash=spectral.stable_hash,
        ensemble_receipt_stable_hash=ensemble.stable_hash,
        geometry_receipt_stable_hash=geometry.stable_hash,
        invariant_receipt_stable_hash=invariant.stable_hash,
        version=self_det.version,
        control_mode=self_det.control_mode,
        observatory_only=True,
    )
    terminal = evaluate_sphaera_runtime_bridge(
        execution_state,
        invariant_receipt=invariant,
        geometry_receipt=geometry,
        ensemble_receipt=ensemble,
        spectral_receipt=spectral,
        self_determination_receipt=terminal_self,
    )
    assert terminal.global_state_label == "terminal_state"

    for dynamics_label, expected in (
        ("rigid", "stable_equilibrium"),
        ("structured", "structured_equilibrium"),
        ("coupled", "adaptive_state"),
        ("highly_coupled", "dynamic_state"),
    ):
        alt_spectral = SpectralStructureReceipt(
            ensemble_operator_shape=spectral.ensemble_operator_shape,
            trace_value=spectral.trace_value,
            frobenius_norm=spectral.frobenius_norm,
            diagonal_energy=spectral.diagonal_energy,
            off_diagonal_energy=spectral.off_diagonal_energy,
            spectral_radius_proxy=spectral.spectral_radius_proxy,
            spectral_gap_proxy=spectral.spectral_gap_proxy,
            spectral_dispersion_score=spectral.spectral_dispersion_score,
            coupling_density_score=spectral.coupling_density_score,
            diagonal_dominance_score=spectral.diagonal_dominance_score,
            ensemble_symmetry_score=spectral.ensemble_symmetry_score,
            dynamics_label=dynamics_label,
            dynamics_rank={"rigid": 0, "structured": 1, "coupled": 2, "highly_coupled": 3}[dynamics_label],
            geometry_receipt_stable_hash=geometry.stable_hash,
            ensemble_receipt_stable_hash=ensemble.stable_hash,
            invariant_receipt_stable_hash=invariant.stable_hash,
            version=spectral.version,
            control_mode=spectral.control_mode,
            observatory_only=True,
        )
        alt_self = SelfDeterminationReceipt(
            allowed_transitions=self_det.allowed_transitions,
            selected_transition_id=self_det.selected_transition_id,
            selected_transition_score=self_det.selected_transition_score,
            selection_confidence=self_det.selection_confidence,
            posture_label=self_det.posture_label,
            transition_count=self_det.transition_count,
            admissible_count=self_det.admissible_count,
            spectral_receipt_stable_hash=alt_spectral.stable_hash,
            ensemble_receipt_stable_hash=ensemble.stable_hash,
            geometry_receipt_stable_hash=geometry.stable_hash,
            invariant_receipt_stable_hash=invariant.stable_hash,
            version=self_det.version,
            control_mode=self_det.control_mode,
            observatory_only=True,
        )
        out = evaluate_sphaera_runtime_bridge(
            execution_state,
            invariant_receipt=invariant,
            geometry_receipt=geometry,
            ensemble_receipt=ensemble,
            spectral_receipt=alt_spectral,
            self_determination_receipt=alt_self,
        )
        assert out.global_state_label == expected

    assert base.global_state_label in {
        "stable_equilibrium",
        "structured_equilibrium",
        "adaptive_state",
        "dynamic_state",
    }


def test_coherence_score_bounds_and_formula() -> None:
    execution_state = _execution_state()
    receipt = evaluate_sphaera_runtime_bridge(execution_state)

    expected = round(
        max(
            0.0,
            min(
                1.0,
                0.4 * receipt.ensemble_receipt.global_consistency_score
                + 0.3 * (1.0 - receipt.spectral_receipt.spectral_dispersion_score)
                + 0.3 * receipt.self_determination_receipt.selection_confidence,
            ),
        ),
        12,
    )

    assert 0.0 <= receipt.coherence_score <= 1.0
    assert receipt.coherence_score == expected


def test_permutation_invariance_for_precomputed_inputs() -> None:
    execution_state = _execution_state()
    computed, invariant, geometry, ensemble, spectral, self_det = _full_pipeline(execution_state)

    from_invariant_only = evaluate_sphaera_runtime_bridge(
        execution_state,
        invariant_receipt=invariant,
    )
    from_through_spectral = evaluate_sphaera_runtime_bridge(
        execution_state,
        invariant_receipt=invariant,
        geometry_receipt=geometry,
        ensemble_receipt=ensemble,
        spectral_receipt=spectral,
    )
    from_all = evaluate_sphaera_runtime_bridge(
        execution_state,
        invariant_receipt=invariant,
        geometry_receipt=geometry,
        ensemble_receipt=ensemble,
        spectral_receipt=spectral,
        self_determination_receipt=self_det,
    )

    assert computed.to_canonical_json() == from_invariant_only.to_canonical_json()
    assert computed.to_canonical_json() == from_through_spectral.to_canonical_json()
    assert computed.to_canonical_json() == from_all.to_canonical_json()
