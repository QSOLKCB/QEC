"""Tests for analog runtime acceleration layer (v137.1.17)."""

from __future__ import annotations

import pytest

from qec.analysis.analog_runtime_acceleration_layer import (
    run_analog_runtime_acceleration_layer,
)


def _build_report(*, enable_fast_path: bool):
    return run_analog_runtime_acceleration_layer(
        state_id="deterministic-state",
        amplitudes=(0.25, 0.5, 0.75, 1.0),
        damping=0.2,
        coupling=0.3,
        enable_fast_path=enable_fast_path,
    )


def test_determinism_across_repeated_runs() -> None:
    first = _build_report(enable_fast_path=False)
    second = _build_report(enable_fast_path=False)
    assert first == second
    assert first.to_canonical_json() == second.to_canonical_json()


def test_canonical_json_stability() -> None:
    report = _build_report(enable_fast_path=False)
    assert report.to_canonical_json() == report.to_canonical_json()


def test_replay_equivalent_fast_path() -> None:
    baseline = _build_report(enable_fast_path=False)
    accelerated = _build_report(enable_fast_path=True)

    assert baseline.convergence == accelerated.convergence
    assert baseline.photonic_model == accelerated.photonic_model
    assert accelerated.benchmark.replay_equivalent is True
    assert accelerated.benchmark.fast_path_operation_count < accelerated.benchmark.baseline_operation_count


def test_bounded_score_ranges() -> None:
    report = _build_report(enable_fast_path=True)
    assert 0.0 <= report.convergence.convergence_score <= 1.0
    assert 0.0 <= report.convergence.residual_score <= 1.0
    assert 0.0 <= report.photonic_model.propagation_energy <= 1.0
    assert 0.0 <= report.benchmark.acceleration_score <= 1.0


def test_fail_fast_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="state_id must be a non-empty string"):
        run_analog_runtime_acceleration_layer(
            state_id="",
            amplitudes=(0.25, 0.5),
            damping=0.2,
            coupling=0.1,
            enable_fast_path=False,
        )

    with pytest.raises(ValueError, match="amplitudes must be non-empty"):
        run_analog_runtime_acceleration_layer(
            state_id="x",
            amplitudes=(),
            damping=0.2,
            coupling=0.1,
            enable_fast_path=False,
        )

    with pytest.raises(ValueError, match=r"damping must be in \[0, 1\]"):
        run_analog_runtime_acceleration_layer(
            state_id="x",
            amplitudes=(0.2,),
            damping=1.1,
            coupling=0.1,
            enable_fast_path=False,
        )
