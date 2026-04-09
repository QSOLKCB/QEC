from __future__ import annotations

import pytest

from qec.analysis.neuromorphic_substrate_simulator import (
    SCHEMA_VERSION,
    build_substrate_receipt,
    compile_substrate_report,
    normalize_substrate_input,
    simulate_substrate,
    stable_substrate_report_hash,
    validate_substrate_input,
)


def _base_input() -> dict[str, object]:
    return {
        "simulation_id": "sim-001",
        "node_count": 2,
        "input_signal": [1, 2, 3, 4],
        "threshold": 5,
        "time_steps": 4,
        "decay_factor": 0.5,
        "epoch_id": "epoch-1",
        "schema_version": SCHEMA_VERSION,
    }


def test_stable_state_evolution() -> None:
    report = compile_substrate_report(_base_input())
    assert tuple((s.node_id, s.time_index, s.signal_value) for s in report.states) == (
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 2),
        (1, 1, 2),
        (0, 2, 4),
        (1, 2, 4),
        (0, 3, 0),
        (1, 3, 0),
    )


def test_threshold_crossing_correctness_and_spike_reset_behavior() -> None:
    report = compile_substrate_report(_base_input())
    crossings = tuple((s.node_id, s.time_index, s.threshold_crossed, s.signal_value) for s in report.states if s.time_index == 3)
    assert crossings == ((0, 3, True, 0), (1, 3, True, 0))


def test_stable_spike_count_and_receipt() -> None:
    report = compile_substrate_report(_base_input())
    assert report.receipt.spike_count == 2
    assert report.receipt.validation_passed is True


def test_stable_hashes_and_receipts() -> None:
    report_a = compile_substrate_report(_base_input())
    report_b = compile_substrate_report(_base_input())
    assert report_a.stable_hash == report_b.stable_hash
    assert report_a.receipt.receipt_hash == report_b.receipt.receipt_hash
    assert all(a.stable_hash == b.stable_hash for a, b in zip(report_a.states, report_b.states))


def test_repeated_run_byte_identity() -> None:
    artifacts = tuple(compile_substrate_report(_base_input()).to_canonical_bytes() for _ in range(8))
    assert len(set(artifacts)) == 1


def test_malformed_input_rejection() -> None:
    bad = _base_input()
    bad["input_signal"] = "not-a-sequence"
    with pytest.raises(ValueError, match="input_signal must be a list or tuple"):
        compile_substrate_report(bad)


def test_zero_node_rejection() -> None:
    bad = _base_input()
    bad["node_count"] = 0
    with pytest.raises(ValueError, match="node_count must be > 0"):
        compile_substrate_report(bad)


def test_schema_rejection() -> None:
    bad = _base_input()
    bad["schema_version"] = "v137.12.1"
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_substrate_report(bad)


def test_ordering_stability() -> None:
    report = compile_substrate_report(_base_input())
    assert tuple((state.time_index, state.node_id) for state in report.states) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    )


def test_validation_flag_correctness_and_hash_verifiability() -> None:
    sim_input = validate_substrate_input(normalize_substrate_input(_base_input()))
    states = simulate_substrate(sim_input)
    receipt = build_substrate_receipt(states, sim_input)
    assert receipt.validation_passed is True
    report = compile_substrate_report(_base_input())
    assert stable_substrate_report_hash(report) == report.stable_hash


def test_callable_leakage_rejected() -> None:
    bad = _base_input()
    bad["leak"] = lambda: None
    with pytest.raises(ValueError, match="callable leakage"):
        normalize_substrate_input(bad)


def test_input_signal_length_must_equal_time_steps() -> None:
    bad = _base_input()
    bad["time_steps"] = 3
    with pytest.raises(ValueError, match="input_signal length must equal time_steps"):
        compile_substrate_report(bad)


def test_decay_factor_validation() -> None:
    bad = _base_input()
    bad["decay_factor"] = 1.1
    with pytest.raises(ValueError, match=r"decay_factor must be in \[0, 1\]"):
        compile_substrate_report(bad)
