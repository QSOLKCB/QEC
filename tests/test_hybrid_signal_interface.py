from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.hybrid_signal_interface import (
    HybridSignalInterfaceConfig,
    SCHEMA_VERSION,
    build_hybrid_signal_receipt,
    build_hybrid_signal_trace,
    compute_hybrid_activity_summary,
    project_substrate_events_to_frames,
    run_hybrid_signal_interface,
)
from qec.analysis.neuromorphic_substrate_simulator import compile_substrate_report


def _base_input() -> dict[str, object]:
    return {
        "simulation_id": "sim-001",
        "node_count": 2,
        "input_signal": [1, 2, 3, 4],
        "threshold": 5,
        "time_steps": 4,
        "decay_factor": 0.5,
        "epoch_id": "epoch-1",
        "schema_version": "v137.12.0",
    }


def test_frozen_dataclass_behavior() -> None:
    config = HybridSignalInterfaceConfig()
    with pytest.raises(FrozenInstanceError):
        config.interface_version = "v0"


def test_canonical_export_stability() -> None:
    report = compile_substrate_report(_base_input())
    trace = build_hybrid_signal_trace(report)
    assert trace.to_canonical_json() == trace.to_canonical_json()
    assert trace.to_canonical_bytes() == trace.to_canonical_bytes()


def test_same_input_same_bytes() -> None:
    report = compile_substrate_report(_base_input())
    artifacts = tuple(build_hybrid_signal_trace(report).to_canonical_bytes() for _ in range(4))
    assert len(set(artifacts)) == 1


def test_stable_sha256_identity_repeated_runs() -> None:
    report = compile_substrate_report(_base_input())
    trace_a = build_hybrid_signal_trace(report)
    trace_b = build_hybrid_signal_trace(report)
    receipt_a = build_hybrid_signal_receipt(trace_a)
    receipt_b = build_hybrid_signal_receipt(trace_b)
    assert trace_a.stable_hash == trace_b.stable_hash
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_deterministic_event_frame_ordering() -> None:
    report = compile_substrate_report(_base_input())
    frames = project_substrate_events_to_frames(report, HybridSignalInterfaceConfig())
    assert tuple(frame.time_index for frame in frames) == (0, 1, 2, 3)
    assert tuple((frame.spike_event_lane[0], frame.spike_event_lane[1]) for frame in frames) == (
        (0, 0),
        (0, 0),
        (0, 0),
        (1, 1),
    )


def test_bounded_normalization_guarantees() -> None:
    report = compile_substrate_report(_base_input())
    trace = build_hybrid_signal_trace(report)
    for frame in trace.frames:
        assert all(0.0 <= value <= 1.0 for value in frame.node_state_lane)
        assert 0.0 <= frame.aggregate_activity_lane <= 1.0


def test_fail_fast_validation_on_malformed_input() -> None:
    bad = _base_input()
    bad["input_signal"] = [1, -3, 2, 0]
    report = compile_substrate_report(bad)
    with pytest.raises(ValueError, match=r"signal_value must be >= 0"):
        build_hybrid_signal_trace(report)


def test_interface_receipt_integrity() -> None:
    report = compile_substrate_report(_base_input())
    trace = build_hybrid_signal_trace(report)
    receipt = build_hybrid_signal_receipt(trace)
    assert receipt.interface_version == SCHEMA_VERSION
    assert receipt.schema_version == SCHEMA_VERSION
    assert receipt.output_stable_hash == trace.stable_hash
    assert receipt.input_stable_hash == report.stable_hash
    assert receipt.validation_passed is True


def test_aggregate_activity_summary_deterministic() -> None:
    report = compile_substrate_report(_base_input())
    trace = build_hybrid_signal_trace(report)
    summary_a = compute_hybrid_activity_summary(trace)
    summary_b = compute_hybrid_activity_summary(trace)
    assert summary_a == summary_b


def test_wrapper_output_matches_manual_pipeline() -> None:
    report = compile_substrate_report(_base_input())
    manual_trace = build_hybrid_signal_trace(report)
    manual_summary = compute_hybrid_activity_summary(manual_trace)
    manual_receipt = build_hybrid_signal_receipt(manual_trace, summary_metrics=manual_summary)

    wrapped_trace, wrapped_receipt = run_hybrid_signal_interface(report)

    assert wrapped_trace.to_canonical_bytes() == manual_trace.to_canonical_bytes()
    assert wrapped_receipt.to_canonical_bytes() == manual_receipt.to_canonical_bytes()


def test_strict_ordering_rejects_unordered_states() -> None:
    report = compile_substrate_report(_base_input())
    shuffled = tuple(sorted(report.states, key=lambda s: (s.node_id, s.time_index)))
    broken = type(report)(
        input=report.input,
        states=shuffled,
        receipt=report.receipt,
        stable_hash=report.stable_hash,
        schema_version=report.schema_version,
    )
    with pytest.raises(ValueError, match="states must be ordered"):
        build_hybrid_signal_trace(broken)
