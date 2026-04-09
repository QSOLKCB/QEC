from __future__ import annotations

import pytest

from qec.benchmark.bio_signal_benchmark_battery import (
    BioSignalBenchmarkConfig,
    SCHEMA_VERSION,
    run_bio_signal_benchmark_battery,
    run_bio_signal_benchmark_case,
    run_replay_fidelity_benchmark,
    run_scaling_benchmark,
    run_threshold_sweep_benchmark,
)
from qec.analysis.hybrid_signal_interface import build_hybrid_signal_trace
from qec.analysis.neuromorphic_substrate_simulator import SubstrateInput, compile_substrate_report


def _base_input() -> SubstrateInput:
    return SubstrateInput(
        simulation_id="bio-bench-test",
        node_count=3,
        input_signal=(1.0, 2.0, 3.0, 4.0),
        threshold=5,
        time_steps=4,
        decay_factor=0.5,
        epoch_id="epoch-test",
        schema_version="v137.12.0",
    )


def test_deterministic_replay_identity() -> None:
    result_a = run_replay_fidelity_benchmark(_base_input())
    result_b = run_replay_fidelity_benchmark(_base_input())
    assert result_a.stable_hash == result_b.stable_hash
    assert result_a.metrics["replay_fidelity_score"] == 1.0


def test_canonical_export_stability() -> None:
    result = run_bio_signal_benchmark_case(_base_input(), case_id="case-export")
    assert result.to_canonical_json() == result.to_canonical_json()
    assert result.to_canonical_bytes() == result.to_canonical_bytes()


def test_threshold_sweep_determinism() -> None:
    trace = build_hybrid_signal_trace(compile_substrate_report(_base_input()))
    result_a = run_threshold_sweep_benchmark(trace, (0.25, 0.5, 0.75, 1.0))
    result_b = run_threshold_sweep_benchmark(trace, (0.25, 0.5, 0.75, 1.0))
    assert result_a.to_canonical_bytes() == result_b.to_canonical_bytes()


def test_scaling_sweep_determinism() -> None:
    config = BioSignalBenchmarkConfig(
        scaling_node_counts=(2, 3),
        scaling_time_steps=(4,),
        scaling_frame_counts=(4,),
    )
    result_a = run_scaling_benchmark(config)
    result_b = run_scaling_benchmark(config)
    assert result_a.stable_hash == result_b.stable_hash


def test_malformed_input_rejection() -> None:
    with pytest.raises(ValueError, match="threshold entry must be > 0"):
        run_threshold_sweep_benchmark(
            build_hybrid_signal_trace(compile_substrate_report(_base_input())),
            (0.0, 0.5),
        )


def test_bounded_metric_validation() -> None:
    report = run_bio_signal_benchmark_battery()
    for result in report.results:
        for value in result.metrics.values():
            assert 0.0 <= value <= 1.0
    for value in report.aggregate_metrics.values():
        assert 0.0 <= value <= 1.0


def test_stable_ordering_benchmark() -> None:
    result = run_bio_signal_benchmark_case(_base_input(), case_id="ordering-check")
    assert result.metrics["ordering_integrity_score"] == 1.0


def test_wrapper_manual_equivalence() -> None:
    config = BioSignalBenchmarkConfig(
        perturbation_profiles=("shifted_threshold",),
        scaling_node_counts=(2,),
        scaling_time_steps=(4,),
        scaling_frame_counts=(4,),
    )
    wrapped = run_bio_signal_benchmark_battery(config)

    base_input = SubstrateInput(
        simulation_id=f"{config.simulation_id}-base",
        node_count=config.node_count,
        input_signal=(1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0),
        threshold=config.threshold,
        time_steps=config.time_steps,
        decay_factor=config.decay_factor,
        epoch_id=config.epoch_id,
        schema_version="v137.12.0",
    )
    manual_first = run_bio_signal_benchmark_case(base_input, case_id="signal-stability", category="signal_stability")

    assert wrapped.results[0].to_canonical_bytes() == manual_first.to_canonical_bytes()


def test_report_hash_stability() -> None:
    report_a = run_bio_signal_benchmark_battery(BioSignalBenchmarkConfig())
    report_b = run_bio_signal_benchmark_battery(BioSignalBenchmarkConfig())
    assert report_a.stable_hash == report_b.stable_hash
    assert report_a.schema_version == SCHEMA_VERSION


def test_repeated_run_byte_identity() -> None:
    artifacts = tuple(run_bio_signal_benchmark_battery().to_canonical_bytes() for _ in range(5))
    assert len(set(artifacts)) == 1
