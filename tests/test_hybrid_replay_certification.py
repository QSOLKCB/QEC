from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.hybrid_replay_certification import (
    HybridReplayCertificationConfig,
    build_hybrid_replay_certificate,
    run_hybrid_replay_certification,
)
from qec.analysis.hybrid_signal_interface import run_hybrid_signal_interface
from qec.analysis.neuromorphic_substrate_simulator import SubstrateInput, compile_substrate_report
from qec.benchmark.bio_signal_benchmark_battery import BioSignalBenchmarkConfig, run_bio_signal_benchmark_battery


def _build_stack():
    bench_config = BioSignalBenchmarkConfig()
    sim_input = SubstrateInput(
        simulation_id=f"{bench_config.simulation_id}-base",
        node_count=bench_config.node_count,
        input_signal=tuple(float((idx % 5) + 1) for idx in range(bench_config.time_steps)),
        threshold=bench_config.threshold,
        time_steps=bench_config.time_steps,
        decay_factor=bench_config.decay_factor,
        epoch_id=bench_config.epoch_id,
        schema_version="v137.12.0",
    )
    substrate_report = compile_substrate_report(sim_input)
    trace, interface_receipt = run_hybrid_signal_interface(substrate_report)
    benchmark_report = run_bio_signal_benchmark_battery(bench_config)
    assert any(result.case.trace_hash == trace.stable_hash for result in benchmark_report.results)
    return substrate_report, trace, interface_receipt, benchmark_report


def test_same_input_identical_certification() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()

    cert_a = run_hybrid_replay_certification(substrate_report, trace, interface_receipt, benchmark_report)
    cert_b = run_hybrid_replay_certification(substrate_report, trace, interface_receipt, benchmark_report)

    assert cert_a == cert_b
    assert cert_a.to_canonical_bytes() == cert_b.to_canonical_bytes()


def test_repeated_run_byte_identity() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()

    payloads = tuple(
        run_hybrid_replay_certification(substrate_report, trace, interface_receipt, benchmark_report).to_canonical_bytes()
        for _ in range(8)
    )
    assert len(set(payloads)) == 1


def test_broken_lineage_rejection() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()
    broken_trace = replace(trace, input_stable_hash="f" * 64)

    with pytest.raises(ValueError, match="broken lineage"):
        run_hybrid_replay_certification(substrate_report, broken_trace, interface_receipt, benchmark_report)


def test_hash_mismatch_failure() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()
    bad_result = replace(benchmark_report.results[0], stable_hash="0" * 64)
    broken_report = replace(benchmark_report, results=(bad_result,) + benchmark_report.results[1:])

    with pytest.raises(ValueError, match="invalid hashes"):
        run_hybrid_replay_certification(substrate_report, trace, interface_receipt, broken_report)


def test_structural_mismatch_failure() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()
    bad_trace = replace(trace, node_ids=(1, 0, 2, 3))

    with pytest.raises(ValueError, match="invalid ordering"):
        run_hybrid_replay_certification(substrate_report, bad_trace, interface_receipt, benchmark_report)


def test_metric_mismatch_failure() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()
    mutated_metrics = dict(benchmark_report.aggregate_metrics)
    mutated_metrics["threshold_response_score"] = float(mutated_metrics["threshold_response_score"]) - 0.1
    replay_report = replace(benchmark_report, aggregate_metrics=mutated_metrics)

    cert = run_hybrid_replay_certification(
        substrate_report,
        trace,
        interface_receipt,
        benchmark_report,
        replay_benchmark_report=replay_report,
    )
    assert cert.result.metric_replay_passed is False
    assert cert.result.certification_score == 0.0


def test_bounded_score_validation() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()
    cert = run_hybrid_replay_certification(substrate_report, trace, interface_receipt, benchmark_report)
    assert 0.0 <= cert.result.certification_score <= 1.0


def test_canonical_bytes_stability() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()
    cert = run_hybrid_replay_certification(substrate_report, trace, interface_receipt, benchmark_report)
    assert cert.to_canonical_bytes() == cert.to_canonical_bytes()


def test_stable_hash_stability() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()
    cert = run_hybrid_replay_certification(substrate_report, trace, interface_receipt, benchmark_report)
    assert cert.stable_hash() == cert.report_hash


def test_wrapper_manual_equivalence() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()
    wrapped = run_hybrid_replay_certification(substrate_report, trace, interface_receipt, benchmark_report)

    manual = build_hybrid_replay_certificate(
        config=wrapped.config,
        evidence=wrapped.evidence,
        result=wrapped.result,
    )
    assert wrapped.to_canonical_bytes() == manual.to_canonical_bytes()


def test_schema_mismatch_rejection() -> None:
    substrate_report, trace, interface_receipt, benchmark_report = _build_stack()
    bad_config = HybridReplayCertificationConfig(schema_version="v0")

    with pytest.raises(ValueError, match="schema mismatch"):
        run_hybrid_replay_certification(substrate_report, trace, interface_receipt, benchmark_report, config=bad_config)
