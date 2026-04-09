"""Tests for v137.12.4 — Experimental Research Pack."""

from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.experimental_research_pack import (
    ExperimentalResearchPackConfig,
    build_artifact_manifest,
    build_experimental_research_pack,
    build_research_receipt,
    export_research_pack_json,
    run_experimental_research_pack,
)
from qec.analysis.hybrid_replay_certification import (
    run_hybrid_replay_certification,
)
from qec.analysis.hybrid_signal_interface import run_hybrid_signal_interface
from qec.analysis.neuromorphic_substrate_simulator import (
    SubstrateInput,
    compile_substrate_report,
)
from qec.benchmark.bio_signal_benchmark_battery import (
    BioSignalBenchmarkConfig,
    run_bio_signal_benchmark_battery,
)


def _build_full_stack():
    """Build the complete v137.12.x artifact stack for testing."""
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
    certification_report = run_hybrid_replay_certification(
        substrate_report, trace, interface_receipt, benchmark_report,
    )
    return substrate_report, trace, interface_receipt, benchmark_report, certification_report


def test_same_input_same_pack_bytes() -> None:
    """Same input must produce byte-identical packs."""
    stack = _build_full_stack()
    pack_a, _ = run_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    pack_b, _ = run_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    assert pack_a.to_canonical_bytes() == pack_b.to_canonical_bytes()


def test_same_input_same_stable_hash() -> None:
    """Same input must produce identical stable hashes."""
    stack = _build_full_stack()
    pack_a, _ = run_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    pack_b, _ = run_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    assert pack_a.stable_hash == pack_b.stable_hash


def test_broken_lineage_rejection() -> None:
    """Broken trace lineage must be rejected."""
    stack = _build_full_stack()
    broken_trace = replace(stack[1], input_stable_hash="f" * 64)
    with pytest.raises(ValueError, match="broken lineage"):
        build_experimental_research_pack(
            substrate_report=stack[0], trace=broken_trace,
            interface_receipt=stack[2], benchmark_report=stack[3],
            certification_report=stack[4],
        )


def test_missing_artifact_rejection() -> None:
    """Invalid receipts must be rejected as missing artifacts."""
    stack = _build_full_stack()
    bad_receipt = replace(stack[2], validation_passed=False)
    with pytest.raises(ValueError, match="missing artifact"):
        build_experimental_research_pack(
            substrate_report=stack[0], trace=stack[1],
            interface_receipt=bad_receipt, benchmark_report=stack[3],
            certification_report=stack[4],
        )


def test_canonical_export_stability() -> None:
    """Canonical JSON export must be stable across calls."""
    stack = _build_full_stack()
    pack, receipt = run_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    json_a = export_research_pack_json(pack, receipt)
    json_b = export_research_pack_json(pack, receipt)
    assert json_a == json_b
    assert isinstance(json_a, str)
    assert len(json_a) > 0


def test_receipt_integrity() -> None:
    """Receipt hash must match recomputed hash."""
    stack = _build_full_stack()
    pack, receipt = run_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    assert receipt.receipt_hash == receipt.stable_hash()
    assert receipt.pack_hash == pack.stable_hash
    assert receipt.manifest_hash == pack.manifest.stable_hash()


def test_manifest_integrity() -> None:
    """Manifest must record correct artifact metadata."""
    stack = _build_full_stack()
    pack, _ = run_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    manifest = pack.manifest
    assert manifest.frame_count == stack[1].frame_count
    assert manifest.node_count == len(stack[1].node_ids)
    assert manifest.benchmark_case_count == len(stack[3].results)
    assert manifest.certification_passed == stack[4].result.validation_passed
    assert manifest.artifact_hashes["substrate_report"] == stack[0].stable_hash
    assert manifest.artifact_hashes["trace"] == stack[1].stable_hash
    assert manifest.artifact_hashes["benchmark_report"] == stack[3].stable_hash
    assert manifest.artifact_hashes["certification_report"] == stack[4].report_hash
    # Byte sizes must be positive
    for key, size in manifest.artifact_byte_sizes.items():
        assert size > 0, f"artifact byte size for {key} must be positive"


def test_wrapper_manual_equivalence() -> None:
    """run_experimental_research_pack must match manual build_* calls."""
    stack = _build_full_stack()
    pack_wrapped, receipt_wrapped = run_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    pack_manual = build_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    receipt_manual = build_research_receipt(pack_manual)
    assert pack_wrapped.to_canonical_bytes() == pack_manual.to_canonical_bytes()
    assert receipt_wrapped.to_canonical_bytes() == receipt_manual.to_canonical_bytes()


def test_bounded_metric_validation() -> None:
    """All summary metrics must be bounded in [0, 1]."""
    stack = _build_full_stack()
    pack, receipt = run_experimental_research_pack(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4],
    )
    required_keys = {
        "simulation_integrity_score",
        "interface_integrity_score",
        "benchmark_integrity_score",
        "certification_score",
        "global_reproducibility_score",
    }
    assert required_keys <= set(pack.summary_metrics.keys())
    for key, value in pack.summary_metrics.items():
        assert 0.0 <= value <= 1.0, f"{key} = {value} not in [0, 1]"
    assert 0.0 <= receipt.global_reproducibility_score <= 1.0


def test_repeated_run_byte_identity() -> None:
    """Repeated runs must produce byte-identical output."""
    stack = _build_full_stack()
    payloads = tuple(
        run_experimental_research_pack(
            substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
            benchmark_report=stack[3], certification_report=stack[4],
        )[0].to_canonical_bytes()
        for _ in range(8)
    )
    assert len(set(payloads)) == 1


def test_schema_mismatch_rejection() -> None:
    """Mismatched config schema version must be rejected."""
    stack = _build_full_stack()
    bad_config = ExperimentalResearchPackConfig(schema_version="v0")
    with pytest.raises(ValueError, match="schema mismatch"):
        build_experimental_research_pack(
            substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
            benchmark_report=stack[3], certification_report=stack[4],
            config=bad_config,
        )


def test_manifest_build_standalone() -> None:
    """build_artifact_manifest must produce a deterministic standalone manifest."""
    stack = _build_full_stack()
    config = ExperimentalResearchPackConfig()
    manifest_a = build_artifact_manifest(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4], config=config,
    )
    manifest_b = build_artifact_manifest(
        substrate_report=stack[0], trace=stack[1], interface_receipt=stack[2],
        benchmark_report=stack[3], certification_report=stack[4], config=config,
    )
    assert manifest_a.to_canonical_bytes() == manifest_b.to_canonical_bytes()
    assert manifest_a.stable_hash() == manifest_b.stable_hash()
