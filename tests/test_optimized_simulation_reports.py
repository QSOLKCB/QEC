from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from subprocess import CalledProcessError, check_output

import pytest

import qec.analysis.optimized_simulation_reports as reports_module
from qec.analysis.optimized_simulation_reports import (
    _SCHEMA_VERSION,
    OptimizedSimulationReport,
    SimulationReportSection,
    SimulationReplaySummary,
    SimulationBenchmarkSummary,
    SimulationTelemetrySummary,
    build_optimized_simulation_report,
    build_simulation_benchmark_summary,
    build_simulation_optimization_lineage,
    build_simulation_replay_summary,
    build_simulation_report_section,
    build_simulation_report_summary,
    build_simulation_telemetry_summary,
    validate_optimized_simulation_report,
    validate_simulation_section,
    validate_simulation_replay,
    validate_simulation_benchmark,
    validate_simulation_telemetry,
)


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _report() -> OptimizedSimulationReport:
    r = build_simulation_replay_summary(replay_passed=True, replay_receipt_hash=_h("r"), replay_scenario_count=1, replay_observation_count=1, replay_comparison_count=1, replay_failure_count=0)
    b = build_simulation_benchmark_summary(benchmark_passed=True, benchmark_receipt_hash=_h("b"), benchmark_measurement_count=1, benchmark_claim_count=1, benchmark_failure_count=0)
    t = build_simulation_telemetry_summary(telemetry_passed=True, telemetry_receipt_hash=_h("t"), telemetry_metric_count=1, telemetry_aggregation_count=1, telemetry_claim_count=1, telemetry_failure_count=0)
    l = build_simulation_optimization_lineage(optimized_simulation_spec_hash=_h("s"), backend_equivalence_replay_receipt_hash=_h("r"), optimized_qec_benchmark_receipt_hash=_h("b"), optimized_telemetry_receipt_hash=_h("t"), optimization_scope="x", dependency_name="qldpc", dependency_class="qldpc_external")
    s0 = build_simulation_report_section(section_index=0, section_name="replay", section_kind="REPLAY", section_summary="summary", source_hashes=(_h("r"),), section_status="PASS", reason="ok")
    s1 = build_simulation_report_section(section_index=1, section_name="benchmark", section_kind="BENCHMARK", section_summary="summary", source_hashes=(_h("b"),), section_status="PASS", reason="ok")
    s2 = build_simulation_report_section(section_index=2, section_name="telemetry", section_kind="TELEMETRY", section_summary="summary", source_hashes=(_h("t"),), section_status="PASS", reason="ok")
    sm = build_simulation_report_summary(report_passed=True, replay_passed=True, benchmark_passed=True, telemetry_passed=True, report_section_count=3, report_failure_count=0)
    return build_optimized_simulation_report(schema_version=_SCHEMA_VERSION, report_mode="DETERMINISTIC_SIMULATION_REPORT", report_status="OPTIMIZED_SIMULATION_REPORT_PASSED", dependency_name="qldpc", dependency_class="qldpc_external", optimization_scope="x", source_optimized_simulation_spec_hash=_h("s"), source_backend_equivalence_replay_receipt_hash=_h("r"), source_optimized_qec_benchmark_receipt_hash=_h("b"), source_optimized_telemetry_receipt_hash=_h("t"), report_sections=(s0, s1, s2), replay_summary=r, benchmark_summary=b, telemetry_summary=t, optimization_lineage=l, report_summary=sm, report_generated_from_canonical_receipts=True, report_contains_runtime_execution=False, report_contains_live_telemetry=False, report_contains_benchmark_execution=False)


def test_hash_stability_and_idempotence() -> None:
    a = _report()
    b = _report()
    assert a.optimized_simulation_report_hash == b.optimized_simulation_report_hash
    assert a.to_canonical_json() == b.to_canonical_json()


def test_malformed_and_wrong_hash_rejection() -> None:
    r = _report()
    bad = OptimizedSimulationReport(**{**r.__dict__, "optimized_simulation_report_hash": "xyz"})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_optimized_simulation_report(bad)
    wrong = OptimizedSimulationReport(**{**r.__dict__, "optimized_simulation_report_hash": _h("wrong")})
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_optimized_simulation_report(wrong)


def test_count_ordering_and_status_semantics() -> None:
    r = _report()
    bad_summary = build_simulation_report_summary(report_passed=True, replay_passed=True, benchmark_passed=True, telemetry_passed=True, report_section_count=9, report_failure_count=0)
    x = OptimizedSimulationReport(**{**r.__dict__, "report_summary": bad_summary, "optimized_simulation_report_hash": ""})
    with pytest.raises(ValueError, match="COUNT_MISMATCH"):
        validate_optimized_simulation_report(x, True)


def test_no_runtime_benchmark_or_telemetry_claim_tokens() -> None:
    r = _report()
    bad_section = build_simulation_report_section(section_index=0, section_name="replay", section_kind="REPLAY", section_summary="simulation executed", source_hashes=(_h("r"),), section_status="PASS", reason="ok")
    x = OptimizedSimulationReport(**{**r.__dict__, "report_sections": (bad_section, r.report_sections[1], r.report_sections[2]), "optimized_simulation_report_hash": ""})
    with pytest.raises(ValueError, match="FORBIDDEN_CONTENT"):
        validate_optimized_simulation_report(x, True)


def test_passed_failure_count_consistency() -> None:
    # Test replay: passed=True but failure_count > 0 should fail
    with pytest.raises(ValueError, match="PASSED_FAILURE_COUNT_MISMATCH"):
        build_simulation_replay_summary(
            replay_passed=True, replay_receipt_hash=_h("r"),
            replay_scenario_count=1, replay_observation_count=1,
            replay_comparison_count=1, replay_failure_count=1
        )
    # Test replay: passed=False but failure_count == 0 should fail
    with pytest.raises(ValueError, match="FAILED_FAILURE_COUNT_MISMATCH"):
        build_simulation_replay_summary(
            replay_passed=False, replay_receipt_hash=_h("r"),
            replay_scenario_count=1, replay_observation_count=1,
            replay_comparison_count=1, replay_failure_count=0
        )
    # Test benchmark: passed=True but failure_count > 0 should fail
    with pytest.raises(ValueError, match="PASSED_FAILURE_COUNT_MISMATCH"):
        build_simulation_benchmark_summary(
            benchmark_passed=True, benchmark_receipt_hash=_h("b"),
            benchmark_measurement_count=1, benchmark_claim_count=1,
            benchmark_failure_count=1
        )
    # Test telemetry: passed=True but failure_count > 0 should fail
    with pytest.raises(ValueError, match="PASSED_FAILURE_COUNT_MISMATCH"):
        build_simulation_telemetry_summary(
            telemetry_passed=True, telemetry_receipt_hash=_h("t"),
            telemetry_metric_count=1, telemetry_aggregation_count=1,
            telemetry_claim_count=1, telemetry_failure_count=1
        )


def test_invalid_source_hashes_rejected() -> None:
    # Test that invalid source_hashes (non-64-char hex) are rejected
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_simulation_report_section(
            section_index=0, section_name="replay", section_kind="REPLAY",
            section_summary="summary", source_hashes=("not-a-hash",),
            section_status="PASS", reason="ok"
        )


def _find_git_root(start_path: Path) -> Path | None:
    """Find the git repository root starting from the given path."""
    current = start_path
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def test_forbidden_import_scanning_and_decoder_boundary() -> None:
    # Use module's __file__ to locate the source file reliably
    module_path = Path(reports_module.__file__)
    src = module_path.read_text(encoding="utf-8")
    for token in ("numpy", "scipy", "pandas", "matplotlib", "qiskit", "stim", "pymatching", "requests", "urllib", "subprocess", "eval(", "exec(", "os.system"):
        assert token not in src
    # Guard git-based test: skip if git is not available or not in a git repo
    if shutil.which("git") is None:
        pytest.skip("git not available on PATH")
    git_dir = _find_git_root(module_path.parent)
    if git_dir is None:
        pytest.skip("not in a git repository")
    try:
        diff = check_output(["git", "diff", "--name-only"], cwd=git_dir).decode("utf-8")
    except CalledProcessError:
        pytest.skip("git diff failed")
    assert not any(line.startswith("src/qec/decoder/") for line in diff.splitlines())
