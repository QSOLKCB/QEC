"""Tests for terminal kernel handoff release (v137.1.20)."""

from __future__ import annotations

import hashlib
import json

import pytest

from qec.analysis.terminal_kernel_handoff_release import (
    TERMINAL_KERNEL_HANDOFF_VERSION,
    build_terminal_kernel_handoff_release,
)


def _build_release():
    return build_terminal_kernel_handoff_release(
        benchmark_samples=(
            {"benchmark_id": "phase_scan", "operation_count": 720, "determinism_score": 0.99},
            {"benchmark_id": "baseline_decode", "operation_count": 500, "determinism_score": 1.0},
            {"benchmark_id": "stability_sweep", "operation_count": 500, "determinism_score": 1.0},
        ),
        replay_anchor="replay_anchor_v137_1_20",
        provenance_anchor="provenance_anchor_v137_1_20",
        advisory_codes=("strict_hardening", "kernel_handoff", "strict_hardening"),
        observatory_metrics={"advisory_coverage": 1.0, "replay_health": 0.95, "provenance_health": 0.98},
        enable_terminal_handoff_freeze=True,
    )


def test_repeated_run_determinism_and_identity_stability():
    a = _build_release()
    b = _build_release()

    assert a.to_dict() == b.to_dict()
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.replay_identity == b.replay_identity


def test_canonical_json_and_bytes_stability():
    release = _build_release()

    canonical_json = release.to_canonical_json()
    canonical_bytes = release.to_canonical_bytes()

    assert canonical_bytes == canonical_json.encode("utf-8")
    assert json.loads(canonical_json)["version"] == TERMINAL_KERNEL_HANDOFF_VERSION


def test_replay_identity_is_stable_sha256_hex():
    release = _build_release()

    assert len(release.replay_identity) == 64
    assert int(release.replay_identity, 16) >= 0


def test_deterministic_benchmark_freeze_output_ordering_and_stability():
    release = _build_release()
    frozen = release.benchmark_freeze.frozen_samples

    assert tuple(sample.benchmark_id for sample in frozen) == (
        "baseline_decode",
        "stability_sweep",
        "phase_scan",
    )
    assert release.benchmark_freeze.sample_count == 3
    assert release.benchmark_freeze.baseline_operation_count == 500


def test_migration_contract_and_bootstrap_schema_stability():
    release = _build_release()

    assert release.migration_contract.from_version == TERMINAL_KERNEL_HANDOFF_VERSION
    assert release.migration_contract.to_version == "v137.2.x"
    assert release.migration_contract.contract_id == "v1371x_to_v1372x_terminal_contract"
    assert release.bootstrap_schema.schema_id == "autonomous_planning_kernel.bootstrap"
    assert release.bootstrap_schema.strict_mode is True

    digest = hashlib.sha256(release.to_canonical_bytes()).hexdigest()
    assert len(digest) == 64


def test_fail_fast_invalid_input_handling_and_bounded_validation():
    with pytest.raises(ValueError, match="enable_terminal_handoff_freeze"):
        build_terminal_kernel_handoff_release(
            benchmark_samples=({"benchmark_id": "b", "operation_count": 1, "determinism_score": 1.0},),
            replay_anchor="r",
            provenance_anchor="p",
            advisory_codes=(),
            observatory_metrics={},
            enable_terminal_handoff_freeze=False,
        )

    with pytest.raises(ValueError, match="benchmark_samples must be non-empty"):
        build_terminal_kernel_handoff_release(
            benchmark_samples=(),
            replay_anchor="r",
            provenance_anchor="p",
            advisory_codes=(),
            observatory_metrics={},
            enable_terminal_handoff_freeze=True,
        )

    with pytest.raises(ValueError, match=r"determinism_score must be in \[0, 1\]"):
        build_terminal_kernel_handoff_release(
            benchmark_samples=({"benchmark_id": "b", "operation_count": 1, "determinism_score": 1.1},),
            replay_anchor="r",
            provenance_anchor="p",
            advisory_codes=(),
            observatory_metrics={},
            enable_terminal_handoff_freeze=True,
        )

    with pytest.raises(ValueError, match=r"observatory_metrics\[m\] must be in \[0, 1\]"):
        build_terminal_kernel_handoff_release(
            benchmark_samples=({"benchmark_id": "b", "operation_count": 1, "determinism_score": 1.0},),
            replay_anchor="r",
            provenance_anchor="p",
            advisory_codes=(),
            observatory_metrics={"m": 1.1},
            enable_terminal_handoff_freeze=True,
        )
