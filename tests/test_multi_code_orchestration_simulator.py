# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.1.2 multi-code orchestration simulator."""

from __future__ import annotations

import json

import pytest

from qec.simulation.multi_code_orchestration_simulator import (
    MULTI_CODE_ORCHESTRATION_SIMULATOR_VERSION,
    MultiCodeOrchestrationValidationError,
    SimulationLaneResult,
    build_multi_code_orchestration,
    orchestration_replay_identity,
    validate_multi_code_orchestration,
)


def _h(ch: str) -> str:
    return ch * 64


def _manifest(*, lane_order: tuple[str, ...] = ("lane-b", "lane-a"), seed: int | str = 77) -> dict:
    return {
        "orchestration_version": MULTI_CODE_ORCHESTRATION_SIMULATOR_VERSION,
        "experiment_id": "orchestrator-exp-001",
        "orchestration_policy": "strict_deterministic",
        "lane_order": lane_order,
        "upstream_package_hashes": (_h("f"), _h("e")),
        "seed": seed,
        "policy_flags": ("benchmarkable", "replay_safe"),
        "notes": ("v138.1.2", "multi-lane"),
    }


def _lanes() -> tuple[dict, ...]:
    return (
        {
            "lane_id": "lane-a",
            "code_family": "surface_code",
            "topology_family": "grid",
            "package_hash": _h("1"),
            "priority_rank": 20,
            "execution_policy": "simulate_only",
            "metadata": {"k": 1},
        },
        {
            "lane_id": "lane-b",
            "code_family": "qldpc",
            "topology_family": "hypergraph",
            "package_hash": _h("2"),
            "priority_rank": 10,
            "execution_policy": "simulate_only",
            "metadata": {"k": 2},
        },
    )


def test_same_lanes_different_input_order_same_execution_hash() -> None:
    sim_a = build_multi_code_orchestration(manifest=_manifest(), lanes=_lanes())
    sim_b = build_multi_code_orchestration(manifest=_manifest(), lanes=tuple(reversed(_lanes())))
    assert sim_a.receipt.execution_hash == sim_b.receipt.execution_hash


def test_duplicate_lane_id_rejection() -> None:
    lanes = list(_lanes())
    lanes[1] = dict(lanes[1], lane_id="lane-a", package_hash=_h("3"))
    with pytest.raises(MultiCodeOrchestrationValidationError, match="duplicate lane ids"):
        build_multi_code_orchestration(manifest=_manifest(), lanes=tuple(lanes))


def test_invalid_package_hash_rejection() -> None:
    lanes = list(_lanes())
    lanes[0] = dict(lanes[0], package_hash="abc")
    with pytest.raises(MultiCodeOrchestrationValidationError, match="package_hash"):
        build_multi_code_orchestration(manifest=_manifest(), lanes=tuple(lanes))


def test_priority_ordering_determinism() -> None:
    sim = build_multi_code_orchestration(manifest=_manifest(), lanes=_lanes())
    assert tuple(lane.lane_id for lane in sim.lanes) == ("lane-b", "lane-a")


def test_same_seed_same_orchestration_bytes() -> None:
    sim_a = build_multi_code_orchestration(manifest=_manifest(seed=99), lanes=_lanes())
    sim_b = build_multi_code_orchestration(manifest=_manifest(seed=99), lanes=_lanes())
    assert sim_a.to_canonical_json() == sim_b.to_canonical_json()


def test_changed_lane_order_only_changes_when_canonical_order_changes() -> None:
    sim_a = build_multi_code_orchestration(manifest=_manifest(lane_order=("lane-a", "lane-b")), lanes=_lanes())
    sim_b = build_multi_code_orchestration(manifest=_manifest(lane_order=("lane-b", "lane-a")), lanes=_lanes())
    assert sim_a.receipt.receipt_hash == sim_b.receipt.receipt_hash

    changed_lanes = list(_lanes())
    changed_lanes[0] = dict(changed_lanes[0], priority_rank=1)
    sim_c = build_multi_code_orchestration(manifest=_manifest(), lanes=tuple(changed_lanes))
    assert sim_c.receipt.receipt_hash != sim_a.receipt.receipt_hash


def test_stable_replay_identity() -> None:
    sim = build_multi_code_orchestration(manifest=_manifest(), lanes=_lanes())
    result = sim.results[0]
    replay_a = orchestration_replay_identity(
        manifest_hash=sim.receipt.manifest_hash,
        lane_id=result.lane_id,
        package_hash=result.package_hash,
        lane_execution_hash=result.execution_hash,
    )
    replay_b = orchestration_replay_identity(
        manifest_hash=sim.receipt.manifest_hash,
        lane_id=result.lane_id,
        package_hash=result.package_hash,
        lane_execution_hash=result.execution_hash,
    )
    assert replay_a == replay_b
    assert replay_a == result.replay_identity


def test_canonical_json_round_trip() -> None:
    sim = build_multi_code_orchestration(manifest=_manifest(), lanes=_lanes())
    j1 = sim.to_canonical_json()
    j2 = json.dumps(json.loads(j1), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    assert j1 == j2


def test_receipt_hash_matches_returned_receipt_payload() -> None:
    sim = build_multi_code_orchestration(manifest=_manifest(), lanes=_lanes())
    assert sim.receipt.receipt_hash == sim.receipt.stable_hash()


def test_results_package_hash_must_match_lane_package_hash() -> None:
    sim = build_multi_code_orchestration(manifest=_manifest(), lanes=_lanes())
    bad_results = list(sim.results)
    bad_results[0] = SimulationLaneResult(
        lane_id=bad_results[0].lane_id,
        package_hash=_h("a"),
        execution_hash=bad_results[0].execution_hash,
        status=bad_results[0].status,
        topology_stability_score=bad_results[0].topology_stability_score,
        replay_identity=bad_results[0].replay_identity,
        metadata=bad_results[0].metadata,
    )
    report = validate_multi_code_orchestration(
        type(sim)(manifest=sim.manifest, lanes=sim.lanes, results=tuple(bad_results), receipt=sim.receipt)
    )
    assert not report.valid
    assert any("results[0].package_hash must match lanes[0].package_hash" in err for err in report.errors)


def test_invalid_code_family_rejection() -> None:
    lanes = list(_lanes())
    lanes[0] = dict(lanes[0], code_family="unknown_code")
    with pytest.raises(MultiCodeOrchestrationValidationError, match="unsupported lane.code_family"):
        build_multi_code_orchestration(manifest=_manifest(), lanes=tuple(lanes))


def test_none_required_field_rejected() -> None:
    lanes = list(_lanes())
    # lane_id=None must fail fast; "None" string must never slip through
    lanes[0] = dict(lanes[0], lane_id=None)
    with pytest.raises(MultiCodeOrchestrationValidationError, match="lane_id"):
        build_multi_code_orchestration(manifest=_manifest(), lanes=tuple(lanes))

    # experiment_id=None in manifest must also fail
    manifest = dict(_manifest())
    manifest["experiment_id"] = None
    with pytest.raises(MultiCodeOrchestrationValidationError, match="experiment_id"):
        build_multi_code_orchestration(manifest=manifest, lanes=_lanes())


def test_nan_inf_rejection() -> None:
    sim = build_multi_code_orchestration(manifest=_manifest(), lanes=_lanes())
    bad_results = list(sim.results)
    bad_results[0] = SimulationLaneResult(
        lane_id=bad_results[0].lane_id,
        package_hash=bad_results[0].package_hash,
        execution_hash=bad_results[0].execution_hash,
        status=bad_results[0].status,
        topology_stability_score=float("nan"),
        replay_identity=bad_results[0].replay_identity,
        metadata=bad_results[0].metadata,
    )
    report_nan = validate_multi_code_orchestration(
        type(sim)(manifest=sim.manifest, lanes=sim.lanes, results=tuple(bad_results), receipt=sim.receipt)
    )
    assert not report_nan.valid
    assert any("topology_stability_score" in err for err in report_nan.errors)

    bad_results[0] = SimulationLaneResult(
        lane_id=bad_results[0].lane_id,
        package_hash=bad_results[0].package_hash,
        execution_hash=bad_results[0].execution_hash,
        status=bad_results[0].status,
        topology_stability_score=float("inf"),
        replay_identity=bad_results[0].replay_identity,
        metadata=bad_results[0].metadata,
    )
    report_inf = validate_multi_code_orchestration(
        type(sim)(manifest=sim.manifest, lanes=sim.lanes, results=tuple(bad_results), receipt=sim.receipt)
    )
    assert not report_inf.valid
    assert any("topology_stability_score" in err for err in report_inf.errors)
