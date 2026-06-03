from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from qec.analysis import decoder_implementation_boundary_receipts as ib

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64
G = "1" * 64
H = "2" * 64
I = "3" * 64


def _hash(obj, field):
    return ib._hash_payload(ib._dataclass_payload(obj, exclude_hash_field=field))


def _unsafe_replace(obj, **changes):
    clone = object.__new__(type(obj))
    for f in obj.__dataclass_fields__:
        object.__setattr__(clone, f, changes.get(f, getattr(obj, f)))
    return clone


def _rehash(obj, field):
    return _unsafe_replace(obj, **{field: _hash(obj, field)})


def _fixture(order=False):
    upstream = ib.build_decoder_implementation_upstream_binding(
        upstream_canonical_decoder_baseline_receipt_hash=A,
        upstream_decoder_candidate_manifest_hash=B,
        upstream_decoder_replay_equivalence_receipt_hash=C,
        upstream_decoder_optimization_contract_hash=D,
        upstream_decoder_fast_path_equivalence_receipt_hash=E,
        candidate_declaration_hash=F,
        fast_path_identity_hash=G,
    )
    ident = ib.build_decoder_implementation_identity(
        associated_candidate_declaration_hash=F,
        associated_fast_path_identity_hash=G,
        associated_fast_path_equivalence_receipt_hash=E,
    )
    art1 = ib.build_decoder_implementation_artifact(
        artifact_id="a1",
        artifact_path="implementation_boundaries/a.json",
        artifact_sha256=A,
        artifact_schema_hash=B,
    )
    art2 = ib.build_decoder_implementation_artifact(
        artifact_id="a2",
        artifact_path="implementation_boundaries/b.json",
        artifact_sha256=C,
        artifact_role="IMPLEMENTATION_SOURCE_HASH_DECLARATION",
        artifact_schema_hash=D,
    )
    artifacts = (art2, art1) if order else (art1, art2)
    source = ib.build_decoder_implementation_source_boundary(artifacts)
    runtime = ib.build_decoder_implementation_runtime_boundary()
    config = ib.build_decoder_implementation_config_boundary(config_schema_hash=A, config_payload_hash=B)
    build = ib.build_decoder_implementation_build_boundary(
        source_boundary_hash=source.decoder_implementation_source_boundary_hash,
        build_manifest_hash=C,
        dependency_manifest_hash=D,
    )
    equiv = ib.build_decoder_implementation_equivalence_binding(
        required_replay_equivalence_receipt_hash=C,
        required_fast_path_equivalence_receipt_hash=E,
        required_optimization_contract_hash=D,
    )
    audit = ib.build_decoder_implementation_audit_boundary()
    rollback = ib.build_decoder_implementation_rollback_gate()
    authority = ib.build_decoder_implementation_authority_boundary()
    receipt = ib.build_decoder_implementation_boundary_receipt(
        upstream, ident, source, runtime, config, build, equiv, audit, rollback, authority
    )
    return locals()


def _assert_code(fn, code):
    with pytest.raises(ib.DecoderImplementationBoundaryError) as exc:
        fn()
    assert exc.value.code is code
    assert code.value in str(exc.value)
    assert exc.value.detail


def test_happy_path_builds_validates_and_is_frozen():
    fx = _fixture()
    checks = [
        ("upstream", ib.validate_decoder_implementation_upstream_binding),
        ("ident", ib.validate_decoder_implementation_identity),
        ("art1", ib.validate_decoder_implementation_artifact),
        ("source", ib.validate_decoder_implementation_source_boundary),
        ("runtime", ib.validate_decoder_implementation_runtime_boundary),
        ("config", ib.validate_decoder_implementation_config_boundary),
        ("build", ib.validate_decoder_implementation_build_boundary),
        ("equiv", ib.validate_decoder_implementation_equivalence_binding),
        ("audit", ib.validate_decoder_implementation_audit_boundary),
        ("rollback", ib.validate_decoder_implementation_rollback_gate),
        ("authority", ib.validate_decoder_implementation_authority_boundary),
        ("receipt", ib.validate_decoder_implementation_boundary_receipt),
    ]
    for name, validator in checks:
        assert validator(fx[name]) is fx[name]
    receipt = fx["receipt"]
    assert len(receipt.decoder_implementation_boundary_receipt_hash) == 64
    assert receipt.implementation_boundary_safe is True
    assert receipt.candidate_remains_adapter_only is True
    assert receipt.runtime_enabled is False
    assert receipt.implementation_authority_allowed is False
    assert receipt.benchmark_claim_allowed is False
    assert receipt.speedup_claim_allowed is False
    assert receipt.promotion_allowed is False
    assert receipt.global_correctness_claim_allowed is False
    with pytest.raises(FrozenInstanceError):
        receipt.runtime_enabled = True


def test_canonical_json_hash_determinism_and_ordering():
    r1 = _fixture(False)["receipt"]
    r2 = _fixture(True)["receipt"]
    assert r1.decoder_implementation_boundary_receipt_hash == r2.decoder_implementation_boundary_receipt_hash
    assert _fixture()["receipt"].decoder_implementation_boundary_receipt_hash == r1.decoder_implementation_boundary_receipt_hash
    assert _fixture(False)["source"].source_tree_hash == _fixture(True)["source"].source_tree_hash
    payload = {"z": tuple(sorted({"b", "a"})), "a": {"k": "v"}}
    assert ib._hash_payload(payload) == ib._hash_payload(payload)


def test_self_hash_exclusion_and_stale_hashes_fail():
    fx = _fixture()
    art = _unsafe_replace(fx["art1"], artifact_role="IMPLEMENTATION_CONFIG_DECLARATION")
    _assert_code(lambda: ib.validate_decoder_implementation_artifact(art), ib.DecoderImplementationBoundaryErrorCode.HASH_MISMATCH)
    source = _unsafe_replace(fx["source"], source_files_exist_required=True)
    _assert_code(lambda: ib.validate_decoder_implementation_source_boundary(source), ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY)
    receipt = _unsafe_replace(fx["receipt"], runtime_enabled=True)
    _assert_code(lambda: ib.validate_decoder_implementation_boundary_receipt(receipt), ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY)
    assert _hash(fx["receipt"], "decoder_implementation_boundary_receipt_hash") == fx["receipt"].decoder_implementation_boundary_receipt_hash
    assert _hash(fx["source"], "decoder_implementation_source_boundary_hash") == fx["source"].decoder_implementation_source_boundary_hash
    assert _hash(fx["art1"], "decoder_implementation_artifact_hash") == fx["art1"].decoder_implementation_artifact_hash


def test_child_before_aggregate_validation():
    fx = _fixture()
    bad_art = _rehash(_unsafe_replace(fx["art1"], executable_runtime_artifact=True), "decoder_implementation_artifact_hash")
    bad_source = _unsafe_replace(fx["source"], implementation_artifacts=(bad_art, fx["art2"]))
    bad_source = _rehash(bad_source, "decoder_implementation_source_boundary_hash")
    _assert_code(lambda: ib.validate_decoder_implementation_source_boundary(bad_source), ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT)
    forged_receipt = _rehash(_unsafe_replace(fx["receipt"], source_boundary=bad_source), "decoder_implementation_boundary_receipt_hash")
    _assert_code(lambda: ib.validate_decoder_implementation_boundary_receipt(forged_receipt), ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT)
    bad_runtime = _rehash(_unsafe_replace(fx["runtime"], network_allowed=True), "decoder_implementation_runtime_boundary_hash")
    forged_receipt = _rehash(_unsafe_replace(fx["receipt"], runtime_boundary=bad_runtime), "decoder_implementation_boundary_receipt_hash")
    _assert_code(lambda: ib.validate_decoder_implementation_boundary_receipt(forged_receipt), ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY)
    bad_authority = _rehash(_unsafe_replace(fx["authority"], ml_decoder_authority_allowed=True), "decoder_implementation_authority_boundary_hash")
    forged_receipt = _rehash(_unsafe_replace(fx["receipt"], authority_boundary=bad_authority), "decoder_implementation_boundary_receipt_hash")
    _assert_code(lambda: ib.validate_decoder_implementation_boundary_receipt(forged_receipt), ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY)


@pytest.mark.parametrize("field,value,code", [
    ("previous_release_tag", "v166.3", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("previous_release_url", "https://example.invalid", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("implementation_boundary_release", "v166.6", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("upstream_canonical_decoder_baseline_receipt_hash", "A"*64, ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("upstream_decoder_candidate_manifest_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("upstream_decoder_replay_equivalence_receipt_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("upstream_decoder_optimization_contract_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("upstream_decoder_fast_path_equivalence_receipt_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("candidate_declaration_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("fast_path_identity_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("candidate_name", "", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("candidate_version", "", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("replay_equivalence_proven_for_declared_corpus", False, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("optimization_contract_safe", False, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("fast_path_equivalence_proven_for_declared_corpus", False, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("candidate_adapter_only", False, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("candidate_promoted", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("baseline_immutable", False, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("baseline_mutation_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("runtime_authority_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("runtime_authority_allowed", 0, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
])
def test_upstream_binding_validation(field, value, code):
    obj = _unsafe_replace(_fixture()["upstream"], **{field: value})
    obj = _rehash(obj, "decoder_implementation_upstream_binding_hash") if code is not ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH else obj
    _assert_code(lambda: ib.validate_decoder_implementation_upstream_binding(obj), code)


@pytest.mark.parametrize("field,value,code", [
    ("implementation_id", "", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("implementation_name", "", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("implementation_version", "", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("implementation_kind", "OTHER", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("implementation_status", "ENABLED", ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("implementation_mode", "RUNTIME", ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("associated_candidate_declaration_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("associated_fast_path_identity_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("associated_fast_path_equivalence_receipt_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("adapter_only", False, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("boundary_only", False, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("runtime_enabled", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("importable_runtime_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("implementation_authority_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("promotion_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("benchmark_claim_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("speedup_claim_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("hardware_authority_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("qec_advantage_claim_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY),
    ("implementation_name", "speed proves correctness", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
])
def test_implementation_identity_validation(field, value, code):
    obj = _unsafe_replace(_fixture()["ident"], **{field: value})
    obj = _rehash(obj, "decoder_implementation_identity_hash") if code is not ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH else obj
    _assert_code(lambda: ib.validate_decoder_implementation_identity(obj), code)


@pytest.mark.parametrize("field,value,code", [
    ("artifact_id", "", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_path", "/implementation_boundaries/a.json", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_path", "implementation_boundaries/./a.json", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_path", "implementation_boundaries/../a.json", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_path", "implementation_boundaries//a.json", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_path", "implementation_boundaries\\a.json", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_path", "src/qec/decoder/a.py", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_sha256", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("artifact_role", "OTHER", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_mode", "OTHER", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_language", "PYTHON", ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("artifact_schema_hash", "x", ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH),
    ("executable_runtime_artifact", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("import_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("execution_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("benchmark_allowed", True, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
    ("benchmark_allowed", 0, ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT),
])
def test_artifact_validation(field, value, code):
    obj = _unsafe_replace(_fixture()["art1"], **{field: value})
    obj = _rehash(obj, "decoder_implementation_artifact_hash") if code is not ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH else obj
    _assert_code(lambda: ib.validate_decoder_implementation_artifact(obj), code)


def test_source_boundary_validation_matrix():
    fx = _fixture(); source = fx["source"]
    cases = [
        ("implementation_source_root", "/implementation_boundaries/"), ("implementation_source_root", "implementation_boundaries"),
        ("implementation_source_root", "src/qec/decoder/"), ("implementation_source_root", "runtime_decoders/"),
        ("source_boundary_mode", "OTHER"), ("implementation_artifacts", ()), ("implementation_artifacts", (object(),)),
        ("implementation_artifact_count", 1), ("implementation_artifact_count", True), ("source_files_exist_required", True),
        ("repository_walk_allowed", True), ("runtime_import_allowed", True), ("runtime_execution_allowed", True),
        ("implementation_file_creation_allowed", True), ("baseline_mutation_allowed", True), ("filesystem_mutation_allowed", True),
    ]
    for field, value in cases:
        obj = _unsafe_replace(source, **{field: value})
        if field != "implementation_artifacts":
            obj = _rehash(obj, "decoder_implementation_source_boundary_hash")
        _assert_code(lambda obj=obj: ib.validate_decoder_implementation_source_boundary(obj), ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT if field in {"implementation_source_root", "implementation_artifacts", "implementation_artifact_count"} else ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY)
    dup_id = _unsafe_replace(fx["art2"], artifact_id="a1")
    dup_id = _rehash(dup_id, "decoder_implementation_artifact_hash")
    dup_path = _unsafe_replace(fx["art2"], artifact_path=fx["art1"].artifact_path)
    dup_path = _rehash(dup_path, "decoder_implementation_artifact_hash")
    outside = _unsafe_replace(fx["art2"], artifact_path="external/decoder_implementation_boundaries/b.json")
    outside = _rehash(outside, "decoder_implementation_artifact_hash")
    for arts in ((fx["art1"], dup_id), (fx["art1"], dup_path), (fx["art1"], outside)):
        obj = _unsafe_replace(source, implementation_artifacts=tuple(sorted(arts, key=lambda a: a.artifact_path)))
        obj = _rehash(obj, "decoder_implementation_source_boundary_hash")
        _assert_code(lambda obj=obj: ib.validate_decoder_implementation_source_boundary(obj), ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT)
    bad_tree = _unsafe_replace(source, source_tree_hash=H)
    _assert_code(lambda: ib.validate_decoder_implementation_source_boundary(bad_tree), ib.DecoderImplementationBoundaryErrorCode.HASH_MISMATCH)


def test_runtime_config_build_equivalence_audit_rollback_authority_validation():
    fx = _fixture()
    groups = [
        (fx["runtime"], ib.validate_decoder_implementation_runtime_boundary, "decoder_implementation_runtime_boundary_hash", ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY, ["runtime_boundary_mode", "declared_boundary_only", "baseline_decoder_import_allowed", "candidate_decoder_import_allowed", "fast_path_import_allowed", "implementation_import_allowed", "runtime_decoder_execution_allowed", "candidate_runtime_execution_allowed", "fast_path_runtime_execution_allowed", "implementation_runtime_execution_allowed", "replay_execution_allowed", "optimization_execution_allowed", "benchmark_execution_allowed", "network_allowed", "heavy_backend_import_allowed", "hardware_sdk_allowed", "filesystem_mutation_allowed"]),
        (fx["config"], ib.validate_decoder_implementation_config_boundary, "decoder_implementation_config_boundary_hash", ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY, ["config_mode", "deterministic_config_ordering", "mutable_runtime_config_allowed", "environment_variable_dependency_allowed", "wall_clock_dependency_allowed", "randomness_dependency_allowed", "filesystem_probe_dependency_allowed", "network_dependency_allowed", "hardware_probe_dependency_allowed"]),
        (fx["build"], ib.validate_decoder_implementation_build_boundary, "decoder_implementation_build_boundary_hash", ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY, ["build_boundary_mode", "build_execution_allowed", "dependency_install_allowed", "network_resolution_allowed", "native_extension_build_allowed", "hardware_specific_build_allowed", "unpinned_dependency_allowed", "build_cache_authority_allowed"]),
        (fx["equiv"], ib.validate_decoder_implementation_equivalence_binding, "decoder_implementation_equivalence_binding_hash", ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY, ["equivalence_mode", "fast_path_equivalence_scope", "declared_corpus_only", "output_schema_match_required", "output_payload_match_required", "canonical_ordering_match_required", "precision_policy", "approximation_policy", "equivalence_required_before_runtime", "implementation_valid_without_fast_path_equivalence"]),
        (fx["audit"], ib.validate_decoder_implementation_audit_boundary, "decoder_implementation_audit_boundary_hash", ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY, ["audit_mode", "static_boundary_review_required", "source_hash_review_required", "no_decoder_mutation_review_required", "no_runtime_import_review_required", "no_runtime_execution_review_required", "no_benchmark_claim_review_required", "future_benchmark_ladder_required", "future_rollback_receipt_required", "future_promotion_receipt_required", "audit_complete"]),
        (fx["rollback"], ib.validate_decoder_implementation_rollback_gate, "decoder_implementation_rollback_gate_hash", ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY, ["rollback_gate_mode", "rollback_receipt_required_before_promotion", "required_future_rollback_receipt_kind", "required_future_rollback_release", "rollback_path_deletion_allowed", "baseline_restore_required", "candidate_disable_required_on_failure", "promotion_blocked_without_rollback_receipt", "implementation_disable_required_on_failure"]),
        (fx["authority"], ib.validate_decoder_implementation_authority_boundary, "decoder_implementation_authority_boundary_hash", ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY, ["authority_mode", "candidate_adapter_only", "boundary_only", "runtime_authority_allowed", "implementation_authority_allowed", "benchmark_authority_allowed", "hardware_authority_allowed", "ml_decoder_authority_allowed", "probabilistic_decoder_authority_allowed", "qec_advantage_claim_allowed", "global_correctness_claim_allowed", "silent_replacement_allowed", "baseline_mutation_allowed", "candidate_promotion_allowed"]),
    ]
    for obj, validator, hash_field, code, fields in groups:
        for field in fields:
            if field.endswith("_id"):
                value = ""
            elif field in {"runtime_boundary_mode", "config_mode", "build_boundary_mode", "equivalence_mode", "fast_path_equivalence_scope", "precision_policy", "approximation_policy", "audit_mode", "rollback_gate_mode", "required_future_rollback_receipt_kind", "required_future_rollback_release", "authority_mode"}:
                value = "OTHER"
            else:
                value = not getattr(obj, field)
            bad = _rehash(_unsafe_replace(obj, **{field: value}), hash_field)
            _assert_code(lambda bad=bad, validator=validator: validator(bad), code)
    for obj, validator, hash_field, field in [
        (fx["runtime"], ib.validate_decoder_implementation_runtime_boundary, "decoder_implementation_runtime_boundary_hash", "declared_boundary_only"),
        (fx["config"], ib.validate_decoder_implementation_config_boundary, "decoder_implementation_config_boundary_hash", "deterministic_config_ordering"),
    ]:
        bad = _rehash(_unsafe_replace(obj, **{field: 1}), hash_field)
        _assert_code(lambda bad=bad, validator=validator: validator(bad), ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT)
    for field in ("config_schema_hash", "config_payload_hash"):
        _assert_code(lambda field=field: ib.validate_decoder_implementation_config_boundary(_unsafe_replace(fx["config"], **{field: "x"})), ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH)
    for field in ("build_manifest_hash", "dependency_manifest_hash", "source_boundary_hash"):
        _assert_code(lambda field=field: ib.validate_decoder_implementation_build_boundary(_unsafe_replace(fx["build"], **{field: "x"})), ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH)
    for field in ("required_replay_equivalence_receipt_hash", "required_fast_path_equivalence_receipt_hash", "required_optimization_contract_hash"):
        _assert_code(lambda field=field: ib.validate_decoder_implementation_equivalence_binding(_unsafe_replace(fx["equiv"], **{field: "x"})), ib.DecoderImplementationBoundaryErrorCode.INVALID_HASH)


@pytest.mark.parametrize("field,value", [
    ("receipt_version", "v166.4"), ("receipt_kind", "Other"), ("previous_release_tag", "v166.3"),
    ("previous_release_url", "https://example.invalid"), ("implementation_artifact_count", 99), ("implementation_artifact_count", True),
    ("runtime_enabled", True), ("implementation_authority_allowed", True), ("benchmark_claim_allowed", True),
    ("speedup_claim_allowed", True), ("promotion_allowed", True), ("global_correctness_claim_allowed", True),
])
def test_aggregate_receipt_validation(field, value):
    receipt = _rehash(_unsafe_replace(_fixture()["receipt"], **{field: value}), "decoder_implementation_boundary_receipt_hash")
    expected = ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT if field in {"receipt_version", "receipt_kind", "previous_release_tag", "previous_release_url", "implementation_artifact_count"} else ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY
    _assert_code(lambda: ib.validate_decoder_implementation_boundary_receipt(receipt), expected)


def test_aggregate_cross_hash_mismatches_and_forged_safe_values():
    fx = _fixture(); receipt = fx["receipt"]
    mutations = [
        ("implementation_identity", _rehash(_unsafe_replace(fx["ident"], associated_candidate_declaration_hash=A), "decoder_implementation_identity_hash")),
        ("implementation_identity", _rehash(_unsafe_replace(fx["ident"], associated_fast_path_identity_hash=A), "decoder_implementation_identity_hash")),
        ("implementation_identity", _rehash(_unsafe_replace(fx["ident"], associated_fast_path_equivalence_receipt_hash=A), "decoder_implementation_identity_hash")),
        ("build_boundary", _rehash(_unsafe_replace(fx["build"], source_boundary_hash=A), "decoder_implementation_build_boundary_hash")),
        ("equivalence_binding", _rehash(_unsafe_replace(fx["equiv"], required_replay_equivalence_receipt_hash=A), "decoder_implementation_equivalence_binding_hash")),
        ("equivalence_binding", _rehash(_unsafe_replace(fx["equiv"], required_fast_path_equivalence_receipt_hash=A), "decoder_implementation_equivalence_binding_hash")),
        ("equivalence_binding", _rehash(_unsafe_replace(fx["equiv"], required_optimization_contract_hash=A), "decoder_implementation_equivalence_binding_hash")),
    ]
    for field, child in mutations:
        bad = _rehash(_unsafe_replace(receipt, **{field: child}), "decoder_implementation_boundary_receipt_hash")
        _assert_code(lambda bad=bad: ib.validate_decoder_implementation_boundary_receipt(bad), ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY)
    bad = _rehash(_unsafe_replace(receipt, implementation_boundary_safe=False), "decoder_implementation_boundary_receipt_hash")
    _assert_code(lambda: ib.validate_decoder_implementation_boundary_receipt(bad), ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY)
    bad = _rehash(_unsafe_replace(receipt, candidate_remains_adapter_only=False), "decoder_implementation_boundary_receipt_hash")
    _assert_code(lambda: ib.validate_decoder_implementation_boundary_receipt(bad), ib.DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY)


@pytest.mark.parametrize("phrase", [
    "silent_decoder_replacement", "candidate-replaces-baseline", "decoder replaced because faster",
    "speed proves correctness", "benchmark proves correctness", "benchmark marketing", "runtime promotion",
    "candidate decoder promoted", "probabilistic decoder authority", "ML decoder authority", "hardware authority",
    "QEC advantage proven", "hidden precision drift", "undeclared approximation policy",
    "output accepted as universal canonical truth", "global correctness proven", "replay equivalence implies promotion",
    "replay equivalence implies speedup", "optimization implies correctness", "optimization grants execution authority",
    "contract permits implementation", "fast path accepted", "fast path implemented", "fast path runtime enabled",
    "fast path proves speedup", "benchmark proves fast path", "implementation permission granted",
    "implementation enabled", "implementation proves correctness", "implementation proves speedup",
    "implementation replaces baseline", "runtime implementation authority", "build proves correctness",
    "config grants runtime authority", "fast\npath\timplemented", "speed___proves---correctness",
])
def test_forbidden_semantic_hardening(phrase):
    _assert_code(lambda: ib.build_decoder_implementation_identity(implementation_id=phrase), ib.DecoderImplementationBoundaryErrorCode.INVALID_INPUT)


@pytest.mark.parametrize("phrase", [
    "implementation_boundary_safe", "implementation_boundary_release", "rollback_receipt_required_before_promotion",
    "future_benchmark_ladder_required", "future_promotion_receipt_required",
])
def test_legitimate_boundary_phrases_allowed(phrase):
    assert ib.build_decoder_implementation_identity(implementation_id=phrase)


def test_boundary_static_guards_and_decoder_immutability():
    repo = Path(__file__).resolve().parents[1]
    module = repo / "src/qec/analysis/decoder_implementation_boundary_receipts.py"
    test_file = repo / "tests/test_decoder_implementation_boundary_receipts.py"
    banned_imports = {"qec.decoder", "numpy", "scipy", "qldpc", "stim", "pymatching", "qiskit", "qutip", "pandas", "polars", "torch", "tensorflow", "jax"}
    for path in (module, test_file):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert alias.name not in banned_imports
                    assert root not in banned_imports
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert node.module not in banned_imports
                assert root not in banned_imports
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (node.func.attr == "import_module" and node.args and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith("qec.decoder"))
    text = module.read_text().lower()
    for token in ("socket", "urllib", "requests", "subprocess", "benchmark.mark", "qec.decoder", "torch.nn"):
        assert token not in text
    diff = subprocess.run(["git", "diff", "--name-only", "--", "src/qec/decoder/"], cwd=repo, text=True, stdout=subprocess.PIPE, check=True)
    assert diff.stdout.strip() == ""


def test_hash_seed_stability_subprocess():
    code = """
from qec.analysis import decoder_implementation_boundary_receipts as ib
A='a'*64; B='b'*64; C='c'*64; D='d'*64; E='e'*64; F='f'*64; G='1'*64
u=ib.build_decoder_implementation_upstream_binding(upstream_canonical_decoder_baseline_receipt_hash=A,upstream_decoder_candidate_manifest_hash=B,upstream_decoder_replay_equivalence_receipt_hash=C,upstream_decoder_optimization_contract_hash=D,upstream_decoder_fast_path_equivalence_receipt_hash=E,candidate_declaration_hash=F,fast_path_identity_hash=G)
i=ib.build_decoder_implementation_identity(associated_candidate_declaration_hash=F,associated_fast_path_identity_hash=G,associated_fast_path_equivalence_receipt_hash=E)
a1=ib.build_decoder_implementation_artifact(artifact_id='a1',artifact_path='implementation_boundaries/a.json',artifact_sha256=A,artifact_schema_hash=B)
a2=ib.build_decoder_implementation_artifact(artifact_id='a2',artifact_path='implementation_boundaries/b.json',artifact_sha256=C,artifact_schema_hash=D)
s=ib.build_decoder_implementation_source_boundary((a2,a1)); r=ib.build_decoder_implementation_runtime_boundary(); c=ib.build_decoder_implementation_config_boundary(config_schema_hash=A,config_payload_hash=B); b=ib.build_decoder_implementation_build_boundary(source_boundary_hash=s.decoder_implementation_source_boundary_hash,build_manifest_hash=C,dependency_manifest_hash=D); e=ib.build_decoder_implementation_equivalence_binding(required_replay_equivalence_receipt_hash=C,required_fast_path_equivalence_receipt_hash=E,required_optimization_contract_hash=D); au=ib.build_decoder_implementation_audit_boundary(); ro=ib.build_decoder_implementation_rollback_gate(); at=ib.build_decoder_implementation_authority_boundary()
print(ib.build_decoder_implementation_boundary_receipt(u,i,s,r,c,b,e,au,ro,at).decoder_implementation_boundary_receipt_hash)
"""
    env = os.environ.copy(); env["PYTHONPATH"] = "src"
    outs = []
    for seed in ("0", "1"):
        e = env.copy(); e["PYTHONHASHSEED"] = seed
        outs.append(subprocess.check_output([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], env=e, text=True).strip())
    assert outs[0] == outs[1]
    assert len(outs[0]) == 64
