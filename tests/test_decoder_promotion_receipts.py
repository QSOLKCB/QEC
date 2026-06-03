from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qec.analysis.decoder_promotion_receipts import (
    DecoderPromotionAuthorityBoundary,
    DecoderPromotionError,
    DecoderPromotionErrorCode,
    DecoderPromotionReceipt,
    build_decoder_promotion_audit_boundary,
    build_decoder_promotion_authority_boundary,
    build_decoder_promotion_decision,
    build_decoder_promotion_eligibility_gate,
    build_decoder_promotion_receipt,
    build_decoder_promotion_rollback_binding,
    build_decoder_promotion_runtime_boundary,
    build_decoder_promotion_scope,
    build_decoder_promotion_target,
    build_decoder_promotion_upstream_binding,
    validate_decoder_promotion_audit_boundary,
    validate_decoder_promotion_authority_boundary,
    validate_decoder_promotion_decision,
    validate_decoder_promotion_eligibility_gate,
    validate_decoder_promotion_receipt,
    validate_decoder_promotion_rollback_binding,
    validate_decoder_promotion_runtime_boundary,
    validate_decoder_promotion_scope,
    validate_decoder_promotion_target,
    validate_decoder_promotion_upstream_binding,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/qec/analysis/decoder_promotion_receipts.py"
TEST_FILE = ROOT / "tests/test_decoder_promotion_receipts.py"
H = {c: c * 64 for c in "abcdef0123456789"}
UPSTREAM_KEYS = (
    "upstream_canonical_decoder_baseline_receipt_hash",
    "upstream_decoder_candidate_manifest_hash",
    "upstream_decoder_replay_equivalence_receipt_hash",
    "upstream_decoder_optimization_contract_hash",
    "upstream_decoder_fast_path_equivalence_receipt_hash",
    "upstream_decoder_implementation_boundary_receipt_hash",
    "upstream_decoder_benchmark_ladder_receipt_hash",
    "upstream_decoder_rollback_receipt_hash",
)
KIND_MAP = {
    "CANONICAL_BASELINE_REFERENCE_TARGET": ("PRESERVE_CANONICAL_BASELINE_REFERENCE", "CANONICAL_BASELINE_REFERENCE", "BASELINE_REFERENCE_PRESERVED"),
    "CANDIDATE_DECLARATION_PROMOTION_TARGET": ("PROMOTE_CANDIDATE_IN_RECEIPT_CHAIN", "ADAPTER_ONLY_CANDIDATE", "PROMOTED_CANDIDATE_RECEIPT_BOUNDARY"),
    "FAST_PATH_IDENTITY_PROMOTION_TARGET": ("PROMOTE_FAST_PATH_IN_RECEIPT_CHAIN", "FAST_PATH_TRANSCRIPT_ONLY", "PROMOTED_FAST_PATH_RECEIPT_BOUNDARY"),
    "IMPLEMENTATION_BOUNDARY_PROMOTION_TARGET": ("PROMOTE_IMPLEMENTATION_BOUNDARY_IN_RECEIPT_CHAIN", "IMPLEMENTATION_BOUNDARY_ONLY", "PROMOTED_IMPLEMENTATION_RECEIPT_BOUNDARY"),
    "BENCHMARK_LADDER_REFERENCE_TARGET": ("PRESERVE_BENCHMARK_LADDER_REFERENCE", "BENCHMARK_LADDER_ONLY", "BENCHMARK_LADDER_REFERENCE_PRESERVED"),
    "ROLLBACK_RECEIPT_REFERENCE_TARGET": ("PRESERVE_ROLLBACK_REFERENCE", "ROLLBACK_READY", "ROLLBACK_REFERENCE_PRESERVED"),
}


def assert_code(exc, code):
    assert exc.value.code == code
    assert code.value in str(exc.value)
    assert exc.value.detail


def make_upstream(**overrides):
    payload = dict(
        upstream_canonical_decoder_baseline_receipt_hash=H["a"],
        upstream_decoder_candidate_manifest_hash=H["b"],
        upstream_decoder_replay_equivalence_receipt_hash=H["c"],
        upstream_decoder_optimization_contract_hash=H["d"],
        upstream_decoder_fast_path_equivalence_receipt_hash=H["e"],
        upstream_decoder_implementation_boundary_receipt_hash=H["f"],
        upstream_decoder_benchmark_ladder_receipt_hash=H["0"],
        upstream_decoder_rollback_receipt_hash=H["1"],
        candidate_declaration_hash=H["2"],
        fast_path_identity_hash=H["3"],
        implementation_identity_hash=H["4"],
        candidate_name="adapter-candidate",
        candidate_version="1.0",
    )
    payload.update(overrides)
    return build_decoder_promotion_upstream_binding(**payload)


def make_gate(upstream=None, **overrides):
    upstream = upstream or make_upstream()
    payload = dict(
        gate_id="gate",
        gate_version="v1",
        required_receipt_hashes=tuple(getattr(upstream, k) for k in reversed(UPSTREAM_KEYS)),
    )
    payload.update(overrides)
    return build_decoder_promotion_eligibility_gate(**payload)


def make_target(default_target_id, default_target_kind, default_target_hash, **overrides):
    kind_for_mapping = overrides.get("target_kind", default_target_kind)
    role, pre, post = KIND_MAP.get(kind_for_mapping, KIND_MAP[default_target_kind])
    payload = dict(
        target_id=default_target_id,
        target_kind=default_target_kind,
        target_hash=default_target_hash,
        target_role=role,
        pre_promotion_status=pre,
        post_promotion_status=post,
    )
    payload.update(overrides)
    return build_decoder_promotion_target(**payload)


def make_targets(upstream=None):
    u = upstream or make_upstream()
    return (
        make_target("candidate", "CANDIDATE_DECLARATION_PROMOTION_TARGET", u.candidate_declaration_hash),
        make_target("baseline", "CANONICAL_BASELINE_REFERENCE_TARGET", u.upstream_canonical_decoder_baseline_receipt_hash),
        make_target("fast", "FAST_PATH_IDENTITY_PROMOTION_TARGET", u.fast_path_identity_hash),
        make_target("implementation", "IMPLEMENTATION_BOUNDARY_PROMOTION_TARGET", u.implementation_identity_hash),
        make_target("benchmark", "BENCHMARK_LADDER_REFERENCE_TARGET", u.upstream_decoder_benchmark_ladder_receipt_hash),
        make_target("rollback", "ROLLBACK_RECEIPT_REFERENCE_TARGET", u.upstream_decoder_rollback_receipt_hash),
    )


def make_decision(**overrides):
    payload = dict(
        decision_id="decision",
        decision_version="v1",
        promotion_reason="deterministic receipt chain gates are satisfied for declared corpus and baseline reference is preserved",
    )
    payload.update(overrides)
    return build_decoder_promotion_decision(**payload)


def make_scope(**overrides):
    payload = dict(scope_id="scope")
    payload.update(overrides)
    return build_decoder_promotion_scope(**payload)


def make_runtime(**overrides):
    payload = dict(runtime_boundary_id="runtime")
    payload.update(overrides)
    return build_decoder_promotion_runtime_boundary(**payload)


def make_rollback(upstream=None, **overrides):
    upstream = upstream or make_upstream()
    payload = dict(rollback_binding_id="rollback", required_rollback_receipt_hash=upstream.upstream_decoder_rollback_receipt_hash)
    payload.update(overrides)
    return build_decoder_promotion_rollback_binding(**payload)


def make_audit(**overrides):
    payload = dict(audit_boundary_id="audit")
    payload.update(overrides)
    return build_decoder_promotion_audit_boundary(**payload)


def make_authority(**overrides):
    payload = dict(authority_boundary_id="authority")
    payload.update(overrides)
    return build_decoder_promotion_authority_boundary(**payload)


def make_receipt(**overrides):
    upstream = overrides.pop("upstream_binding", make_upstream())
    payload = dict(
        upstream_binding=upstream,
        eligibility_gate=overrides.pop("eligibility_gate", make_gate(upstream)),
        promotion_targets=overrides.pop("promotion_targets", make_targets(upstream)),
        promotion_decision=overrides.pop("promotion_decision", make_decision()),
        promotion_scope=overrides.pop("promotion_scope", make_scope()),
        runtime_boundary=overrides.pop("runtime_boundary", make_runtime()),
        rollback_binding=overrides.pop("rollback_binding", make_rollback(upstream)),
        audit_boundary=overrides.pop("audit_boundary", make_audit()),
        authority_boundary=overrides.pop("authority_boundary", make_authority()),
    )
    payload.update(overrides)
    return build_decoder_promotion_receipt(**payload)


def test_happy_path_builds_validates_and_is_immutable():
    receipt = make_receipt()
    validate_decoder_promotion_upstream_binding(receipt.upstream_binding)
    validate_decoder_promotion_eligibility_gate(receipt.eligibility_gate)
    for target in receipt.promotion_targets:
        validate_decoder_promotion_target(target)
    validate_decoder_promotion_decision(receipt.promotion_decision)
    validate_decoder_promotion_scope(receipt.promotion_scope)
    validate_decoder_promotion_runtime_boundary(receipt.runtime_boundary)
    validate_decoder_promotion_rollback_binding(receipt.rollback_binding)
    validate_decoder_promotion_audit_boundary(receipt.audit_boundary)
    validate_decoder_promotion_authority_boundary(receipt.authority_boundary)
    assert validate_decoder_promotion_receipt(receipt) is receipt
    assert len(receipt.decoder_promotion_receipt_hash) == 64
    int(receipt.decoder_promotion_receipt_hash, 16)
    assert receipt.promotion_receipt_safe is True
    assert receipt.all_required_gates_satisfied is True
    assert receipt.candidate_promoted_by_receipt is True
    assert receipt.fast_path_promoted_by_receipt is True
    assert receipt.implementation_boundary_promoted_by_receipt is True
    assert receipt.canonical_baseline_reference_preserved is True
    assert receipt.source_replacement_performed is False
    assert receipt.runtime_activation_performed is False
    assert receipt.promotion_execution_performed_by_receipt is False
    assert receipt.global_correctness_claim_allowed is False
    assert receipt.qec_advantage_claim_allowed is False
    assert receipt.hardware_authority_allowed is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.receipt_kind = "mutated"


def test_canonical_hash_determinism_ordering_and_no_builtin_hash_identity():
    upstream = make_upstream()
    receipt_a = make_receipt(upstream_binding=upstream, promotion_targets=tuple(reversed(make_targets(upstream))))
    receipt_b = make_receipt(upstream_binding=upstream, promotion_targets=make_targets(upstream))
    assert receipt_a.decoder_promotion_receipt_hash == receipt_b.decoder_promotion_receipt_hash
    gate_a = make_gate(upstream, required_receipt_hashes=tuple(reversed([getattr(upstream, k) for k in UPSTREAM_KEYS])))
    gate_b = make_gate(upstream, required_receipt_hashes=tuple(getattr(upstream, k) for k in UPSTREAM_KEYS))
    assert gate_a.decoder_promotion_eligibility_gate_hash == gate_b.decoder_promotion_eligibility_gate_hash
    kinds = (
        "DecoderRollbackReceipt", "DecoderReplayEquivalenceReceipt", "DecoderOptimizationContract",
        "DecoderImplementationBoundaryReceipt", "DecoderFastPathEquivalenceReceipt", "DecoderCandidateManifest",
        "DecoderBenchmarkLadderReceipt", "CanonicalDecoderBaselineReceipt",
    )
    assert make_gate(upstream, required_receipt_kinds=kinds).decoder_promotion_eligibility_gate_hash == gate_b.decoder_promotion_eligibility_gate_hash
    assert make_receipt().decoder_promotion_receipt_hash == make_receipt().decoder_promotion_receipt_hash
    source = MODULE.read_text()
    assert "builtins.hash" not in source
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hash" for n in ast.walk(tree))


def test_self_hash_exclusion_and_child_hash_staleness():
    receipt = make_receipt()
    object.__setattr__(receipt, "decoder_promotion_receipt_hash", H["9"])
    with pytest.raises(DecoderPromotionError) as exc:
        validate_decoder_promotion_receipt(receipt)
    assert_code(exc, DecoderPromotionErrorCode.HASH_MISMATCH)

    scope = make_scope()
    object.__setattr__(scope, "scope_hash", H["8"])
    with pytest.raises(DecoderPromotionError) as exc2:
        validate_decoder_promotion_scope(scope)
    assert_code(exc2, DecoderPromotionErrorCode.HASH_MISMATCH)

    target = make_targets()[0]
    object.__setattr__(target, "decoder_promotion_target_hash", H["7"])
    with pytest.raises(DecoderPromotionError) as exc3:
        validate_decoder_promotion_target(target)
    assert_code(exc3, DecoderPromotionErrorCode.HASH_MISMATCH)


def test_child_before_aggregate_validation_rejects_corrupt_children():
    gate = make_gate()
    object.__setattr__(gate, "gate_authority_allowed", True)
    with pytest.raises(DecoderPromotionError) as exc:
        make_receipt(eligibility_gate=gate)
    assert exc.value.code in {DecoderPromotionErrorCode.INVALID_INPUT, DecoderPromotionErrorCode.HASH_MISMATCH}

    target = make_targets()[0]
    object.__setattr__(target, "source_replacement_allowed", True)
    with pytest.raises(DecoderPromotionError):
        make_receipt(promotion_targets=(target, *make_targets()[1:]))

    authority = make_authority()
    object.__setattr__(authority, "hardware_authority_allowed", True)
    with pytest.raises(DecoderPromotionError):
        make_receipt(authority_boundary=authority)


@pytest.mark.parametrize("field,bad,code", [
    ("previous_release_tag", "v166.6", DecoderPromotionErrorCode.INVALID_INPUT),
    ("previous_release_url", "https://example.test/v166.7", DecoderPromotionErrorCode.INVALID_INPUT),
    ("promotion_release", "v166.9", DecoderPromotionErrorCode.INVALID_INPUT),
    ("upstream_canonical_decoder_baseline_receipt_hash", "A" * 64, DecoderPromotionErrorCode.INVALID_HASH),
    ("upstream_decoder_candidate_manifest_hash", "short", DecoderPromotionErrorCode.INVALID_HASH),
    ("upstream_decoder_replay_equivalence_receipt_hash", "g" * 64, DecoderPromotionErrorCode.INVALID_HASH),
    ("candidate_declaration_hash", "", DecoderPromotionErrorCode.INVALID_HASH),
    ("fast_path_identity_hash", "bad", DecoderPromotionErrorCode.INVALID_HASH),
    ("implementation_identity_hash", "G" * 64, DecoderPromotionErrorCode.INVALID_HASH),
    ("candidate_name", "", DecoderPromotionErrorCode.INVALID_INPUT),
    ("candidate_version", "", DecoderPromotionErrorCode.INVALID_INPUT),
    ("replay_equivalence_proven_for_declared_corpus", False, DecoderPromotionErrorCode.INVALID_INPUT),
    ("optimization_contract_safe", False, DecoderPromotionErrorCode.INVALID_INPUT),
    ("fast_path_equivalence_proven_for_declared_corpus", False, DecoderPromotionErrorCode.INVALID_INPUT),
    ("implementation_boundary_safe", False, DecoderPromotionErrorCode.INVALID_INPUT),
    ("benchmark_ladder_safe", False, DecoderPromotionErrorCode.INVALID_INPUT),
    ("rollback_receipt_safe", False, DecoderPromotionErrorCode.INVALID_INPUT),
    ("rollback_ready_for_future_promotion_gate", False, DecoderPromotionErrorCode.INVALID_INPUT),
    ("candidate_adapter_only_before_promotion", False, DecoderPromotionErrorCode.INVALID_INPUT),
    ("candidate_promoted_before_receipt", True, DecoderPromotionErrorCode.INVALID_INPUT),
    ("baseline_immutable", False, DecoderPromotionErrorCode.INVALID_INPUT),
    ("baseline_mutation_allowed", True, DecoderPromotionErrorCode.INVALID_INPUT),
    ("runtime_authority_allowed", True, DecoderPromotionErrorCode.INVALID_INPUT),
    ("candidate_adapter_only_before_promotion", 1, DecoderPromotionErrorCode.INVALID_INPUT),
])
def test_upstream_binding_validation(field, bad, code):
    with pytest.raises(DecoderPromotionError) as exc:
        make_upstream(**{field: bad})
    assert_code(exc, code)


@pytest.mark.parametrize("overrides", [
    {"gate_id": ""}, {"gate_version": ""}, {"gate_mode": "OTHER"},
    {"required_receipt_hashes": ()}, {"required_receipt_hashes": (H["a"], H["a"])}, {"required_receipt_hashes": ("bad",)},
    {"required_receipt_kinds": ("CanonicalDecoderBaselineReceipt",)},
    {"required_receipt_kinds": ("CanonicalDecoderBaselineReceipt", "CanonicalDecoderBaselineReceipt")},
    {"required_receipt_kinds": ("UNKNOWN",) + tuple(k for k in ("DecoderBenchmarkLadderReceipt", "DecoderCandidateManifest", "DecoderFastPathEquivalenceReceipt", "DecoderImplementationBoundaryReceipt", "DecoderOptimizationContract", "DecoderReplayEquivalenceReceipt", "DecoderRollbackReceipt"))},
    {"required_receipt_count": 7}, {"required_receipt_count": True},
    {"canonical_baseline_required": False}, {"candidate_manifest_required": False}, {"replay_equivalence_required": False},
    {"optimization_contract_required": False}, {"fast_path_equivalence_required": False}, {"implementation_boundary_required": False},
    {"benchmark_ladder_required": False}, {"rollback_receipt_required": False}, {"all_required_gates_satisfied": False},
    {"gate_authority_allowed": True},
])
def test_eligibility_gate_validation(overrides):
    with pytest.raises(DecoderPromotionError):
        make_gate(**overrides)


@pytest.mark.parametrize("overrides", [
    {"target_id": ""}, {"target_kind": "UNKNOWN"}, {"target_hash": "bad"}, {"target_role": "UNKNOWN"},
    {"pre_promotion_status": "UNKNOWN"}, {"post_promotion_status": "UNKNOWN"},
    {"target_role": "PRESERVE_ROLLBACK_REFERENCE"}, {"pre_promotion_status": "ROLLBACK_READY"}, {"post_promotion_status": "ROLLBACK_REFERENCE_PRESERVED"},
    {"promotion_target_declared": False}, {"source_replacement_allowed": True}, {"baseline_source_mutation_allowed": True},
    {"runtime_activation_allowed_by_receipt": True}, {"rollback_protection_required": False}, {"promotion_target_authority_allowed": True},
])
def test_promotion_target_validation(overrides):
    with pytest.raises(DecoderPromotionError):
        make_target("candidate", "CANDIDATE_DECLARATION_PROMOTION_TARGET", H["2"], **overrides)


@pytest.mark.parametrize("overrides", [
    {"decision_id": ""}, {"decision_version": ""}, {"decision_mode": "OTHER"}, {"decision_status": "OTHER"},
    {"promotion_declared": False}, {"candidate_promoted_by_receipt": False}, {"fast_path_promoted_by_receipt": False},
    {"implementation_boundary_promoted_by_receipt": False}, {"canonical_baseline_reference_preserved": False},
    {"source_replacement_performed": True}, {"runtime_activation_performed": True}, {"promotion_execution_performed_by_receipt": True},
    {"promotion_reason": ""}, {"decision_authority_scope": "RUNTIME"}, {"promotion_reason": "speed proves correctness"},
])
def test_promotion_decision_validation(overrides):
    with pytest.raises(DecoderPromotionError):
        make_decision(**overrides)


@pytest.mark.parametrize("overrides", [
    {"scope_id": ""}, {"scope_mode": "OTHER"}, {"declared_promotion_scope": "GLOBAL"},
    {"promotion_scope_limited_to_receipt_chain": False}, {"declared_corpus_replay_scope_preserved": False},
    {"benchmark_scope_preserved": False}, {"rollback_scope_preserved": False}, {"global_correctness_claim_allowed": True},
    {"universal_speedup_claim_allowed": True}, {"qec_advantage_claim_allowed": True}, {"hardware_authority_allowed": True},
    {"benchmark_marketing_allowed": True}, {"scope_hash": H["9"]},
])
def test_promotion_scope_validation(overrides):
    with pytest.raises(DecoderPromotionError):
        make_scope(**overrides)


@pytest.mark.parametrize("field", [
    "decoder_import_allowed", "candidate_import_allowed", "fast_path_import_allowed", "implementation_import_allowed",
    "runtime_decoder_execution_allowed", "runtime_activation_allowed", "promotion_runtime_execution_allowed",
    "benchmark_execution_allowed", "rollback_execution_allowed", "filesystem_mutation_allowed", "source_replacement_allowed",
    "git_operation_allowed", "subprocess_promotion_allowed", "network_allowed", "heavy_backend_import_allowed", "hardware_sdk_allowed",
])
def test_runtime_boundary_rejects_forbidden_allowances(field):
    with pytest.raises(DecoderPromotionError):
        make_runtime(**{field: True})


@pytest.mark.parametrize("overrides", [{"runtime_boundary_id": ""}, {"runtime_boundary_mode": "OTHER"}, {"promotion_receipt_only": False}, {"promotion_receipt_only": 1}])
def test_runtime_boundary_core_validation(overrides):
    with pytest.raises(DecoderPromotionError):
        make_runtime(**overrides)


@pytest.mark.parametrize("overrides", [
    {"rollback_binding_id": ""}, {"rollback_binding_mode": "OTHER"}, {"required_rollback_receipt_hash": "bad"},
    {"rollback_receipt_safe": False}, {"rollback_ready_for_future_promotion_gate": False}, {"rollback_reference_preserved": False},
    {"rollback_execution_required_for_promotion_receipt": True}, {"rollback_execution_performed_by_promotion_receipt": True},
    {"rollback_path_deletion_allowed": True}, {"rollback_bypass_allowed": True}, {"baseline_restore_path_preserved": False},
    {"candidate_disable_path_preserved": False},
])
def test_rollback_binding_validation(overrides):
    with pytest.raises(DecoderPromotionError):
        make_rollback(**overrides)


@pytest.mark.parametrize("field", [
    "upstream_receipts_review_required", "eligibility_gate_review_required", "target_binding_review_required", "decision_review_required",
    "scope_review_required", "runtime_boundary_review_required", "rollback_binding_review_required", "no_source_replacement_review_required",
    "no_runtime_activation_review_required", "no_benchmark_marketing_review_required", "no_global_correctness_review_required",
    "audit_complete_for_promotion_receipt",
])
def test_audit_boundary_review_validation(field):
    with pytest.raises(DecoderPromotionError):
        make_audit(**{field: False})


@pytest.mark.parametrize("overrides", [{"audit_boundary_id": ""}, {"audit_mode": "OTHER"}, {"audit_authority_allowed": True}])
def test_audit_boundary_core_validation(overrides):
    with pytest.raises(DecoderPromotionError):
        make_audit(**overrides)


@pytest.mark.parametrize("field", [
    "promotion_receipt_authority_allowed", "promotion_execution_authority_allowed", "runtime_authority_allowed",
    "implementation_authority_allowed", "benchmark_authority_allowed", "rollback_execution_authority_allowed",
    "hardware_authority_allowed", "ml_decoder_authority_allowed", "probabilistic_decoder_authority_allowed",
    "qec_advantage_claim_allowed", "global_correctness_claim_allowed", "universal_speedup_claim_allowed",
    "benchmark_marketing_allowed", "silent_replacement_allowed", "baseline_mutation_allowed", "source_replacement_allowed",
])
def test_authority_boundary_validation(field):
    with pytest.raises(DecoderPromotionError):
        make_authority(**{field: True})


@pytest.mark.parametrize("overrides", [{"authority_boundary_id": ""}, {"authority_mode": "OTHER"}, {"hardware_authority_allowed": 1}])
def test_authority_boundary_core_validation(overrides):
    with pytest.raises(DecoderPromotionError):
        make_authority(**overrides)


@pytest.mark.parametrize("overrides", [
    {"receipt_version": "v166.9"}, {"receipt_kind": "Other"}, {"previous_release_tag": "v166.6"},
    {"previous_release_url": "https://example.test"}, {"promotion_targets": ()}, {"promotion_target_count": 5},
    {"promotion_target_count": True}, {"source_replacement_performed": True}, {"runtime_activation_performed": True},
    {"promotion_execution_performed_by_receipt": True}, {"global_correctness_claim_allowed": True},
    {"qec_advantage_claim_allowed": True}, {"hardware_authority_allowed": True},
])
def test_aggregate_receipt_validation_simple_overrides(overrides):
    with pytest.raises(DecoderPromotionError):
        make_receipt(**overrides)


def test_aggregate_receipt_rejects_non_child_duplicate_missing_mismatch_and_forged_flags():
    with pytest.raises(DecoderPromotionError):
        make_receipt(promotion_targets=(object(),))

    targets = list(make_targets())
    targets[1] = make_target("candidate", "CANONICAL_BASELINE_REFERENCE_TARGET", H["a"])
    with pytest.raises(DecoderPromotionError):
        make_receipt(promotion_targets=tuple(targets))

    targets = list(make_targets())[:-1]
    with pytest.raises(DecoderPromotionError):
        make_receipt(promotion_targets=tuple(targets))

    upstream = make_upstream()
    targets = list(make_targets(upstream))
    targets[0] = make_target("candidate", "CANDIDATE_DECLARATION_PROMOTION_TARGET", H["5"])
    with pytest.raises(DecoderPromotionError):
        make_receipt(upstream_binding=upstream, promotion_targets=tuple(targets))

    gate = make_gate(upstream, required_receipt_hashes=(H["a"], H["b"], H["c"], H["d"], H["e"], H["f"], H["0"], H["5"]))
    with pytest.raises(DecoderPromotionError):
        make_receipt(upstream_binding=upstream, eligibility_gate=gate)

    with pytest.raises(DecoderPromotionError):
        make_receipt(upstream_binding=upstream, rollback_binding=make_rollback(upstream, required_rollback_receipt_hash=H["5"]))

    decision = make_decision()
    object.__setattr__(decision, "candidate_promoted_by_receipt", False)
    with pytest.raises(DecoderPromotionError):
        make_receipt(promotion_decision=decision, candidate_promoted_by_receipt=True)


@pytest.mark.parametrize("phrase", [
    "silent_decoder_replacement", "candidate-replaces-baseline", "decoder replaced because faster", "speed proves correctness",
    "benchmark proves correctness", "benchmark marketing", "runtime promotion", "runtime activation", "candidate decoder authority",
    "probabilistic decoder authority", "ML decoder authority", "hardware authority", "QEC advantage proven", "hidden precision drift",
    "undeclared approximation policy", "global correctness proven", "replay equivalence implies promotion", "replay equivalence implies speedup",
    "optimization implies correctness", "fast path accepted because faster", "fast path runtime enabled", "benchmark proves decoder correctness",
    "benchmark replaces replay equivalence", "benchmark replaces rollback", "implementation enabled", "implementation proves correctness",
    "implementation replaces baseline", "rollback permits source mutation", "promotion without receipt", "promotion mutates source",
    "promotion executes decoder", "promotion performs git operation", "promotion replaces baseline source", "promotion proves correctness",
    "promotion proves QEC advantage", "source replacement allowed", "baseline mutation allowed", "speed\n\tproves\r correctness",
])
def test_forbidden_semantic_hardening_blocks_variants(phrase):
    with pytest.raises(DecoderPromotionError) as exc:
        make_decision(promotion_reason=phrase)
    assert_code(exc, DecoderPromotionErrorCode.INVALID_INPUT)


@pytest.mark.parametrize("phrase", [
    "candidate_promoted_by_receipt", "fast_path_promoted_by_receipt", "implementation_boundary_promoted_by_receipt",
    "promotion_receipt_safe", "promotion_declared", "PROMOTED_BY_RECEIPT_CHAIN", "PROMOTED_CANDIDATE_RECEIPT_BOUNDARY",
])
def test_positive_semantic_controls_are_allowed(phrase):
    # Directly exercises the scoped declaration guard through text-bearing fields.
    assert make_upstream(candidate_name=phrase).candidate_name == phrase


def _imports_from(path: Path):
    tree = ast.parse(path.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return imports


def test_boundary_imports_and_static_markers():
    imports = _imports_from(MODULE)
    test_imports = _imports_from(TEST_FILE)
    banned_prefixes = (
        "qec.decoder", "numpy", "scipy", "qldpc", "stim", "pymatching", "qiskit", "qutip",
        "pandas", "polars", "torch", "tensorflow", "jax", "requests", "urllib", "socket", "importlib",
    )
    for name in imports:
        assert not any(name == prefix or name.startswith(prefix + ".") for prefix in banned_prefixes)
    for name in test_imports:
        assert not (name == "qec.decoder" or name.startswith("qec.decoder."))
    source = MODULE.read_text().lower()
    for marker in (
        "time.perf_counter", "time.time", "datetime.now", "decoder workload", "benchmark loop", "rollback execution",
        "promotion execution", "fast_path_runtime", "candidate_decoder", "importlib.import_module",
    ):
        assert marker not in source
    assert "import subprocess" not in source
    assert "network_allowed: bool" in MODULE.read_text()


def test_no_forbidden_runtime_files_created_and_decoder_not_modified_in_worktree():
    forbidden = [
        ROOT / "src/qec/analysis/decoder_promotion_runtime.py",
        ROOT / "src/qec/analysis/decoder_fast_path_runtime.py",
        ROOT / "src/qec/analysis/decoder_benchmark_runtime.py",
        ROOT / "src/qec/analysis/decoder_rollback_runtime.py",
        ROOT / "src/qec/analysis/candidate_decoder.py",
        ROOT / "src/qec/analysis/decoder_candidate_implementation.py",
    ]
    assert all(not p.exists() for p in forbidden)
    diff = subprocess.run(["git", "diff", "--name-only", "--", "src/qec/decoder/"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    assert diff.stdout.strip() == ""


def test_hash_seed_stability_subprocesses():
    code = r'''
from qec.analysis.decoder_promotion_receipts import *
H={c:c*64 for c in 'abcdef01234'}
up=build_decoder_promotion_upstream_binding(upstream_canonical_decoder_baseline_receipt_hash=H['a'],upstream_decoder_candidate_manifest_hash=H['b'],upstream_decoder_replay_equivalence_receipt_hash=H['c'],upstream_decoder_optimization_contract_hash=H['d'],upstream_decoder_fast_path_equivalence_receipt_hash=H['e'],upstream_decoder_implementation_boundary_receipt_hash=H['f'],upstream_decoder_benchmark_ladder_receipt_hash=H['0'],upstream_decoder_rollback_receipt_hash=H['1'],candidate_declaration_hash=H['2'],fast_path_identity_hash=H['3'],implementation_identity_hash=H['4'],candidate_name='candidate',candidate_version='1')
gate=build_decoder_promotion_eligibility_gate(gate_id='gate',gate_version='1',required_receipt_hashes=(H['1'],H['0'],H['f'],H['e'],H['d'],H['c'],H['b'],H['a']))
def t(i,k,h):
 m={'CANONICAL_BASELINE_REFERENCE_TARGET':('PRESERVE_CANONICAL_BASELINE_REFERENCE','CANONICAL_BASELINE_REFERENCE','BASELINE_REFERENCE_PRESERVED'),'CANDIDATE_DECLARATION_PROMOTION_TARGET':('PROMOTE_CANDIDATE_IN_RECEIPT_CHAIN','ADAPTER_ONLY_CANDIDATE','PROMOTED_CANDIDATE_RECEIPT_BOUNDARY'),'FAST_PATH_IDENTITY_PROMOTION_TARGET':('PROMOTE_FAST_PATH_IN_RECEIPT_CHAIN','FAST_PATH_TRANSCRIPT_ONLY','PROMOTED_FAST_PATH_RECEIPT_BOUNDARY'),'IMPLEMENTATION_BOUNDARY_PROMOTION_TARGET':('PROMOTE_IMPLEMENTATION_BOUNDARY_IN_RECEIPT_CHAIN','IMPLEMENTATION_BOUNDARY_ONLY','PROMOTED_IMPLEMENTATION_RECEIPT_BOUNDARY'),'BENCHMARK_LADDER_REFERENCE_TARGET':('PRESERVE_BENCHMARK_LADDER_REFERENCE','BENCHMARK_LADDER_ONLY','BENCHMARK_LADDER_REFERENCE_PRESERVED'),'ROLLBACK_RECEIPT_REFERENCE_TARGET':('PRESERVE_ROLLBACK_REFERENCE','ROLLBACK_READY','ROLLBACK_REFERENCE_PRESERVED')}; r,p,po=m[k]; return build_decoder_promotion_target(target_id=i,target_kind=k,target_hash=h,target_role=r,pre_promotion_status=p,post_promotion_status=po)
rec=build_decoder_promotion_receipt(upstream_binding=up,eligibility_gate=gate,promotion_targets=(t('r','ROLLBACK_RECEIPT_REFERENCE_TARGET',H['1']),t('b','CANONICAL_BASELINE_REFERENCE_TARGET',H['a']),t('c','CANDIDATE_DECLARATION_PROMOTION_TARGET',H['2']),t('f','FAST_PATH_IDENTITY_PROMOTION_TARGET',H['3']),t('i','IMPLEMENTATION_BOUNDARY_PROMOTION_TARGET',H['4']),t('l','BENCHMARK_LADDER_REFERENCE_TARGET',H['0'])),promotion_decision=build_decoder_promotion_decision(decision_id='d',decision_version='1',promotion_reason='deterministic receipt chain gates are satisfied for declared corpus and baseline reference is preserved'),promotion_scope=build_decoder_promotion_scope(scope_id='s'),runtime_boundary=build_decoder_promotion_runtime_boundary(runtime_boundary_id='rt'),rollback_binding=build_decoder_promotion_rollback_binding(rollback_binding_id='rb',required_rollback_receipt_hash=H['1']),audit_boundary=build_decoder_promotion_audit_boundary(audit_boundary_id='aud'),authority_boundary=build_decoder_promotion_authority_boundary(authority_boundary_id='auth'))
print(rec.decoder_promotion_receipt_hash)
'''
    outputs = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONPATH="src", PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=env)
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]
