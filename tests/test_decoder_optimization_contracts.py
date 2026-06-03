from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from qec.analysis import decoder_optimization_contracts as doc

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64
HEX_F = "f" * 64


def _unsafe_copy(obj, **changes):
    clone = object.__new__(type(obj))
    for field in doc.fields(obj):
        object.__setattr__(clone, field.name, changes.get(field.name, getattr(obj, field.name)))
    return clone


def _rehash(obj, payload_fn_name: str, hash_field: str, **changes):
    clone = _unsafe_copy(obj, **changes)
    object.__setattr__(clone, hash_field, doc._hash_payload(getattr(doc, payload_fn_name)(clone)))
    return clone


def _expect_error(code: str, fn, *args, **kwargs):
    with pytest.raises(doc.DecoderOptimizationContractError) as exc:
        fn(*args, **kwargs)
    assert exc.value.code.value == code
    assert code in str(exc.value)
    assert exc.value.detail
    return exc.value


def _parts():
    upstream = doc.build_decoder_optimization_upstream_binding(
        upstream_canonical_decoder_baseline_receipt_hash=HEX_A,
        upstream_decoder_candidate_manifest_hash=HEX_B,
        upstream_decoder_replay_equivalence_receipt_hash=HEX_C,
        candidate_declaration_hash=HEX_D,
        candidate_name="adapter candidate",
        candidate_version="1",
    )
    inv1 = doc.build_decoder_optimization_invariant_source(
        invariant_id="inv-a",
        invariant_kind="SPARSE_SYNDROME_STRUCTURE_INVARIANT",
        source_receipt_hash=HEX_A,
        replay_equivalence_receipt_hash=HEX_C,
        declared_input_scope_hash=HEX_D,
        declared_output_scope_hash=HEX_E,
        optimization_relevance="MAY_SUPPORT_SPARSE_HANDLING",
    )
    inv2 = doc.build_decoder_optimization_invariant_source(
        invariant_id="inv-b",
        invariant_kind="DETERMINISTIC_ORDERING_INVARIANT",
        source_receipt_hash=HEX_B,
        replay_equivalence_receipt_hash=HEX_C,
        declared_input_scope_hash=HEX_D,
        declared_output_scope_hash=HEX_F,
        optimization_relevance="MAY_SUPPORT_FUTURE_FAST_PATH",
    )
    target1 = doc.build_decoder_optimization_target(
        target_id="target-a",
        target_kind="SPARSE_HANDLING_TARGET",
        target_description="Declarative sparse representation precondition only",
    )
    target2 = doc.build_decoder_optimization_target(
        target_id="target-b",
        target_kind="MEMORY_EFFICIENCY_TARGET",
        target_description="Declarative memory layout precondition only",
    )
    gate = doc.build_decoder_optimization_equivalence_gate(
        gate_id="gate", required_prior_replay_equivalence_receipt_hash=HEX_C
    )
    transformation = doc.build_decoder_optimization_transformation_boundary(
        transformation_boundary_id="transform",
        allowed_transformation_kinds=(
            "DECLARED_SPARSE_REPRESENTATION_PRECONDITION",
            "DECLARED_CACHE_KEY_PRECONDITION",
        ),
    )
    precision = doc.build_decoder_optimization_precision_boundary(precision_boundary_id="precision")
    benchmark = doc.build_decoder_optimization_benchmark_boundary(benchmark_boundary_id="bench-boundary")
    rollback = doc.build_decoder_optimization_rollback_policy(
        rollback_policy_id="rollback",
        rollback_trigger_conditions=("OUTPUT_SCHEMA_DRIFT", "FAST_PATH_EQUIVALENCE_FAILURE"),
    )
    authority = doc.build_decoder_optimization_authority_boundary(authority_boundary_id="authority")
    contract = doc.build_decoder_optimization_contract(
        upstream_binding=upstream,
        invariant_sources=(inv2, inv1),
        optimization_targets=(target2, target1),
        equivalence_gate=gate,
        transformation_boundary=transformation,
        precision_boundary=precision,
        benchmark_boundary=benchmark,
        rollback_policy=rollback,
        authority_boundary=authority,
    )
    return upstream, (inv1, inv2), (target1, target2), gate, transformation, precision, benchmark, rollback, authority, contract


def test_happy_path_and_immutability():
    upstream, invariants, targets, gate, transformation, precision, benchmark, rollback, authority, contract = _parts()
    for value, validator in (
        (upstream, doc.validate_decoder_optimization_upstream_binding),
        (invariants[0], doc.validate_decoder_optimization_invariant_source),
        (targets[0], doc.validate_decoder_optimization_target),
        (gate, doc.validate_decoder_optimization_equivalence_gate),
        (transformation, doc.validate_decoder_optimization_transformation_boundary),
        (precision, doc.validate_decoder_optimization_precision_boundary),
        (benchmark, doc.validate_decoder_optimization_benchmark_boundary),
        (rollback, doc.validate_decoder_optimization_rollback_policy),
        (authority, doc.validate_decoder_optimization_authority_boundary),
        (contract, doc.validate_decoder_optimization_contract),
    ):
        assert validator(value) is value
    assert doc._HASH_RE.fullmatch(contract.decoder_optimization_contract_hash)
    assert contract.optimization_contract_safe is True
    assert contract.candidate_remains_adapter_only is True
    assert contract.fast_path_implementation_allowed is False
    assert contract.promotion_allowed is False
    assert contract.benchmark_claim_allowed is False
    assert contract.speedup_claim_allowed is False
    with pytest.raises(FrozenInstanceError):
        contract.promotion_allowed = True


def test_canonical_json_hash_determinism_and_ordering():
    upstream, invariants, targets, gate, transformation, precision, benchmark, rollback, authority, contract = _parts()
    contract2 = doc.build_decoder_optimization_contract(
        upstream_binding=upstream,
        invariant_sources=tuple(reversed(invariants)),
        optimization_targets=tuple(reversed(targets)),
        equivalence_gate=gate,
        transformation_boundary=transformation,
        precision_boundary=precision,
        benchmark_boundary=benchmark,
        rollback_policy=rollback,
        authority_boundary=authority,
    )
    contract3 = doc.build_decoder_optimization_contract(
        upstream_binding=upstream,
        invariant_sources=tuple({i.invariant_id: i for i in reversed(invariants)}.values()),
        optimization_targets=tuple({t.target_id: t for t in reversed(targets)}.values()),
        equivalence_gate=gate,
        transformation_boundary=transformation,
        precision_boundary=precision,
        benchmark_boundary=benchmark,
        rollback_policy=rollback,
        authority_boundary=authority,
    )
    assert contract.decoder_optimization_contract_hash == contract2.decoder_optimization_contract_hash == contract3.decoder_optimization_contract_hash
    t2 = doc.build_decoder_optimization_transformation_boundary(
        transformation_boundary_id="transform",
        allowed_transformation_kinds=tuple(reversed(transformation.allowed_transformation_kinds)),
    )
    r2 = doc.build_decoder_optimization_rollback_policy(
        rollback_policy_id="rollback",
        rollback_trigger_conditions=tuple(reversed(rollback.rollback_trigger_conditions)),
    )
    assert transformation.decoder_optimization_transformation_boundary_hash == t2.decoder_optimization_transformation_boundary_hash
    assert rollback.decoder_optimization_rollback_policy_hash == r2.decoder_optimization_rollback_policy_hash
    assert doc._hash_payload({"a": [1, 2]}) == doc._hash_payload({"a": (1, 2)})
    tree = ast.parse(Path("src/qec/analysis/decoder_optimization_contracts.py").read_text())
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash" for node in ast.walk(tree))


def test_self_hash_exclusion_and_stale_hashes_fail():
    upstream, _, _, _, _, _, _, rollback, _, contract = _parts()
    _expect_error("HASH_MISMATCH", doc.validate_decoder_optimization_contract, _unsafe_copy(contract, decoder_optimization_contract_hash=HEX_A))
    assert doc._hash_payload(doc._contract_payload(contract)) == contract.decoder_optimization_contract_hash
    _expect_error("HASH_MISMATCH", doc.validate_decoder_optimization_upstream_binding, _unsafe_copy(upstream, decoder_optimization_upstream_binding_hash=HEX_B))
    assert doc._hash_payload(doc._upstream_binding_payload(upstream)) == upstream.decoder_optimization_upstream_binding_hash
    _expect_error("HASH_MISMATCH", doc.validate_decoder_optimization_rollback_policy, _unsafe_copy(rollback, decoder_optimization_rollback_policy_hash=HEX_C))
    assert doc._hash_payload(doc._rollback_policy_payload(rollback)) == rollback.decoder_optimization_rollback_policy_hash


def test_child_before_aggregate_validation():
    upstream, invariants, targets, gate, transformation, precision, benchmark, rollback, authority, contract = _parts()
    bad_inv = _rehash(invariants[0], "_invariant_source_payload", "decoder_optimization_invariant_source_hash", runtime_discovery_allowed=0)
    forged = _rehash(contract, "_contract_payload", "decoder_optimization_contract_hash", invariant_sources=(bad_inv, invariants[1]))
    _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_contract, forged)
    bad_rollback = _rehash(rollback, "_rollback_policy_payload", "decoder_optimization_rollback_policy_hash", rollback_path_deletion_allowed=True)
    forged = _rehash(contract, "_contract_payload", "decoder_optimization_contract_hash", rollback_policy=bad_rollback)
    _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_contract, forged)
    bad_authority = _rehash(authority, "_authority_boundary_payload", "decoder_optimization_authority_boundary_hash", hardware_authority_allowed=True)
    forged = _rehash(contract, "_contract_payload", "decoder_optimization_contract_hash", authority_boundary=bad_authority)
    _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_contract, forged)


def test_upstream_binding_validation_rejects_malformed_and_unsafe_fields():
    good = _parts()[0]
    cases = [
        ("previous_release_tag", "v166.1", "INVALID_INPUT"),
        ("previous_release_url", "https://example.invalid", "INVALID_INPUT"),
        ("contract_release", "v166.2", "INVALID_INPUT"),
        ("upstream_canonical_decoder_baseline_receipt_hash", "A" * 64, "INVALID_HASH"),
        ("upstream_decoder_candidate_manifest_hash", "x", "INVALID_HASH"),
        ("upstream_decoder_replay_equivalence_receipt_hash", "1" * 63, "INVALID_HASH"),
        ("candidate_declaration_hash", "g" * 64, "INVALID_HASH"),
        ("candidate_name", "", "INVALID_INPUT"),
        ("candidate_version", "", "INVALID_INPUT"),
        ("replay_equivalence_proven_for_declared_corpus", False, "INVALID_INPUT"),
        ("candidate_adapter_only", False, "INVALID_INPUT"),
        ("candidate_promoted", True, "INVALID_INPUT"),
        ("baseline_immutable", False, "INVALID_INPUT"),
        ("baseline_mutation_allowed", True, "INVALID_INPUT"),
        ("candidate_runtime_authority_allowed", True, "INVALID_INPUT"),
        ("candidate_adapter_only", 1, "INVALID_INPUT"),
    ]
    for field, value, code in cases:
        _expect_error(code, doc.validate_decoder_optimization_upstream_binding, _rehash(good, "_upstream_binding_payload", "decoder_optimization_upstream_binding_hash", **{field: value}))


def test_child_validation_rejects_required_semantics():
    _, invariants, targets, gate, transformation, precision, benchmark, rollback, authority, _ = _parts()
    bad_inv_cases = [("invariant_id", ""), ("invariant_kind", "UNKNOWN"), ("invariant_source_mode", "CUSTOM"), ("source_receipt_hash", "x"), ("replay_equivalence_receipt_hash", "A" * 64), ("declared_input_scope_hash", "1" * 63), ("declared_output_scope_hash", "z" * 64), ("invariant_claim_scope", "GLOBAL"), ("optimization_relevance", "CUSTOM"), ("invariant_authority_allowed", True), ("runtime_discovery_allowed", True), ("invariant_id", "speed proves correctness")]
    for field, value in bad_inv_cases:
        code = "INVALID_HASH" if "hash" in field and value not in {True, False} else "INVALID_INPUT"
        _expect_error(code, doc.validate_decoder_optimization_invariant_source, _rehash(invariants[0], "_invariant_source_payload", "decoder_optimization_invariant_source_hash", **{field: value}))
    bad_target_cases = [("target_id", ""), ("target_kind", "UNKNOWN"), ("target_status", "CUSTOM"), ("target_description", ""), ("target_description", "benchmark proves correctness"), ("optimization_mode", "IMPLEMENT"), ("expected_future_fast_path_release", "v166.5"), ("implementation_allowed_in_this_release", True), ("runtime_execution_allowed", True), ("benchmark_claim_allowed", True), ("speedup_claim_allowed", True), ("correctness_claim_allowed", True), ("global_correctness_claim_allowed", True), ("hardware_authority_allowed", True), ("qec_advantage_claim_allowed", True)]
    for field, value in bad_target_cases:
        _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_target, _rehash(targets[0], "_target_payload", "decoder_optimization_target_hash", **{field: value}))
    for field, value, code in [("required_prior_receipt_kind", "Other", "INVALID_INPUT"), ("required_prior_release", "v166.1", "INVALID_INPUT"), ("required_prior_replay_equivalence_receipt_hash", "x", "INVALID_HASH"), ("required_future_receipt_kind", "Other", "INVALID_INPUT"), ("required_future_release", "v166.5", "INVALID_INPUT"), ("equivalence_mode", "APPROXIMATE", "INVALID_INPUT"), ("declared_corpus_only", False, "INVALID_INPUT"), ("exact_output_match_required", False, "INVALID_INPUT"), ("output_schema_match_required", False, "INVALID_INPUT"), ("canonical_ordering_match_required", False, "INVALID_INPUT"), ("precision_policy", "hidden precision drift", "INVALID_INPUT"), ("approximation_policy", "undeclared approximation policy", "INVALID_INPUT"), ("fast_path_equivalence_required_before_implementation", False, "INVALID_INPUT"), ("optimization_valid_without_replay_equivalence", True, "INVALID_INPUT")]:
        _expect_error(code, doc.validate_decoder_optimization_equivalence_gate, _rehash(gate, "_equivalence_gate_payload", "decoder_optimization_equivalence_gate_hash", **{field: value}))
    for field, value in [("transformation_boundary_id", ""), ("transformation_mode", "CUSTOM"), ("allowed_transformation_kinds", ()), ("allowed_transformation_kinds", ("DECLARED_CACHE_KEY_PRECONDITION", "DECLARED_CACHE_KEY_PRECONDITION")), ("allowed_transformation_kinds", ("UNKNOWN",)), ("transformation_count", 99), ("source_mutation_allowed", True), ("baseline_mutation_allowed", True), ("candidate_runtime_import_allowed", True), ("candidate_runtime_execution_allowed", True), ("fast_path_code_allowed", True), ("implementation_code_allowed", True), ("filesystem_mutation_allowed", True)]:
        _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_transformation_boundary, _rehash(transformation, "_transformation_boundary_payload", "decoder_optimization_transformation_boundary_hash", **{field: value}))
    for field, value in [("precision_boundary_id", ""), ("precision_policy", "CUSTOM"), ("approximation_policy", "CUSTOM"), ("reduced_precision_allowed", True), ("hidden_precision_drift_allowed", True), ("float_equality_identity_allowed", True), ("ulp_policy_required_for_future_approximation", False), ("approximation_error_bound_required", False), ("hardware_float_authority_allowed", True)]:
        _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_precision_boundary, _rehash(precision, "_precision_boundary_payload", "decoder_optimization_precision_boundary_hash", **{field: value}))
    for field, value in [("benchmark_boundary_id", ""), ("benchmark_mode", "CUSTOM"), ("benchmark_execution_allowed", True), ("speedup_claim_allowed", True), ("benchmark_claim_allowed", True), ("benchmark_ladder_required_before_claims", False), ("required_future_benchmark_receipt_kind", "Other"), ("required_future_benchmark_release", "v166.5"), ("comparator_receipt_required", False), ("hardware_declaration_required", False), ("corpus_declaration_required", False)]:
        _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_benchmark_boundary, _rehash(benchmark, "_benchmark_boundary_payload", "decoder_optimization_benchmark_boundary_hash", **{field: value}))
    for field, value in [("rollback_policy_id", ""), ("rollback_mode", "CUSTOM"), ("rollback_required_before_promotion", False), ("required_future_rollback_receipt_kind", "Other"), ("required_future_rollback_release", "v166.8"), ("rollback_trigger_conditions", ()), ("rollback_trigger_conditions", ("OUTPUT_SCHEMA_DRIFT", "OUTPUT_SCHEMA_DRIFT")), ("rollback_trigger_conditions", ("UNKNOWN",)), ("rollback_trigger_count", 99), ("rollback_path_deletion_allowed", True), ("baseline_restore_required", False), ("candidate_disable_required_on_failure", False), ("promotion_blocked_without_rollback_receipt", False), ("rollback_policy_id", "rollback bypass")]:
        _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_rollback_policy, _rehash(rollback, "_rollback_policy_payload", "decoder_optimization_rollback_policy_hash", **{field: value}))
    for field, value in [("authority_boundary_id", ""), ("authority_mode", "CUSTOM"), ("candidate_adapter_only", False), ("promotion_allowed_in_this_release", True), ("runtime_authority_allowed", True), ("benchmark_authority_allowed", True), ("hardware_authority_allowed", True), ("ml_decoder_authority_allowed", True), ("probabilistic_decoder_authority_allowed", True), ("qec_advantage_claim_allowed", True), ("global_correctness_claim_allowed", True), ("silent_replacement_allowed", True), ("baseline_mutation_allowed", True)]:
        _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_authority_boundary, _rehash(authority, "_authority_boundary_payload", "decoder_optimization_authority_boundary_hash", **{field: value}))


def test_aggregate_contract_validation_rejects_bad_fields_and_cross_links():
    upstream, invariants, targets, gate, transformation, precision, benchmark, rollback, authority, contract = _parts()
    bad_cases = [("contract_version", "v166.2", "INVALID_INPUT"), ("contract_kind", "Other", "INVALID_INPUT"), ("previous_release_tag", "v166.1", "INVALID_INPUT"), ("previous_release_url", "https://example.invalid", "INVALID_INPUT"), ("invariant_sources", (), "INVALID_INPUT"), ("optimization_targets", (), "INVALID_INPUT"), ("invariant_sources", (invariants[0], invariants[0]), "INVALID_INPUT"), ("optimization_targets", (targets[0], targets[0]), "INVALID_INPUT"), ("invariant_source_count", 99, "INVALID_INPUT"), ("optimization_target_count", 99, "INVALID_INPUT"), ("fast_path_implementation_allowed", True, "INVALID_DECODER_OPTIMIZATION_CONTRACT"), ("promotion_allowed", True, "INVALID_DECODER_OPTIMIZATION_CONTRACT"), ("benchmark_claim_allowed", True, "INVALID_DECODER_OPTIMIZATION_CONTRACT"), ("speedup_claim_allowed", True, "INVALID_DECODER_OPTIMIZATION_CONTRACT")]
    for field, value, code in bad_cases:
        _expect_error(code, doc.validate_decoder_optimization_contract, _rehash(contract, "_contract_payload", "decoder_optimization_contract_hash", **{field: value}))
    other_inv = doc.build_decoder_optimization_invariant_source(invariant_id="inv-c", invariant_kind="CANONICAL_OUTPUT_SCHEMA_INVARIANT", source_receipt_hash=HEX_A, replay_equivalence_receipt_hash=HEX_D, declared_input_scope_hash=HEX_D, declared_output_scope_hash=HEX_E, optimization_relevance="MAY_SUPPORT_FUTURE_FAST_PATH")
    _expect_error("INVALID_DECODER_OPTIMIZATION_CONTRACT", doc.validate_decoder_optimization_contract, _rehash(contract, "_contract_payload", "decoder_optimization_contract_hash", invariant_sources=(invariants[0], other_inv)))
    other_gate = doc.build_decoder_optimization_equivalence_gate(gate_id="gate2", required_prior_replay_equivalence_receipt_hash=HEX_D)
    _expect_error("INVALID_DECODER_OPTIMIZATION_CONTRACT", doc.validate_decoder_optimization_contract, _rehash(contract, "_contract_payload", "decoder_optimization_contract_hash", equivalence_gate=other_gate))
    bad_safe = _rehash(contract, "_contract_payload", "decoder_optimization_contract_hash", optimization_contract_safe=False)
    _expect_error("INVALID_DECODER_OPTIMIZATION_CONTRACT", doc.validate_decoder_optimization_contract, bad_safe)
    bad_adapter = _rehash(contract, "_contract_payload", "decoder_optimization_contract_hash", candidate_remains_adapter_only=False)
    _expect_error("INVALID_DECODER_OPTIMIZATION_CONTRACT", doc.validate_decoder_optimization_contract, bad_adapter)

    single_child_contract = doc.build_decoder_optimization_contract(
        upstream_binding=upstream,
        invariant_sources=(invariants[0],),
        optimization_targets=(targets[0],),
        equivalence_gate=gate,
        transformation_boundary=transformation,
        precision_boundary=precision,
        benchmark_boundary=benchmark,
        rollback_policy=rollback,
        authority_boundary=authority,
    )
    for field in ("invariant_source_count", "optimization_target_count"):
        malformed = _rehash(single_child_contract, "_contract_payload", "decoder_optimization_contract_hash", **{field: True})
        error = _expect_error("INVALID_INPUT", doc.validate_decoder_optimization_contract, malformed)
        assert error.detail == f"{field}:INT"


def test_forbidden_semantic_hardening_and_positive_controls():
    phrases = [
        "silent_decoder_replacement", "candidate-replaces-baseline", "decoder replaced because faster",
        "speed proves correctness", "benchmark proves correctness", "benchmark marketing", "runtime promotion",
        "candidate decoder promoted", "probabilistic decoder authority", "ML decoder authority", "hardware authority",
        "QEC advantage proven", "hidden precision drift", "undeclared approximation policy",
        "output accepted as universal canonical truth", "global correctness proven", "replay equivalence implies promotion",
        "replay equivalence implies speedup", "optimization implies correctness", "optimization grants execution authority",
        "contract permits implementation", "fast path accepted", "benchmark proves optimization", "candidate\\nreplaces\\tbaseline",
        "candidate\nreplaces\tbaseline",
    ]
    for phrase in phrases:
        _expect_error("INVALID_INPUT", doc.build_decoder_optimization_target, target_id="bad", target_kind="SPARSE_HANDLING_TARGET", target_description=phrase)
    for phrase in ["optimization_contract_safe", "rollback_required_before_promotion", "benchmark_ladder_required_before_claims", "fast_path_equivalence_required_before_implementation"]:
        assert doc.build_decoder_optimization_target(target_id=phrase, target_kind="SPARSE_HANDLING_TARGET", target_description="Declarative precondition only")


def test_boundary_static_import_and_decoder_immutability_checks():
    module_path = Path("src/qec/analysis/decoder_optimization_contracts.py")
    test_path = Path("tests/test_decoder_optimization_contracts.py")
    for path in (module_path, test_path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
                assert "qec.decoder" not in names
                assert not (names & {"numpy", "scipy", "qldpc", "stim", "pymatching", "qiskit", "qutip", "pandas", "polars", "torch", "tensorflow", "jax", "socket", "urllib", "requests", "importlib"})
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not mod.startswith("qec.decoder")
                assert mod.split(".")[0] not in {"numpy", "scipy", "qldpc", "stim", "pymatching", "qiskit", "qutip", "pandas", "polars", "torch", "tensorflow", "jax", "socket", "urllib", "requests", "importlib"}
    assert not list(Path("src/qec/decoder").glob("*candidate*optimization*"))
    diff = subprocess.run(["git", "diff", "--name-only", "--", "src/qec/decoder/"], check=True, capture_output=True, text=True)
    assert diff.stdout == ""


def test_hash_seed_stability_subprocesses():
    code = """
from qec.analysis import decoder_optimization_contracts as d
A='a'*64;B='b'*64;C='c'*64;D='d'*64;E='e'*64
u=d.build_decoder_optimization_upstream_binding(upstream_canonical_decoder_baseline_receipt_hash=A,upstream_decoder_candidate_manifest_hash=B,upstream_decoder_replay_equivalence_receipt_hash=C,candidate_declaration_hash=D,candidate_name='candidate',candidate_version='1')
i=d.build_decoder_optimization_invariant_source(invariant_id='inv',invariant_kind='SPARSE_SYNDROME_STRUCTURE_INVARIANT',source_receipt_hash=A,replay_equivalence_receipt_hash=C,declared_input_scope_hash=D,declared_output_scope_hash=E,optimization_relevance='MAY_SUPPORT_SPARSE_HANDLING')
t=d.build_decoder_optimization_target(target_id='target',target_kind='SPARSE_HANDLING_TARGET',target_description='Declarative precondition only')
g=d.build_decoder_optimization_equivalence_gate(gate_id='gate',required_prior_replay_equivalence_receipt_hash=C)
tb=d.build_decoder_optimization_transformation_boundary(transformation_boundary_id='tb',allowed_transformation_kinds={'DECLARED_CACHE_KEY_PRECONDITION','DECLARED_SPARSE_REPRESENTATION_PRECONDITION'})
p=d.build_decoder_optimization_precision_boundary(precision_boundary_id='p')
b=d.build_decoder_optimization_benchmark_boundary(benchmark_boundary_id='b')
r=d.build_decoder_optimization_rollback_policy(rollback_policy_id='r',rollback_trigger_conditions={'OUTPUT_SCHEMA_DRIFT','FAST_PATH_EQUIVALENCE_FAILURE'})
a=d.build_decoder_optimization_authority_boundary(authority_boundary_id='a')
c=d.build_decoder_optimization_contract(upstream_binding=u,invariant_sources=(i,),optimization_targets=(t,),equivalence_gate=g,transformation_boundary=tb,precision_boundary=p,benchmark_boundary=b,rollback_policy=r,authority_boundary=a)
print(c.decoder_optimization_contract_hash)
"""
    outs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONPATH": "src", "PYTHONHASHSEED": seed}
        run = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env)
        outs.append(run.stdout.strip())
    assert outs[0] == outs[1]
