from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from qec.analysis import decoder_fast_path_equivalence_receipts as fp

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64
S = "1" * 64
O = "2" * 64
SRC = "3" * 64


def _hash(obj, field):
    return fp._hash_payload(fp._dataclass_payload(obj, exclude_hash_field=field))


def _unsafe_replace(obj, **changes):
    clone = object.__new__(type(obj))
    for f in obj.__dataclass_fields__:
        object.__setattr__(clone, f, changes.get(f, getattr(obj, f)))
    return clone


def _fixture(order=False):
    upstream = fp.build_decoder_fast_path_upstream_binding(
        upstream_canonical_decoder_baseline_receipt_hash=A,
        upstream_decoder_candidate_manifest_hash=B,
        upstream_decoder_replay_equivalence_receipt_hash=C,
        upstream_decoder_optimization_contract_hash=D,
        candidate_declaration_hash=E,
    )
    ident = fp.build_decoder_fast_path_identity(
        associated_optimization_contract_hash=D,
        associated_candidate_declaration_hash=E,
    )
    source = fp.build_decoder_fast_path_source_boundary(
        declared_source_files=("fast_path_declarations/b.json", "fast_path_declarations/a.json") if order else ("fast_path_declarations/a.json", "fast_path_declarations/b.json"),
        declared_source_file_hashes=(B, A) if order else (A, B),
    )
    contract = fp.build_decoder_fast_path_contract_binding(
        optimization_contract_hash=D,
        invariant_source_hashes=(B, A) if order else (A, B),
        optimization_target_hashes=(E, C) if order else (C, E),
    )
    policy = fp.build_decoder_fast_path_equivalence_policy()
    boundary = fp.build_decoder_fast_path_execution_boundary()
    i1 = fp.build_decoder_fast_path_corpus_item("r1", (0, 1, 1), S, "k1", replay_corpus_item_hash=A)
    i2 = fp.build_decoder_fast_path_corpus_item("r2", (1, 0, 1), S, "k2", replay_corpus_item_hash=B)
    r1 = fp.build_decoder_fast_path_output_record("r1", "REFERENCE_DECODER_TRANSCRIPT", (1, 0, 0), O, "k1", SRC)
    r2 = fp.build_decoder_fast_path_output_record("r2", "REFERENCE_DECODER_TRANSCRIPT", (0, 1, 0), O, "k2", SRC)
    f1 = fp.build_decoder_fast_path_output_record("r1", "FAST_PATH_TRANSCRIPT", (1, 0, 0), O, "k1", SRC)
    f2 = fp.build_decoder_fast_path_output_record("r2", "FAST_PATH_TRANSCRIPT", (0, 1, 0), O, "k2", SRC)
    c1 = fp.build_decoder_fast_path_comparison_record(i1.syndrome_input_hash, i1.canonical_ordering_key, r1, f1)
    c2 = fp.build_decoder_fast_path_comparison_record(i2.syndrome_input_hash, i2.canonical_ordering_key, r2, f2)
    items = (i2, i1) if order else (i1, i2)
    refs = (r2, r1) if order else (r1, r2)
    fast = (f2, f1) if order else (f1, f2)
    comps = (c2, c1) if order else (c1, c2)
    summary = fp.build_decoder_fast_path_transcript_summary("declared transcript", "1", items, refs, fast, comps, S, O)
    receipt = fp.build_decoder_fast_path_equivalence_receipt(upstream, ident, source, contract, policy, boundary, items, refs, fast, comps, summary)
    return locals()


def _assert_code(fn, code):
    with pytest.raises(fp.DecoderFastPathEquivalenceError) as exc:
        fn()
    assert exc.value.code is code
    assert code.value in str(exc.value)
    assert exc.value.detail


def test_happy_path_builds_validates_and_is_frozen():
    fx = _fixture()
    for name, validator in [
        ("upstream", fp.validate_decoder_fast_path_upstream_binding),
        ("ident", fp.validate_decoder_fast_path_identity),
        ("source", fp.validate_decoder_fast_path_source_boundary),
        ("contract", fp.validate_decoder_fast_path_contract_binding),
        ("policy", fp.validate_decoder_fast_path_equivalence_policy),
        ("boundary", fp.validate_decoder_fast_path_execution_boundary),
        ("i1", fp.validate_decoder_fast_path_corpus_item),
        ("r1", fp.validate_decoder_fast_path_output_record),
        ("c1", fp.validate_decoder_fast_path_comparison_record),
        ("summary", fp.validate_decoder_fast_path_transcript_summary),
        ("receipt", fp.validate_decoder_fast_path_equivalence_receipt),
    ]:
        assert validator(fx[name]) is fx[name]
    receipt = fx["receipt"]
    assert len(receipt.decoder_fast_path_equivalence_receipt_hash) == 64
    assert receipt.fast_path_equivalence_proven is True
    assert receipt.fast_path_equivalence_scope == "DECLARED_CORPUS_ONLY"
    assert receipt.candidate_remains_adapter_only is True
    assert receipt.implementation_allowed is False
    assert receipt.runtime_enabled is False
    assert receipt.promotion_allowed is False
    assert receipt.benchmark_claim_allowed is False
    assert receipt.speedup_claim_allowed is False
    assert receipt.global_correctness_claim_allowed is False
    with pytest.raises(FrozenInstanceError):
        receipt.runtime_enabled = True


def test_canonical_hash_determinism_and_ordering():
    r1 = _fixture(False)["receipt"]
    r2 = _fixture(True)["receipt"]
    assert r1.decoder_fast_path_equivalence_receipt_hash == r2.decoder_fast_path_equivalence_receipt_hash
    assert _fixture()["receipt"].decoder_fast_path_equivalence_receipt_hash == r1.decoder_fast_path_equivalence_receipt_hash
    assert _fixture(False)["source"].decoder_fast_path_source_boundary_hash == _fixture(True)["source"].decoder_fast_path_source_boundary_hash
    assert _fixture(False)["contract"].decoder_fast_path_contract_binding_hash == _fixture(True)["contract"].decoder_fast_path_contract_binding_hash


def test_self_hash_exclusion_and_hash_mismatch():
    fx = _fixture()
    stale = _unsafe_replace(fx["receipt"], decoder_fast_path_equivalence_receipt_hash=A)
    _assert_code(lambda: fp.validate_decoder_fast_path_equivalence_receipt(stale), fp.DecoderFastPathEquivalenceErrorCode.HASH_MISMATCH)
    assert _hash(fx["receipt"], "decoder_fast_path_equivalence_receipt_hash") == fx["receipt"].decoder_fast_path_equivalence_receipt_hash
    assert _hash(fx["summary"], "decoder_fast_path_transcript_summary_hash") == fx["summary"].decoder_fast_path_transcript_summary_hash
    assert _hash(fx["c1"], "decoder_fast_path_comparison_record_hash") == fx["c1"].decoder_fast_path_comparison_record_hash
    assert _hash(fx["i1"], "decoder_fast_path_corpus_item_hash") == fx["i1"].decoder_fast_path_corpus_item_hash


def test_child_before_aggregate_validation_rejects_corruption():
    fx = _fixture()
    bad_output = _unsafe_replace(fx["r1"], output_payload_hash=A, decoder_fast_path_output_record_hash=A)
    _assert_code(lambda: fp.build_decoder_fast_path_comparison_record(fx["i1"].syndrome_input_hash, "k1", bad_output, fx["f1"]), fp.DecoderFastPathEquivalenceErrorCode.HASH_MISMATCH)
    bad_contract_raw = _unsafe_replace(fx["contract"], optimization_contract_safe=False)
    bad_contract = _unsafe_replace(bad_contract_raw, decoder_fast_path_contract_binding_hash=_hash(bad_contract_raw, "decoder_fast_path_contract_binding_hash"))
    bad_receipt = _unsafe_replace(fx["receipt"], contract_binding=bad_contract, decoder_fast_path_equivalence_receipt_hash=A)
    _assert_code(lambda: fp.validate_decoder_fast_path_equivalence_receipt(bad_receipt), fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT)


def test_upstream_binding_validation_matrix():
    cases = [
        dict(previous_release_tag="v166.2"), dict(previous_release_url="https://example.test"), dict(fast_path_release="v166.5"),
        dict(upstream_canonical_decoder_baseline_receipt_hash="A"*64), dict(upstream_decoder_candidate_manifest_hash="bad"), dict(upstream_decoder_replay_equivalence_receipt_hash="bad"), dict(upstream_decoder_optimization_contract_hash="bad"), dict(candidate_declaration_hash="bad"),
        dict(candidate_name=""), dict(candidate_version=""), dict(replay_equivalence_proven_for_declared_corpus=False), dict(optimization_contract_safe=False), dict(candidate_adapter_only=False), dict(candidate_promoted=True), dict(baseline_immutable=False), dict(baseline_mutation_allowed=True), dict(implementation_allowed_by_contract=True), dict(runtime_authority_allowed=True), dict(candidate_adapter_only=1),
    ]
    for kwargs in cases:
        _assert_code(lambda kwargs=kwargs: fp.build_decoder_fast_path_upstream_binding(**kwargs), fp.DecoderFastPathEquivalenceErrorCode.INVALID_HASH if "hash" in next(iter(kwargs)) else fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT)


def test_fast_path_identity_validation_matrix_and_semantics():
    cases = [dict(fast_path_id=""), dict(fast_path_name=""), dict(fast_path_version=""), dict(fast_path_kind="UNKNOWN"), dict(fast_path_status="IMPLEMENTED"), dict(fast_path_source_mode="RUNTIME"), dict(associated_optimization_contract_hash="bad"), dict(associated_candidate_declaration_hash="bad"), dict(adapter_only=False), dict(runtime_enabled=True), dict(implementation_allowed=True), dict(promotion_allowed=True), dict(benchmark_claim_allowed=True), dict(speedup_claim_allowed=True), dict(hardware_authority_allowed=True), dict(qec_advantage_claim_allowed=True), dict(fast_path_name="fast path accepted")]
    for kwargs in cases:
        _assert_code(lambda kwargs=kwargs: fp.build_decoder_fast_path_identity(**kwargs), fp.DecoderFastPathEquivalenceErrorCode.INVALID_HASH if "hash" in next(iter(kwargs)) else fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT)


def test_source_boundary_validation_matrix():
    cases = [
        dict(fast_path_source_root="/abs/"), dict(fast_path_source_root="fast_path_declarations"), dict(fast_path_source_root="src/qec/decoder/"), dict(fast_path_source_root="src/qec/analysis/"),
        dict(source_boundary_mode="RUNTIME"), dict(declared_source_files=()), dict(declared_source_files=("fast_path_declarations/a.json", "fast_path_declarations/a.json"), declared_source_file_hashes=(A, B)),
        dict(declared_source_files=("/a",)), dict(declared_source_files=("fast_path_declarations/./a",)), dict(declared_source_files=("fast_path_declarations/../a",)), dict(declared_source_files=("fast_path_declarations//a",)), dict(declared_source_files=("fast_path_declarations\\a",)), dict(declared_source_files=("src/qec/decoder/a.py",)), dict(declared_source_files=("external/other/a",)),
        dict(declared_source_file_hashes=(A, B)), dict(declared_source_file_hashes=("bad",)), dict(source_files_exist_required=True), dict(runtime_import_allowed=True), dict(runtime_execution_allowed=True), dict(implementation_file_creation_allowed=True), dict(baseline_mutation_allowed=True), dict(filesystem_mutation_allowed=True), dict(runtime_import_allowed=1),
    ]
    for kwargs in cases:
        code = fp.DecoderFastPathEquivalenceErrorCode.INVALID_HASH if "declared_source_file_hashes" in kwargs and kwargs["declared_source_file_hashes"] == ("bad",) else fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT
        _assert_code(lambda kwargs=kwargs: fp.build_decoder_fast_path_source_boundary(**kwargs), code)


def test_contract_policy_and_execution_boundary_validation_matrices():
    contract_cases = [dict(optimization_contract_hash="bad"), dict(optimization_contract_safe=False), dict(invariant_source_hashes=()), dict(invariant_source_hashes=(A, A)), dict(invariant_source_hashes=("bad",)), dict(optimization_target_hashes=()), dict(optimization_target_hashes=(A, A)), dict(optimization_target_hashes=("bad",)), dict(equivalence_gate_hash="bad"), dict(transformation_boundary_hash="bad"), dict(precision_boundary_hash="bad"), dict(benchmark_boundary_hash="bad"), dict(rollback_policy_hash="bad"), dict(authority_boundary_hash="bad"), dict(required_future_implementation_boundary_release="v166.6"), dict(implementation_boundary_required_before_runtime=False), dict(benchmark_ladder_required_before_speed_claims=False), dict(rollback_receipt_required_before_promotion=False)]
    for kwargs in contract_cases:
        _assert_code(lambda kwargs=kwargs: fp.build_decoder_fast_path_contract_binding(**kwargs), fp.DecoderFastPathEquivalenceErrorCode.INVALID_HASH if ("hash" in next(iter(kwargs)) and kwargs.get(next(iter(kwargs))) == "bad") or kwargs.get(next(iter(kwargs))) == ("bad",) else fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT)
    policy_cases = [dict(equivalence_mode="APPROXIMATE"), dict(equivalence_mode="PROBABILISTIC"), dict(equivalence_mode="CUSTOM"), dict(comparison_mode="OTHER"), dict(reference_output_role="OTHER"), dict(fast_path_output_role="OTHER"), dict(tie_breaking_policy="NONDETERMINISTIC"), dict(partial_hash_match_allowed=True), dict(approximate_match_allowed=True), dict(probabilistic_match_allowed=True), dict(benchmark_claim_allowed=True), dict(speedup_claim_allowed=True), dict(hardware_authority_allowed=True), dict(qec_advantage_claim_allowed=True), dict(candidate_promotion_allowed=True), dict(global_correctness_claim_allowed=True), dict(precision_policy="hidden precision drift"), dict(approximation_policy="undeclared approximation policy")]
    for kwargs in policy_cases:
        _assert_code(lambda kwargs=kwargs: fp.build_decoder_fast_path_equivalence_policy(**kwargs), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE if list(kwargs)[0] not in {"precision_policy", "approximation_policy"} else fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT)
    boundary_cases = [dict(execution_boundary_mode="OTHER"), dict(declared_fast_path_transcript_only=False), dict(baseline_decoder_import_allowed=True), dict(candidate_decoder_import_allowed=True), dict(fast_path_import_allowed=True), dict(runtime_decoder_execution_allowed=True), dict(fast_path_runtime_execution_allowed=True), dict(optimization_execution_allowed=True), dict(benchmark_execution_allowed=True), dict(network_allowed=True), dict(heavy_backend_import_allowed=True), dict(hardware_sdk_allowed=True), dict(filesystem_mutation_allowed=True), dict(implementation_file_creation_allowed=True), dict(candidate_promotion_allowed=True), dict(network_allowed=1)]
    for kwargs in boundary_cases:
        _assert_code(lambda kwargs=kwargs: fp.build_decoder_fast_path_execution_boundary(**kwargs), fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT if type(kwargs.get(list(kwargs)[0])) is int else fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)


def test_corpus_output_comparison_summary_and_aggregate_rejections():
    fx = _fixture()
    invalid_input_calls = [
        lambda: fp.build_decoder_fast_path_corpus_item("", (1,), S, "k"),
        lambda: fp.build_decoder_fast_path_corpus_item("r", (), S, "k"),
        lambda: fp.build_decoder_fast_path_corpus_item("r", (2,), S, "k"),
        lambda: fp.build_decoder_fast_path_corpus_item("r", (True,), S, "k"),
        lambda: fp.build_decoder_fast_path_corpus_item("r", (1,), S, ""),
        lambda: fp.build_decoder_fast_path_output_record("", "REFERENCE_DECODER_TRANSCRIPT", (1,), O, "k", SRC),
        lambda: fp.build_decoder_fast_path_output_record("r", "OTHER", (1,), O, "k", SRC),
        lambda: fp.build_decoder_fast_path_output_record("r", "REFERENCE_DECODER_TRANSCRIPT", (), O, "k", SRC),
        lambda: fp.build_decoder_fast_path_output_record("r", "REFERENCE_DECODER_TRANSCRIPT", (2,), O, "k", SRC),
        lambda: fp.build_decoder_fast_path_output_record("r", "REFERENCE_DECODER_TRANSCRIPT", (True,), O, "k", SRC),
        lambda: fp.build_decoder_fast_path_output_record("r", "REFERENCE_DECODER_TRANSCRIPT", (1,), O, "", SRC),
    ]
    for call in invalid_input_calls:
        _assert_code(call, fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT)
    invalid_hash_calls = [
        lambda: fp.build_decoder_fast_path_corpus_item("r", (1,), "bad", "k"),
        lambda: fp.build_decoder_fast_path_output_record("r", "REFERENCE_DECODER_TRANSCRIPT", (1,), "bad", "k", SRC),
        lambda: fp.build_decoder_fast_path_output_record("r", "REFERENCE_DECODER_TRANSCRIPT", (1,), O, "k", "bad"),
    ]
    for call in invalid_hash_calls:
        _assert_code(call, fp.DecoderFastPathEquivalenceErrorCode.INVALID_HASH)
    bad_fast = fp.build_decoder_fast_path_output_record("r1", "FAST_PATH_TRANSCRIPT", (0, 0, 0), O, "k1", SRC)
    _assert_code(lambda: fp.build_decoder_fast_path_comparison_record(fx["i1"].syndrome_input_hash, "k1", fx["r1"], bad_fast), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)
    _assert_code(lambda: fp.build_decoder_fast_path_comparison_record(fx["i1"].syndrome_input_hash, "wrong", fx["r1"], fx["f1"]), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)
    _assert_code(lambda: fp.build_decoder_fast_path_comparison_record(fx["i1"].syndrome_input_hash, "k1", fx["f1"], fx["f1"]), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)
    for kwargs in [dict(transcript_name=""), dict(transcript_version=""), dict(transcript_mode="OTHER"), dict(syndrome_schema_hash="bad"), dict(output_schema_hash="bad")]:
        def make(kwargs=kwargs):
            return fp.build_decoder_fast_path_transcript_summary(kwargs.get("transcript_name", "n"), kwargs.get("transcript_version", "1"), fx["items"], fx["refs"], fx["fast"], fx["comps"], kwargs.get("syndrome_schema_hash", S), kwargs.get("output_schema_hash", O), transcript_mode=kwargs.get("transcript_mode", "DECLARED_STATIC_FAST_PATH_TRANSCRIPT"))
        _assert_code(make, fp.DecoderFastPathEquivalenceErrorCode.INVALID_HASH if "hash" in next(iter(kwargs)) else fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT)
    _assert_code(lambda: fp.build_decoder_fast_path_transcript_summary("n", "1", (fx["i1"].decoder_fast_path_corpus_item_hash,), (fx["r1"].decoder_fast_path_output_record_hash,), (fx["f1"].decoder_fast_path_output_record_hash,), (fx["c1"].decoder_fast_path_comparison_record_hash,), S, O), fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT)
    _assert_code(lambda: fp.build_decoder_fast_path_transcript_summary("n", "1", fx["items"], fx["refs"][:1], fx["fast"][:1], fx["comps"][:1], S, O), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)
    _assert_code(lambda: fp.build_decoder_fast_path_transcript_summary("n", "1", (fx["i2"],), (fx["r1"],), (fx["f1"],), (fx["c1"],), S, O), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)
    dup_receipt = _unsafe_replace(fx["receipt"], corpus_items=(fx["i1"], fx["i1"]), decoder_fast_path_equivalence_receipt_hash=A)
    _assert_code(lambda: fp.validate_decoder_fast_path_equivalence_receipt(dup_receipt), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)
    _assert_code(lambda: fp.build_decoder_fast_path_equivalence_receipt(fx["upstream"], fx["ident"], fx["source"], fx["contract"], fx["policy"], fx["boundary"], (), fx["refs"], fx["fast"], fx["comps"], fx["summary"]), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)
    _assert_code(lambda: fp.build_decoder_fast_path_equivalence_receipt(fx["upstream"], fx["ident"], fx["source"], fx["contract"], fx["policy"], fx["boundary"], fx["items"], fx["refs"][:1], fx["fast"], fx["comps"], fx["summary"]), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)
    _assert_code(lambda: fp.build_decoder_fast_path_equivalence_receipt(fx["upstream"], fx["ident"], fx["source"], fx["contract"], fx["policy"], fx["boundary"], fx["items"], fx["refs"], fx["fast"], fx["comps"], fx["summary"], runtime_enabled=True), fp.DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE)


def test_forbidden_semantic_hardening_and_positive_controls():
    variants = [
        "silent_decoder_replacement", "candidate-replaces-baseline", "decoder replaced because faster", "speed proves correctness", "benchmark proves correctness", "benchmark marketing", "runtime promotion", "candidate decoder promoted", "probabilistic decoder authority", "ML decoder authority", "hardware authority", "QEC advantage proven", "hidden precision drift", "undeclared approximation policy", "output accepted as universal canonical truth", "global correctness proven", "replay equivalence implies promotion", "replay equivalence implies speedup", "optimization implies correctness", "optimization grants execution authority", "contract permits implementation", "fast path accepted", "fast path implemented", "fast path runtime enabled", "fast path proves speedup", "benchmark proves fast path", "implementation permission granted", "fast\npath\taccepted", "fast\\npath\\timplemented",
    ]
    for phrase in variants:
        _assert_code(lambda phrase=phrase: fp.build_decoder_fast_path_identity(fast_path_id=f"id {phrase}"), fp.DecoderFastPathEquivalenceErrorCode.INVALID_INPUT)
    for phrase in ["fast_path_equivalence_proven", "fast_path_equivalence_proven_for_declared_corpus", "implementation_boundary_required_before_runtime", "benchmark_ladder_required_before_speed_claims", "rollback_receipt_required_before_promotion"]:
        fp.build_decoder_fast_path_identity(fast_path_id=f"legit {phrase}")


def _is_banned_import_name(name: str, banned_roots: set[str]) -> bool:
    return (
        name == "qec.decoder"
        or name.startswith("qec.decoder.")
        or name.split(".")[0] in banned_roots
        or name == "importlib"
        or name.startswith("importlib.")
    )


def test_boundary_static_inspection_and_no_decoder_diff():
    module_path = Path("src/qec/analysis/decoder_fast_path_equivalence_receipts.py")
    test_path = Path(__file__)
    banned_imports = {"numpy", "scipy", "qldpc", "stim", "pymatching", "qiskit", "qutip", "pandas", "polars", "torch", "tensorflow", "jax"}
    for path in (module_path, test_path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not _is_banned_import_name(alias.name, banned_imports)
            if isinstance(node, ast.ImportFrom):
                assert not _is_banned_import_name(node.module or "", banned_imports)
    base = subprocess.check_output(["git", "rev-parse", "HEAD^"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--name-only", f"{base}..HEAD", "--", "src/qec/decoder/"], text=True)
    assert diff == ""


def test_hash_seed_stability_subprocess():
    code = """
from tests.test_decoder_fast_path_equivalence_receipts import _fixture
print(_fixture()['receipt'].decoder_fast_path_equivalence_receipt_hash)
"""
    env = os.environ.copy(); env["PYTHONPATH"] = "src:."
    env0 = dict(env, PYTHONHASHSEED="0")
    env1 = dict(env, PYTHONHASHSEED="1")
    h0 = subprocess.check_output([sys.executable, "-c", code], env=env0, text=True).strip()
    h1 = subprocess.check_output([sys.executable, "-c", code], env=env1, text=True).strip()
    assert h0 == h1
