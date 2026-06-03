from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from qec.analysis import decoder_replay_equivalence_receipts as dre

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64


def _expect_error(code: str, fn, *args, **kwargs):
    with pytest.raises(dre.DecoderReplayEquivalenceError) as exc:
        fn(*args, **kwargs)
    assert exc.value.code.value == code
    assert code in str(exc.value)
    assert exc.value.detail
    return exc.value


def _unsafe_copy(obj, **updates):
    clone = object.__new__(type(obj))
    for field in obj.__dataclass_fields__:
        object.__setattr__(clone, field, getattr(obj, field))
    for key, value in updates.items():
        object.__setattr__(clone, key, value)
    return clone


def _rehash_clone(obj, payload_fn_name: str, hash_field: str, **updates):
    clone = _unsafe_copy(obj, **updates)
    object.__setattr__(clone, hash_field, dre._hash_payload(getattr(dre, payload_fn_name)(clone)))
    return clone


def _parts():
    upstream = dre.build_decoder_replay_upstream_binding(HEX_A, HEX_B, HEX_C, "candidate alpha", "0.1.0")
    policy = dre.build_decoder_replay_equivalence_policy()
    boundary = dre.build_decoder_replay_execution_boundary()
    items = (
        dre.build_decoder_replay_corpus_item("r2", (1, 0, 1), HEX_D, "k2"),
        dre.build_decoder_replay_corpus_item("r1", (0, 1, 1), HEX_D, "k1"),
    )
    baseline = tuple(
        dre.build_decoder_replay_output_record(item.record_id, "CANONICAL_BASELINE_DECODER", item.syndrome_bits, HEX_E, item.canonical_ordering_key)
        for item in items
    )
    candidate = tuple(
        dre.build_decoder_replay_output_record(item.record_id, "CANDIDATE_DECODER", item.syndrome_bits, HEX_E, item.canonical_ordering_key)
        for item in items
    )
    comparisons = tuple(
        dre.build_decoder_replay_comparison_record(item.syndrome_input_hash, b, c)
        for item, b, c in zip(items, baseline, candidate)
    )
    corpus_summary = dre.build_decoder_replay_corpus_summary("declared replay", "1", items, HEX_D, HEX_E)
    coverage = dre.build_decoder_replay_coverage_summary(comparisons)
    receipt = dre.build_decoder_replay_equivalence_receipt(upstream, policy, boundary, corpus_summary, items, comparisons, coverage)
    return upstream, policy, boundary, items, baseline, candidate, comparisons, corpus_summary, coverage, receipt


def test_happy_path_validates_hashes_and_frozen_artifacts():
    upstream, policy, boundary, items, baseline, candidate, comparisons, corpus_summary, coverage, receipt = _parts()
    validators_and_values = (
        (dre.validate_decoder_replay_upstream_binding, upstream),
        (dre.validate_decoder_replay_equivalence_policy, policy),
        (dre.validate_decoder_replay_execution_boundary, boundary),
        (dre.validate_decoder_replay_corpus_summary, corpus_summary),
        (dre.validate_decoder_replay_coverage_summary, coverage),
        (dre.validate_decoder_replay_equivalence_receipt, receipt),
    )
    for validator, value in validators_and_values:
        assert validator(value) is value
    for value in items:
        assert dre.validate_decoder_replay_corpus_item(value) is value
    for value in baseline + candidate:
        assert dre.validate_decoder_replay_output_record(value) is value
    for value in comparisons:
        assert dre.validate_decoder_replay_comparison_record(value) is value
    assert dre._HASH_RE.fullmatch(receipt.decoder_replay_equivalence_receipt_hash)
    assert receipt.replay_equivalence_proven is True
    assert receipt.candidate_remains_adapter_only is True
    assert receipt.promotion_allowed is False
    assert receipt.benchmark_claim_allowed is False
    assert receipt.global_correctness_claim_allowed is False
    with pytest.raises(FrozenInstanceError):
        receipt.promotion_allowed = True


def test_canonical_json_hash_determinism_and_hash_seed_independence():
    upstream, policy, boundary, items, _, _, comparisons, _, _, _ = _parts()
    first_summary = dre.build_decoder_replay_corpus_summary("declared replay", "1", items, HEX_D, HEX_E)
    second_summary = dre.build_decoder_replay_corpus_summary("declared replay", "1", tuple(reversed(items)), HEX_D, HEX_E)
    third_summary = dre.build_decoder_replay_corpus_summary("declared replay", "1", tuple({i.record_id: i for i in reversed(items)}.values()), HEX_D, HEX_E)
    assert first_summary.replay_corpus_hash == second_summary.replay_corpus_hash == third_summary.replay_corpus_hash
    first_coverage = dre.build_decoder_replay_coverage_summary(comparisons)
    second_coverage = dre.build_decoder_replay_coverage_summary(tuple(reversed(comparisons)))
    assert first_coverage.decoder_replay_coverage_summary_hash == second_coverage.decoder_replay_coverage_summary_hash
    receipt1 = dre.build_decoder_replay_equivalence_receipt(upstream, policy, boundary, first_summary, tuple(reversed(items)), tuple(reversed(comparisons)), first_coverage)
    receipt2 = dre.build_decoder_replay_equivalence_receipt(upstream, policy, boundary, second_summary, items, comparisons, second_coverage)
    assert receipt1.corpus_items == tuple(sorted(items, key=lambda i: (i.canonical_ordering_key, i.record_id, i.syndrome_input_hash)))
    assert receipt1.decoder_replay_equivalence_receipt_hash == receipt2.decoder_replay_equivalence_receipt_hash
    assert dre._hash_payload({"a": [1, 2]}) == dre._hash_payload({"a": (1, 2)})


def test_self_hash_exclusion_and_stale_hashes_fail():
    *_, comparisons, _, coverage, receipt = _parts()
    stale_receipt = _unsafe_copy(receipt, decoder_replay_equivalence_receipt_hash=HEX_A)
    _expect_error("HASH_MISMATCH", dre.validate_decoder_replay_equivalence_receipt, stale_receipt)
    assert dre._hash_payload(dre._receipt_payload(receipt)) == receipt.decoder_replay_equivalence_receipt_hash
    stale_coverage = _unsafe_copy(coverage, decoder_replay_coverage_summary_hash=HEX_A)
    _expect_error("HASH_MISMATCH", dre.validate_decoder_replay_coverage_summary, stale_coverage)
    assert dre._hash_payload(dre._coverage_summary_payload(coverage)) == coverage.decoder_replay_coverage_summary_hash
    stale_comparison = _unsafe_copy(comparisons[0], decoder_replay_comparison_record_hash=HEX_A)
    _expect_error("HASH_MISMATCH", dre.validate_decoder_replay_comparison_record, stale_comparison)
    assert dre._hash_payload(dre._comparison_record_payload(comparisons[0])) == comparisons[0].decoder_replay_comparison_record_hash


def test_child_before_aggregate_validation():
    upstream, policy, boundary, items, baseline, candidate, comparisons, corpus_summary, coverage, receipt = _parts()
    corrupt_output = _rehash_clone(candidate[0], "_output_record_payload", "decoder_replay_output_record_hash", output_payload_hash=HEX_A)
    forged_comparison = _rehash_clone(comparisons[0], "_comparison_record_payload", "decoder_replay_comparison_record_hash", candidate_output=corrupt_output)
    _expect_error("HASH_MISMATCH", dre.validate_decoder_replay_comparison_record, forged_comparison)
    corrupt_comparison = _rehash_clone(comparisons[0], "_comparison_record_payload", "decoder_replay_comparison_record_hash", exact_output_match=0)
    forged_records = (corrupt_comparison,) + comparisons[1:]
    forged_coverage = dre.build_decoder_replay_coverage_summary(tuple(record.decoder_replay_comparison_record_hash for record in forged_records))
    forged_receipt = _rehash_clone(receipt, "_receipt_payload", "decoder_replay_equivalence_receipt_hash", comparison_records=tuple(sorted(forged_records, key=lambda r: (r.record_id, r.syndrome_input_hash, r.decoder_replay_comparison_record_hash))), coverage_summary=forged_coverage)
    _expect_error("INVALID_INPUT", dre.validate_decoder_replay_equivalence_receipt, forged_receipt)
    corrupt_item = _rehash_clone(items[0], "_corpus_item_payload", "decoder_replay_corpus_item_hash", syndrome_input_hash=HEX_A)
    forged_items = tuple(sorted((corrupt_item, items[1]), key=lambda i: (i.canonical_ordering_key, i.record_id, i.syndrome_input_hash)))
    forged_summary = dre.build_decoder_replay_corpus_summary("declared replay", "1", tuple(i.decoder_replay_corpus_item_hash for i in forged_items), HEX_D, HEX_E)
    forged_receipt2 = _rehash_clone(receipt, "_receipt_payload", "decoder_replay_equivalence_receipt_hash", corpus_items=forged_items, replay_corpus_summary=forged_summary)
    _expect_error("HASH_MISMATCH", dre.validate_decoder_replay_equivalence_receipt, forged_receipt2)


def test_upstream_binding_validation_rejects_unsafe_and_malformed_fields():
    good = _parts()[0]
    cases = [
        ("previous_release_tag", "v166.0", "INVALID_INPUT"),
        ("previous_release_url", "https://example.invalid", "INVALID_INPUT"),
        ("replay_release", "v166.3", "INVALID_INPUT"),
        ("upstream_canonical_decoder_baseline_receipt_hash", "A" * 64, "INVALID_HASH"),
        ("upstream_decoder_candidate_manifest_hash", "x", "INVALID_HASH"),
        ("candidate_declaration_hash", "1" * 63, "INVALID_HASH"),
        ("candidate_status", "PROMOTED", "INVALID_INPUT"),
        ("candidate_adapter_only", False, "INVALID_INPUT"),
        ("candidate_promoted", True, "INVALID_INPUT"),
        ("baseline_immutable", False, "INVALID_INPUT"),
        ("baseline_mutation_allowed", True, "INVALID_INPUT"),
        ("candidate_runtime_authority_allowed", True, "INVALID_INPUT"),
        ("candidate_adapter_only", 1, "INVALID_INPUT"),
    ]
    for field, value, code in cases:
        _expect_error(code, dre.validate_decoder_replay_upstream_binding, _rehash_clone(good, "_upstream_binding_payload", "decoder_replay_upstream_binding_hash", **{field: value}))


def test_equivalence_policy_validation_rejects_non_exact_modes_and_authority():
    good = _parts()[1]
    cases = [
        ("equivalence_mode", "APPROXIMATE_EQUIVALENCE"),
        ("equivalence_mode", "PROBABILISTIC_EQUIVALENCE"),
        ("equivalence_mode", "CUSTOM_EQUIVALENCE"),
        ("partial_hash_match_allowed", True),
        ("approximate_match_allowed", True),
        ("probabilistic_match_allowed", True),
        ("benchmark_claim_allowed", True),
        ("hardware_authority_allowed", True),
        ("qec_advantage_claim_allowed", True),
        ("candidate_promotion_allowed", True),
        ("global_correctness_claim_allowed", True),
        ("precision_policy", "HIDDEN_PRECISION_DRIFT"),
        ("approximation_policy", "UNDECLARED_APPROXIMATION_POLICY"),
        ("canonical_ordering_policy", "NONDETERMINISTIC_ORDER"),
        ("tie_breaking_policy", "NONDETERMINISTIC_TIE_BREAK"),
    ]
    assert dre.validate_decoder_replay_equivalence_policy(good) is good
    for field, value in cases:
        expected = "INVALID_INPUT" if field in {"precision_policy", "approximation_policy"} else "INVALID_REPLAY_EQUIVALENCE"
        _expect_error(expected, dre.validate_decoder_replay_equivalence_policy, _rehash_clone(good, "_equivalence_policy_payload", "decoder_replay_equivalence_policy_hash", **{field: value}))


def test_execution_boundary_validation_rejects_runtime_and_import_authority():
    good = _parts()[2]
    cases = [
        ("execution_boundary_mode", "RUNTIME"),
        ("declared_replay_transcript_only", False),
        ("baseline_decoder_import_allowed", True),
        ("candidate_decoder_import_allowed", True),
        ("runtime_decoder_execution_allowed", True),
        ("decoder_workload_execution_allowed", True),
        ("benchmark_execution_allowed", True),
        ("network_allowed", True),
        ("heavy_backend_import_allowed", True),
        ("hardware_sdk_allowed", True),
        ("filesystem_mutation_allowed", True),
        ("candidate_promotion_allowed", True),
        ("network_allowed", 1),
    ]
    for field, value in cases:
        expected = "INVALID_INPUT" if type(value) is int else "INVALID_REPLAY_EQUIVALENCE"
        _expect_error(expected, dre.validate_decoder_replay_execution_boundary, _rehash_clone(good, "_execution_boundary_payload", "decoder_replay_execution_boundary_hash", **{field: value}))


def test_corpus_item_validation_and_receipt_duplicate_rejection():
    item = _parts()[3][0]
    cases = [
        ("record_id", "", "INVALID_INPUT"),
        ("syndrome_bits", tuple(), "INVALID_INPUT"),
        ("syndrome_bits", (0, 2), "INVALID_INPUT"),
        ("syndrome_bits", (True, 0), "INVALID_INPUT"),
        ("syndrome_input_hash", "x", "INVALID_HASH"),
        ("syndrome_input_hash", HEX_A, "HASH_MISMATCH"),
        ("syndrome_schema_hash", "A" * 64, "INVALID_HASH"),
        ("canonical_ordering_key", "", "INVALID_INPUT"),
    ]
    for field, value, code in cases:
        _expect_error(code, dre.validate_decoder_replay_corpus_item, _rehash_clone(item, "_corpus_item_payload", "decoder_replay_corpus_item_hash", **{field: value}))
    assert dre._hash_payload(dre._corpus_item_payload(item)) == item.decoder_replay_corpus_item_hash
    upstream, policy, boundary, items, _, _, comparisons, corpus_summary, coverage, _ = _parts()
    duplicate = _rehash_clone(items[1], "_corpus_item_payload", "decoder_replay_corpus_item_hash", record_id=items[0].record_id)
    _expect_error("INVALID_INPUT", dre.build_decoder_replay_equivalence_receipt, upstream, policy, boundary, corpus_summary, (items[0], duplicate), comparisons, coverage)


def test_output_record_validation():
    output = _parts()[4][0]
    cases = [
        ("record_id", "", "INVALID_INPUT"),
        ("decoder_role", "OTHER", "INVALID_INPUT"),
        ("correction_bits", tuple(), "INVALID_INPUT"),
        ("correction_bits", (0, 3), "INVALID_INPUT"),
        ("correction_bits", (False, 1), "INVALID_INPUT"),
        ("output_payload_hash", "x", "INVALID_HASH"),
        ("output_payload_hash", HEX_A, "HASH_MISMATCH"),
        ("output_schema_hash", "A" * 64, "INVALID_HASH"),
        ("output_status", "OTHER", "INVALID_INPUT"),
        ("output_ordering_key", "", "INVALID_INPUT"),
    ]
    for field, value, code in cases:
        _expect_error(code, dre.validate_decoder_replay_output_record, _rehash_clone(output, "_output_record_payload", "decoder_replay_output_record_hash", **{field: value}))
    assert dre._hash_payload(dre._output_record_payload(output)) == output.decoder_replay_output_record_hash


def test_comparison_record_validation_rejects_mismatches_and_forged_booleans():
    _, _, _, items, baseline, candidate, comparisons, *_ = _parts()
    assert dre.validate_decoder_replay_comparison_record(comparisons[0]) is comparisons[0]
    other_candidate = dre.build_decoder_replay_output_record("other", "CANDIDATE_DECODER", (1, 0, 1), HEX_E, "k")
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_comparison_record, items[0].syndrome_input_hash, baseline[0], other_candidate)
    wrong_baseline = dre.build_decoder_replay_output_record(items[0].record_id, "CANDIDATE_DECODER", items[0].syndrome_bits, HEX_E, "k")
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_comparison_record, items[0].syndrome_input_hash, wrong_baseline, candidate[0])
    wrong_candidate = dre.build_decoder_replay_output_record(items[0].record_id, "CANONICAL_BASELINE_DECODER", items[0].syndrome_bits, HEX_E, "k")
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_comparison_record, items[0].syndrome_input_hash, baseline[0], wrong_candidate)
    schema_candidate = dre.build_decoder_replay_output_record(items[0].record_id, "CANDIDATE_DECODER", items[0].syndrome_bits, HEX_D, "k")
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_comparison_record, items[0].syndrome_input_hash, baseline[0], schema_candidate)
    bit_candidate = dre.build_decoder_replay_output_record(items[0].record_id, "CANDIDATE_DECODER", (0, 0, 0), HEX_E, "k")
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_comparison_record, items[0].syndrome_input_hash, baseline[0], bit_candidate)
    for field, value in (("exact_output_match", False), ("output_schema_match", False), ("output_payload_match", False), ("mismatch_reason", "SOMETHING"), ("equivalence_mode", "CUSTOM")):
        _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.validate_decoder_replay_comparison_record, _rehash_clone(comparisons[0], "_comparison_record_payload", "decoder_replay_comparison_record_hash", **{field: value}))
    forged_payload_candidate = _rehash_clone(candidate[0], "_output_record_payload", "decoder_replay_output_record_hash", correction_bits=(0, 0, 0), output_payload_hash=baseline[0].output_payload_hash)
    forged_comparison = _rehash_clone(comparisons[0], "_comparison_record_payload", "decoder_replay_comparison_record_hash", candidate_output=forged_payload_candidate)
    _expect_error("HASH_MISMATCH", dre.validate_decoder_replay_comparison_record, forged_comparison)


def test_corpus_summary_validation():
    summary = _parts()[7]
    cases = [
        ("corpus_name", "", "INVALID_INPUT"),
        ("corpus_version", "", "INVALID_INPUT"),
        ("corpus_mode", "OTHER", "INVALID_INPUT"),
        ("syndrome_schema_hash", "x", "INVALID_HASH"),
        ("output_schema_hash", "A" * 64, "INVALID_HASH"),
        ("corpus_item_hashes", tuple(), "INVALID_INPUT"),
        ("corpus_item_hashes", (HEX_A, HEX_A), "INVALID_INPUT"),
        ("corpus_item_count", 99, "INVALID_INPUT"),
        ("replay_corpus_hash", HEX_A, "HASH_MISMATCH"),
        ("canonical_ordering_policy", "OTHER", "INVALID_INPUT"),
    ]
    for field, value, code in cases:
        _expect_error(code, dre.validate_decoder_replay_corpus_summary, _rehash_clone(summary, "_corpus_summary_payload", "decoder_replay_corpus_summary_hash", **{field: value}))


def test_coverage_summary_validation():
    coverage = _parts()[8]
    cases = [
        ("comparison_record_hashes", tuple(), "INVALID_INPUT"),
        ("comparison_record_hashes", (HEX_A, HEX_A), "INVALID_INPUT"),
        ("comparison_count", 99, "INVALID_REPLAY_EQUIVALENCE"),
        ("matched_count", 0, "INVALID_REPLAY_EQUIVALENCE"),
        ("mismatched_count", 1, "INVALID_REPLAY_EQUIVALENCE"),
        ("schema_mismatch_count", 1, "INVALID_REPLAY_EQUIVALENCE"),
        ("payload_mismatch_count", 1, "INVALID_REPLAY_EQUIVALENCE"),
        ("all_records_exact_match", False, "INVALID_REPLAY_EQUIVALENCE"),
        ("replay_equivalence_proven_for_declared_corpus", False, "INVALID_REPLAY_EQUIVALENCE"),
    ]
    for field, value, code in cases:
        _expect_error(code, dre.validate_decoder_replay_coverage_summary, _rehash_clone(coverage, "_coverage_summary_payload", "decoder_replay_coverage_summary_hash", **{field: value}))


def test_aggregate_receipt_validation_rejects_unsafe_coverage_and_claims():
    upstream, policy, boundary, items, _, _, comparisons, corpus_summary, coverage, receipt = _parts()
    for field, value in (("receipt_version", "v166.3"), ("receipt_kind", "Other"), ("previous_release_tag", "v166.0"), ("previous_release_url", "https://example.invalid")):
        _expect_error("INVALID_INPUT", dre.validate_decoder_replay_equivalence_receipt, _rehash_clone(receipt, "_receipt_payload", "decoder_replay_equivalence_receipt_hash", **{field: value}))
    _expect_error("INVALID_INPUT", dre.build_decoder_replay_equivalence_receipt, upstream, policy, boundary, corpus_summary, tuple(), comparisons, coverage)
    _expect_error("INVALID_INPUT", dre.build_decoder_replay_equivalence_receipt, upstream, policy, boundary, corpus_summary, items, tuple(), coverage)
    one_comparison = (comparisons[0],)
    one_coverage = dre.build_decoder_replay_coverage_summary(one_comparison)
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_equivalence_receipt, upstream, policy, boundary, corpus_summary, items, one_comparison, one_coverage)
    extra = _rehash_clone(comparisons[0], "_comparison_record_payload", "decoder_replay_comparison_record_hash", record_id="extra")
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_equivalence_receipt, upstream, policy, boundary, corpus_summary, items, tuple(sorted(comparisons + (extra,), key=lambda r: (r.record_id, r.syndrome_input_hash, r.decoder_replay_comparison_record_hash))), dre.build_decoder_replay_coverage_summary(comparisons + (extra,)))
    wrong_syndrome = _rehash_clone(comparisons[0], "_comparison_record_payload", "decoder_replay_comparison_record_hash", syndrome_input_hash=HEX_A)
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_equivalence_receipt, upstream, policy, boundary, corpus_summary, items, (wrong_syndrome, comparisons[1]), dre.build_decoder_replay_coverage_summary((wrong_syndrome, comparisons[1])))
    wrong_summary = dre.build_decoder_replay_corpus_summary("declared replay", "1", (items[0],), HEX_D, HEX_E)
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_equivalence_receipt, upstream, policy, boundary, wrong_summary, items, comparisons, coverage)
    wrong_coverage = dre.build_decoder_replay_coverage_summary((comparisons[0],))
    _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.build_decoder_replay_equivalence_receipt, upstream, policy, boundary, corpus_summary, items, comparisons, wrong_coverage)
    for field in ("promotion_allowed", "benchmark_claim_allowed", "global_correctness_claim_allowed"):
        _expect_error("INVALID_REPLAY_EQUIVALENCE", dre.validate_decoder_replay_equivalence_receipt, _rehash_clone(receipt, "_receipt_payload", "decoder_replay_equivalence_receipt_hash", **{field: True}))
    forged_receipt = _rehash_clone(receipt, "_receipt_payload", "decoder_replay_equivalence_receipt_hash", replay_equivalence_proven=0)
    _expect_error("INVALID_INPUT", dre.validate_decoder_replay_equivalence_receipt, forged_receipt)


def test_forbidden_semantic_hardening_and_positive_control():
    rejected = [
        "silent_decoder_replacement",
        "candidate-replaces-baseline",
        "decoder   replaced\nbecause   faster",
        "speed proves correctness",
        "benchmark proves correctness",
        "benchmark marketing",
        "runtime promotion",
        "candidate decoder promoted",
        "probabilistic decoder authority",
        "ML decoder authority",
        "hardware authority",
        "QEC advantage proven",
        "hidden precision drift",
        "undeclared approximation policy",
        "output accepted as universal canonical truth",
        "global correctness proven",
        "replay equivalence implies promotion",
        "replay\\nequivalence__implies--speedup",
    ]
    for phrase in rejected:
        _expect_error("INVALID_INPUT", dre.build_decoder_replay_upstream_binding, HEX_A, HEX_B, HEX_C, phrase, "0.1.0")
    dre._check_forbidden_declaration_semantics("replay_equivalence_proven_for_declared_corpus", "positive_control")


def test_boundary_static_guards_no_decoder_imports_heavy_imports_network_or_runtime_markers():
    module_path = Path("src/qec/analysis/decoder_replay_equivalence_receipts.py")
    test_path = Path(__file__)
    tree = ast.parse(module_path.read_text())
    test_tree = ast.parse(test_path.read_text())
    heavy = {"numpy", "scipy", "qldpc", "stim", "pymatching", "qiskit", "qutip", "pandas", "polars", "torch", "tensorflow", "jax"}
    forbidden_modules = heavy | {"qec.decoder", "importlib", "socket", "urllib", "requests", "http", "subprocess"}
    for parsed in (tree, test_tree):
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "qec.decoder" or alias.name.split(".")[0] in forbidden_modules:
                        # subprocess is allowed only in this test file for hash-seed stability.
                        if alias.name == "subprocess" and parsed is test_tree:
                            continue
                        raise AssertionError(f"forbidden import: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "qec.decoder" or module.split(".")[0] in forbidden_modules:
                    raise AssertionError(f"forbidden import-from: {module}")
    text = module_path.read_text().lower()
    assert "candidate_decoders/" not in text
    assert "benchmark(" not in text
    assert "run_decoder" not in text
    assert "execute_decoder" not in text
    assert "cuda" not in text


def test_hash_seed_stability_subprocess():
    script = """
from qec.analysis import decoder_replay_equivalence_receipts as dre
H='a'*64; I='b'*64; J='c'*64; S='d'*64; O='e'*64
u=dre.build_decoder_replay_upstream_binding(H,I,J,'candidate alpha','0.1.0')
p=dre.build_decoder_replay_equivalence_policy(); b=dre.build_decoder_replay_execution_boundary()
i1=dre.build_decoder_replay_corpus_item('r2',(1,0),S,'k2'); i2=dre.build_decoder_replay_corpus_item('r1',(0,1),S,'k1')
items=(i1,i2)
cs=[]
for item in items:
    x=dre.build_decoder_replay_output_record(item.record_id,'CANONICAL_BASELINE_DECODER',item.syndrome_bits,O,item.canonical_ordering_key)
    y=dre.build_decoder_replay_output_record(item.record_id,'CANDIDATE_DECODER',item.syndrome_bits,O,item.canonical_ordering_key)
    cs.append(dre.build_decoder_replay_comparison_record(item.syndrome_input_hash,x,y))
s=dre.build_decoder_replay_corpus_summary('declared replay','1',items,S,O)
c=dre.build_decoder_replay_coverage_summary(tuple(cs))
r=dre.build_decoder_replay_equivalence_receipt(u,p,b,s,items,tuple(cs),c)
print(r.decoder_replay_equivalence_receipt_hash)
"""
    outputs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}
        result = subprocess.run([sys.executable, "-c", script], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]
    assert dre._HASH_RE.fullmatch(outputs[0])
