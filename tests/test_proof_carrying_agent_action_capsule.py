"""Tests for v137.18.0 — Proof-Carrying Agent Action Capsule."""

from __future__ import annotations

import dataclasses
import json

import pytest

from qec.orchestration.proof_carrying_agent_action_capsule import (
    AgentActionProofObligation,
    AgentActionProofReceipt,
    AgentActionValidationReport,
    ProofCarryingAgentActionCapsule,
    SUPPORTED_ACTION_TYPES,
    SUPPORTED_OBLIGATION_KINDS,
    SUPPORTED_VALIDATION_FLAGS,
    build_action_proof_receipt,
    build_proof_carrying_agent_action_capsule,
    certify_agent_action_capsule,
    compare_action_capsule_replay,
    validate_agent_action_capsule,
)
from qec.orchestration.proof_carrying_agent_action_capsule import (
    _normalize_receipt,
)


def _sample_obligations():
    return (
        AgentActionProofObligation(
            obligation_id="ob_invariant_schema",
            obligation_kind="invariant",
            obligation_statement="schema must be stable",
            obligation_scope="ledger",
            obligation_epoch=1,
        ),
        AgentActionProofObligation(
            obligation_id="ob_pre_bounded",
            obligation_kind="precondition",
            obligation_statement="input bounded",
            obligation_scope="ledger",
            obligation_epoch=0,
        ),
        AgentActionProofObligation(
            obligation_id="ob_determinism",
            obligation_kind="determinism",
            obligation_statement="byte-identical replay",
            obligation_scope="ledger",
            obligation_epoch=0,
        ),
    )


def _sample_payload():
    return {
        "target_ledger": "ledger_v137_17_5",
        "mode": "canonical",
        "tags": ["deterministic", "replay_safe"],
    }


def _build_sample_capsule(**overrides):
    obligations = overrides.pop("proof_obligations", _sample_obligations())
    kwargs = dict(
        action_id="action_v137_18_0_observe",
        action_type="observe",
        action_scope="orchestration/ledger",
        action_payload=_sample_payload(),
        preconditions=("ledger_exists", "ledger_certified"),
        invariants=("determinism:byte-identical", "safety:no-mutation"),
        proof_obligations=obligations,
        validation_flags=("deterministic_only", "replay_safe"),
    )
    kwargs.update(overrides)
    return build_proof_carrying_agent_action_capsule(**kwargs)


# ---------------------------------------------------------------------------
# Construction + determinism
# ---------------------------------------------------------------------------


def test_deterministic_repeated_build():
    a = _build_sample_capsule()
    b = _build_sample_capsule()
    assert a == b
    assert a.capsule_hash == b.capsule_hash
    assert a.replay_identity == b.replay_identity
    assert a.to_canonical_json() == b.to_canonical_json()


def test_stable_hash_reproducibility():
    capsule = _build_sample_capsule()
    first = capsule.stable_hash()
    second = capsule.stable_hash()
    assert first == second == capsule.capsule_hash
    assert len(capsule.capsule_hash) == 64
    # Stable hash matches canonical-byte SHA-256 of body (minus self-hash).
    assert capsule.replay_identity.endswith(capsule.capsule_hash[:16])


def test_proof_obligation_ordering_is_canonical():
    capsule = _build_sample_capsule()
    # Ordering: (epoch, kind, obligation_id) ascending
    expected_order = (
        "ob_determinism",   # epoch 0, kind "determinism"
        "ob_pre_bounded",   # epoch 0, kind "precondition"
        "ob_invariant_schema",  # epoch 1, kind "invariant"
    )
    actual = tuple(ob.obligation_id for ob in capsule.proof_obligations)
    assert actual == expected_order


def test_canonical_byte_equality_and_json_stability():
    capsule = _build_sample_capsule()
    json_text = capsule.to_canonical_json()
    # Robust canonical-form check: parse and re-dump with canonical
    # separators; the round-trip must be byte-identical to the original.
    parsed = json.loads(json_text)
    roundtrip = json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert roundtrip == json_text
    # Rebuilding should yield identical bytes.
    rebuilt = _build_sample_capsule()
    assert rebuilt.to_canonical_json().encode("utf-8") == json_text.encode("utf-8")


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def test_empty_action_id_rejected():
    with pytest.raises(ValueError, match="action_id"):
        _build_sample_capsule(action_id="   ")


def test_unsupported_action_type_rejected():
    with pytest.raises(ValueError, match="unsupported action_type"):
        _build_sample_capsule(action_type="execute")


def test_duplicate_proof_obligation_rejected():
    dup = (
        AgentActionProofObligation(
            obligation_id="ob_same",
            obligation_kind="precondition",
            obligation_statement="s1",
            obligation_scope="ledger",
            obligation_epoch=0,
        ),
        AgentActionProofObligation(
            obligation_id="ob_same",
            obligation_kind="invariant",
            obligation_statement="s2",
            obligation_scope="ledger",
            obligation_epoch=0,
        ),
    )
    with pytest.raises(ValueError, match="duplicate proof obligation"):
        _build_sample_capsule(proof_obligations=dup)


def test_malformed_payload_rejected():
    with pytest.raises(ValueError, match="action_payload"):
        _build_sample_capsule(action_payload=["not", "a", "mapping"])


def test_invalid_invariant_structure_rejected():
    with pytest.raises(ValueError, match="invalid invariant structure"):
        _build_sample_capsule(invariants=("missing_colon",))


def test_unsorted_receipt_chain_rejected():
    capsule = _build_sample_capsule()
    r0 = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=0
    )
    r1 = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=1
    )
    with pytest.raises(ValueError, match="unsorted proof receipt chain"):
        _build_sample_capsule(receipt_chain=(r1, r0))


def test_validation_report_determinism():
    capsule = _build_sample_capsule()
    report_a = validate_agent_action_capsule(capsule)
    report_b = validate_agent_action_capsule(capsule)
    assert isinstance(report_a, AgentActionValidationReport)
    assert report_a == report_b
    assert report_a.is_valid is True
    assert report_a.violations == ()
    assert report_a.report_hash == report_b.report_hash


# ---------------------------------------------------------------------------
# Certification + replay
# ---------------------------------------------------------------------------


def test_certification_reproducibility():
    capsule = _build_sample_capsule()
    receipt_a = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=0
    )
    capsule_with_receipt = _build_sample_capsule(receipt_chain=(receipt_a,))
    report_1 = certify_agent_action_capsule(capsule_with_receipt)
    report_2 = certify_agent_action_capsule(capsule_with_receipt)
    assert report_1.report_hash == report_2.report_hash
    assert report_1.is_valid
    assert capsule_with_receipt.receipt_chain[0].action_id == capsule_with_receipt.action_id


def test_replay_equality_and_drift_failure():
    capsule_a = _build_sample_capsule()
    capsule_b = _build_sample_capsule()
    assert compare_action_capsule_replay(capsule_a, capsule_b) is True

    # Drift: change action_scope
    capsule_c = _build_sample_capsule(action_scope="orchestration/audit")
    with pytest.raises(ValueError, match="replay drift"):
        compare_action_capsule_replay(capsule_a, capsule_c)


def test_receipt_tamper_detection():
    capsule = _build_sample_capsule()
    receipt = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=0
    )
    capsule_with_receipt = _build_sample_capsule(receipt_chain=(receipt,))

    # Tamper by replacing the receipt's receipt_hash with a fabricated value.
    tampered_receipt = dataclasses.replace(
        capsule_with_receipt.receipt_chain[0], receipt_hash="0" * 64
    )
    tampered_capsule = dataclasses.replace(
        capsule_with_receipt, receipt_chain=(tampered_receipt,)
    )
    report = validate_agent_action_capsule(tampered_capsule)
    assert report.is_valid is False
    # Tampering the receipt_hash must invalidate the receipt binding.
    assert any("receipt mismatch" in v for v in report.violations)
    with pytest.raises(ValueError, match="failed certification"):
        certify_agent_action_capsule(tampered_capsule)

    # Independently, tampering the canonical body drifts the capsule hash.
    body_tampered = dataclasses.replace(
        capsule_with_receipt, action_scope="orchestration/other"
    )
    body_report = validate_agent_action_capsule(body_tampered)
    assert body_report.is_valid is False
    assert any("replay drift" in v for v in body_report.violations)


def test_all_supported_action_types_are_constructible():
    built_hashes = set()
    for action_type in SUPPORTED_ACTION_TYPES:
        cap = _build_sample_capsule(
            action_id=f"action_{action_type}",
            action_type=action_type,
        )
        assert cap.action_type == action_type
        built_hashes.add(cap.capsule_hash)
    # Each distinct action yields a distinct canonical hash.
    assert len(built_hashes) == len(SUPPORTED_ACTION_TYPES)


def test_unsupported_validation_flag_rejected():
    with pytest.raises(ValueError, match="unsupported validation_flag"):
        _build_sample_capsule(validation_flags=("deterministic_only", "mutation_allowed"))


def test_unsupported_obligation_kind_rejected():
    bad = (
        AgentActionProofObligation(
            obligation_id="ob_bad",
            obligation_kind="heuristic",
            obligation_statement="stmt",
            obligation_scope="ledger",
            obligation_epoch=0,
        ),
    )
    with pytest.raises(ValueError, match="unsupported obligation_kind"):
        _build_sample_capsule(proof_obligations=bad)


def test_canonical_constants_are_sorted_tuples():
    assert isinstance(SUPPORTED_ACTION_TYPES, tuple)
    assert tuple(sorted(SUPPORTED_ACTION_TYPES)) == SUPPORTED_ACTION_TYPES
    assert tuple(sorted(SUPPORTED_OBLIGATION_KINDS)) == SUPPORTED_OBLIGATION_KINDS
    assert tuple(sorted(SUPPORTED_VALIDATION_FLAGS)) == SUPPORTED_VALIDATION_FLAGS


def test_receipt_chain_with_multiple_epochs_validates():
    capsule = _build_sample_capsule()
    r0 = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=0
    )
    r1 = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=1
    )
    capsule_with_chain = _build_sample_capsule(receipt_chain=(r0, r1))
    report = certify_agent_action_capsule(capsule_with_chain)
    assert report.is_valid
    assert len(capsule_with_chain.receipt_chain) == 2
    assert capsule_with_chain.receipt_chain[0].receipt_epoch == 0
    assert capsule_with_chain.receipt_chain[1].receipt_epoch == 1


# ---------------------------------------------------------------------------
# v137.18.0 hardening tests
# ---------------------------------------------------------------------------


def _valid_receipt_mapping(capsule):
    receipt = build_action_proof_receipt(
        capsule.action_id,
        capsule.capsule_hash,
        capsule.proof_obligations,
        receipt_epoch=0,
    )
    return receipt.to_dict()


def test_build_action_proof_receipt_rejects_bool_epoch():
    capsule = _build_sample_capsule()
    with pytest.raises(ValueError, match="receipt_epoch must be a non-negative integer"):
        build_action_proof_receipt(
            capsule.action_id,
            capsule.capsule_hash,
            capsule.proof_obligations,
            receipt_epoch=True,
        )


def test_build_action_proof_receipt_rejects_negative_epoch():
    capsule = _build_sample_capsule()
    with pytest.raises(ValueError, match="receipt_epoch must be a non-negative integer"):
        build_action_proof_receipt(
            capsule.action_id,
            capsule.capsule_hash,
            capsule.proof_obligations,
            receipt_epoch=-1,
        )


def test_build_action_proof_receipt_rejects_non_hex_capsule_hash():
    capsule = _build_sample_capsule()
    with pytest.raises(ValueError, match="capsule_hash must be a 64-char lowercase hex"):
        build_action_proof_receipt(
            capsule.action_id,
            "not-a-valid-hash",
            capsule.proof_obligations,
            receipt_epoch=0,
        )


def test_normalize_receipt_rejects_bool_epoch():
    capsule = _build_sample_capsule()
    mapping = _valid_receipt_mapping(capsule)
    mapping["receipt_epoch"] = True
    with pytest.raises(ValueError, match="receipt_epoch must be a non-negative integer"):
        _normalize_receipt(mapping)


def test_normalize_receipt_rejects_negative_epoch():
    capsule = _build_sample_capsule()
    mapping = _valid_receipt_mapping(capsule)
    mapping["receipt_epoch"] = -3
    with pytest.raises(ValueError, match="receipt_epoch must be a non-negative integer"):
        _normalize_receipt(mapping)


def test_normalize_receipt_rejects_string_epoch():
    capsule = _build_sample_capsule()
    mapping = _valid_receipt_mapping(capsule)
    mapping["receipt_epoch"] = "0"
    with pytest.raises(ValueError, match="receipt_epoch must be a non-negative integer"):
        _normalize_receipt(mapping)


def test_normalize_receipt_rejects_none_obligation_hashes():
    capsule = _build_sample_capsule()
    mapping = _valid_receipt_mapping(capsule)
    mapping["obligation_hashes"] = None
    with pytest.raises(ValueError, match="obligation_hashes must be a non-scalar iterable"):
        _normalize_receipt(mapping)


def test_normalize_receipt_rejects_scalar_obligation_hashes():
    capsule = _build_sample_capsule()
    mapping = _valid_receipt_mapping(capsule)
    mapping["obligation_hashes"] = "deadbeef" * 8  # a bare string scalar
    with pytest.raises(ValueError, match="obligation_hashes must be a non-scalar iterable"):
        _normalize_receipt(mapping)


def test_normalize_receipt_rejects_empty_obligation_hash_entry():
    capsule = _build_sample_capsule()
    mapping = _valid_receipt_mapping(capsule)
    mapping["obligation_hashes"] = ["   "]
    with pytest.raises(ValueError, match="obligation_hashes entries must be non-empty strings"):
        _normalize_receipt(mapping)


def test_normalize_receipt_rejects_non_hex_obligation_hash_entry():
    capsule = _build_sample_capsule()
    mapping = _valid_receipt_mapping(capsule)
    mapping["obligation_hashes"] = ["not_a_hash"]
    with pytest.raises(ValueError, match="64-char lowercase hex"):
        _normalize_receipt(mapping)


def test_validator_never_raises_on_malformed_receipt():
    capsule = _build_sample_capsule()
    receipt = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=0
    )
    capsule_with_receipt = _build_sample_capsule(receipt_chain=(receipt,))
    # Tamper with the receipt_hash to a non-hex value. The validator must
    # still return a deterministic report instead of raising.
    tampered = dataclasses.replace(
        capsule_with_receipt.receipt_chain[0], receipt_hash="not-a-hash"
    )
    tampered_capsule = dataclasses.replace(
        capsule_with_receipt, receipt_chain=(tampered,)
    )
    report = validate_agent_action_capsule(tampered_capsule)
    assert isinstance(report, AgentActionValidationReport)
    assert report.is_valid is False
    assert any("receipt validation failed" in v for v in report.violations)


def test_validator_never_raises_on_malformed_payload():
    capsule = _build_sample_capsule()
    # Tamper the payload to a non-mapping. Body recompute would raise —
    # the validator must catch and convert to a violation.
    tampered = dataclasses.replace(capsule, action_payload=["not", "a", "mapping"])
    report = validate_agent_action_capsule(tampered)
    assert isinstance(report, AgentActionValidationReport)
    assert report.is_valid is False
    assert any("malformed payload" in v for v in report.violations)


def test_validator_never_raises_on_non_hex_capsule_hash():
    capsule = _build_sample_capsule()
    tampered = dataclasses.replace(capsule, capsule_hash="not-a-hash")
    report = validate_agent_action_capsule(tampered)
    assert isinstance(report, AgentActionValidationReport)
    assert report.is_valid is False
    assert any("malformed capsule_hash" in v for v in report.violations)


def test_non_string_payload_keys_rejected():
    with pytest.raises(ValueError, match="payload keys must be strings"):
        _build_sample_capsule(action_payload={1: "one", "two": 2})


def test_empty_payload_keys_rejected():
    with pytest.raises(ValueError, match="payload keys must be non-empty strings"):
        _build_sample_capsule(action_payload={"": "empty"})


def test_whitespace_only_payload_keys_rejected():
    with pytest.raises(ValueError, match="payload keys must be non-empty strings"):
        _build_sample_capsule(action_payload={"   ": "whitespace"})


def test_unsorted_violation_wording_exact_match():
    capsule = _build_sample_capsule()
    r0 = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=0
    )
    r1 = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=1
    )
    # Build via a valid path, then splice into unsorted order.
    good = _build_sample_capsule(receipt_chain=(r0, r1))
    tampered = dataclasses.replace(good, receipt_chain=(r1, r0))
    report = validate_agent_action_capsule(tampered)
    assert "unsorted proof receipt chain" in report.violations
    # Constructor path must raise with the same wording.
    with pytest.raises(ValueError, match=r"^unsorted proof receipt chain$"):
        _build_sample_capsule(receipt_chain=(r1, r0))


def test_validator_never_raises_full_fuzz_matrix():
    """The validator must never raise under any combination of tampered fields."""
    capsule = _build_sample_capsule()
    good_receipt = build_action_proof_receipt(
        capsule.action_id, capsule.capsule_hash, capsule.proof_obligations, receipt_epoch=0
    )
    base = _build_sample_capsule(receipt_chain=(good_receipt,))

    mutations = (
        {"action_id": ""},
        {"action_type": "execute"},
        {"action_scope": ""},
        {"action_payload": ["not", "a", "mapping"]},
        {"capsule_hash": ""},
        {"capsule_hash": "zz" * 32},
        {"replay_identity": "agent-action::observe::deadbeef"},
        {
            "receipt_chain": (
                dataclasses.replace(good_receipt, receipt_hash="0" * 63),
            )
        },
        {
            "receipt_chain": (
                dataclasses.replace(good_receipt, obligation_hashes=("bad",)),
            )
        },
    )
    for mutation in mutations:
        tampered = dataclasses.replace(base, **mutation)
        report = validate_agent_action_capsule(tampered)
        assert isinstance(report, AgentActionValidationReport)
        assert report.is_valid is False
