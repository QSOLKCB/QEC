from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.decay_checkpoint_contract import (
    build_decay_checkpoint,
    build_decay_checkpoint_set,
)
from qec.analysis.digital_decay_signature import (
    DecayThresholdContract,
    DigitalDecaySignature,
    build_decay_threshold_contract,
    build_digital_decay_signature,
    validate_digital_decay_signature,
)

_H0 = "0" * 64
_H1 = "1" * 64
_H2 = "2" * 64


def _checkpoint_set_with_drift_count(drift_count: int, total: int = 8):
    checkpoints = []
    for i in range(total):
        expected = _H0
        observed = _H1 if i < drift_count else _H0
        checkpoints.append(build_decay_checkpoint(f"p{i}", expected, observed))
    return build_decay_checkpoint_set(checkpoints)


def test_decay_threshold_contract_determinism():
    hashes = {build_decay_threshold_contract(2, 5).threshold_contract_hash for _ in range(10)}
    assert len(hashes) == 1


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_invalid_degraded_threshold_rejected(bad):
    with pytest.raises(ValueError, match="INVALID_THRESHOLD"):
        build_decay_threshold_contract(bad, 5)


def test_invalid_threshold_order_rejected():
    with pytest.raises(ValueError, match="INVALID_THRESHOLD_ORDER"):
        build_decay_threshold_contract(3, 3)


def test_threshold_contract_hash_tamper_detected():
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        DecayThresholdContract(2, 5, "a" * 64)


def test_threshold_contract_malformed_hash_rejected():
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        DecayThresholdContract(2, 5, "xyz")


def test_decay_class_clean_at_zero():
    sig = build_digital_decay_signature(_checkpoint_set_with_drift_count(0), build_decay_threshold_contract(2, 4), [])
    assert sig.decay_class == "CLEAN"


def test_decay_class_degraded_at_lower_boundary():
    sig = build_digital_decay_signature(_checkpoint_set_with_drift_count(1), build_decay_threshold_contract(2, 4), ["p0"])
    assert sig.decay_class == "DEGRADED"


def test_decay_class_degraded_at_threshold():
    sig = build_digital_decay_signature(_checkpoint_set_with_drift_count(2), build_decay_threshold_contract(2, 4), ["p0"])
    assert sig.decay_class == "DEGRADED"


def test_decay_class_corrupted_above_degraded_threshold():
    sig = build_digital_decay_signature(
        _checkpoint_set_with_drift_count(3), build_decay_threshold_contract(2, 4), ["p0", "p1"]
    )
    assert sig.decay_class == "CORRUPTED"


def test_decay_class_corrupted_at_threshold():
    sig = build_digital_decay_signature(
        _checkpoint_set_with_drift_count(4), build_decay_threshold_contract(2, 4), ["p0", "p1", "p2"]
    )
    assert sig.decay_class == "CORRUPTED"


def test_decay_class_adversarial_above_corrupted_threshold():
    sig = build_digital_decay_signature(
        _checkpoint_set_with_drift_count(5), build_decay_threshold_contract(2, 4), ["p0", "p1", "p2", "p3"]
    )
    assert sig.decay_class == "ADVERSARIAL"


def test_digital_decay_signature_determinism():
    cps = _checkpoint_set_with_drift_count(3)
    tc = build_decay_threshold_contract(2, 4)
    hashes = {build_digital_decay_signature(cps, tc, ["p2", "p0", "p1"]).digital_decay_signature_hash for _ in range(10)}
    assert len(hashes) == 1


def test_adversarial_positions_sorted_deterministically():
    cps = _checkpoint_set_with_drift_count(3)
    tc = build_decay_threshold_contract(2, 4)
    s1 = build_digital_decay_signature(cps, tc, ["p2", "p0", "p1"])
    s2 = build_digital_decay_signature(cps, tc, ["p1", "p2", "p0"])
    assert s1.adversarial_positions == ("p0", "p1", "p2")
    assert s1.adversarial_positions == s2.adversarial_positions
    assert s1.digital_decay_signature_hash == s2.digital_decay_signature_hash


def test_adversarial_positions_must_be_drifted_positions():
    cps = _checkpoint_set_with_drift_count(1)
    tc = build_decay_threshold_contract(2, 4)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_digital_decay_signature(cps, tc, ["missing"])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_digital_decay_signature(cps, tc, ["p2"])




def test_builder_rejects_invalid_adversarial_position_item_types() -> None:
    cps = _checkpoint_set_with_drift_count(2)
    tc = build_decay_threshold_contract(2, 4)
    for bad in ([1], [""], [[]], [1, "p0"]):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            build_digital_decay_signature(cps, tc, bad)


def test_direct_signature_rejects_invalid_adversarial_position_tuple_contents() -> None:
    cps = _checkpoint_set_with_drift_count(2)
    tc = build_decay_threshold_contract(2, 4)
    for bad in (("p0", 1), ("",)):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            DigitalDecaySignature(
                checkpoint_set_hash=cps.checkpoint_set_hash,
                threshold_contract_hash=tc.threshold_contract_hash,
                decay_class="DEGRADED",
                decay_score=2,
                adversarial_positions=bad,
                digital_decay_signature_hash="0" * 64,
            )


def test_duplicate_adversarial_positions_rejected():
    cps = _checkpoint_set_with_drift_count(2)
    tc = build_decay_threshold_contract(2, 4)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_digital_decay_signature(cps, tc, ["p0", "p0"])


def test_direct_unsorted_adversarial_positions_rejected():
    cps = _checkpoint_set_with_drift_count(2)
    tc = build_decay_threshold_contract(2, 4)
    good = build_digital_decay_signature(cps, tc, ["p0", "p1"])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DigitalDecaySignature(
            checkpoint_set_hash=good.checkpoint_set_hash,
            threshold_contract_hash=good.threshold_contract_hash,
            decay_class=good.decay_class,
            decay_score=good.decay_score,
            adversarial_positions=("p1", "p0"),
            digital_decay_signature_hash=good.digital_decay_signature_hash,
        )


def test_invalid_decay_class_rejected():
    cps = _checkpoint_set_with_drift_count(2)
    tc = build_decay_threshold_contract(2, 4)
    good = build_digital_decay_signature(cps, tc, ["p0"])
    with pytest.raises(ValueError, match="INVALID_DECAY_CLASS"):
        DigitalDecaySignature(
            checkpoint_set_hash=good.checkpoint_set_hash,
            threshold_contract_hash=good.threshold_contract_hash,
            decay_class="UNKNOWN",
            decay_score=good.decay_score,
            adversarial_positions=good.adversarial_positions,
            digital_decay_signature_hash=good.digital_decay_signature_hash,
        )


def test_signature_hash_tamper_detected():
    cps = _checkpoint_set_with_drift_count(2)
    tc = build_decay_threshold_contract(2, 4)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        DigitalDecaySignature(
            checkpoint_set_hash=cps.checkpoint_set_hash,
            threshold_contract_hash=tc.threshold_contract_hash,
            decay_class="DEGRADED",
            decay_score=2,
            adversarial_positions=("p0",),
            digital_decay_signature_hash="b" * 64,
        )


def test_signature_malformed_hash_rejected():
    cps = _checkpoint_set_with_drift_count(2)
    tc = build_decay_threshold_contract(2, 4)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        DigitalDecaySignature(
            checkpoint_set_hash=cps.checkpoint_set_hash,
            threshold_contract_hash=tc.threshold_contract_hash,
            decay_class="DEGRADED",
            decay_score=2,
            adversarial_positions=("p0",),
            digital_decay_signature_hash="x",
        )


def test_validate_digital_decay_signature_returns_true_for_valid_signature():
    sig = build_digital_decay_signature(_checkpoint_set_with_drift_count(2), build_decay_threshold_contract(2, 4), ["p0"])
    assert validate_digital_decay_signature(sig) is True


def test_validate_digital_decay_signature_rejects_tampered_signature():
    sig = build_digital_decay_signature(_checkpoint_set_with_drift_count(2), build_decay_threshold_contract(2, 4), ["p0"])
    object.__setattr__(sig, "decay_score", 3)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_digital_decay_signature(sig)


def test_no_float_probability_or_threshold_fuzzing():
    with pytest.raises(ValueError, match="INVALID_THRESHOLD"):
        build_decay_threshold_contract(1.0, 3)
    with pytest.raises(ValueError, match="INVALID_THRESHOLD"):
        build_decay_threshold_contract(True, 3)

    cps = _checkpoint_set_with_drift_count(1)
    tc = build_decay_threshold_contract(2, 4)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DigitalDecaySignature(
            checkpoint_set_hash=cps.checkpoint_set_hash,
            threshold_contract_hash=tc.threshold_contract_hash,
            decay_class="DEGRADED",
            decay_score=1.0,
            adversarial_positions=("p0",),
            digital_decay_signature_hash="0" * 64,
        )

    sig = build_digital_decay_signature(cps, tc, ["p0"])
    assert not hasattr(sig, "probability")
    assert not hasattr(sig, "entropy_rate")
    assert not hasattr(sig, "entropy_drift_receipt_hash")
    assert not hasattr(sig, "decay_resistance_proof_hash")
    assert not hasattr(sig, "subsystem_receipts")


def test_artifacts_are_frozen():
    tc = build_decay_threshold_contract(2, 4)
    sig = build_digital_decay_signature(_checkpoint_set_with_drift_count(1), tc, ["p0"])
    with pytest.raises(FrozenInstanceError):
        tc.degraded_threshold = 9
    with pytest.raises(FrozenInstanceError):
        sig.decay_class = "CLEAN"


def test_same_process_determinism():
    cps = _checkpoint_set_with_drift_count(3)
    tc = build_decay_threshold_contract(2, 4)
    s1 = build_digital_decay_signature(cps, tc, ["p2", "p1", "p0"])
    s2 = build_digital_decay_signature(cps, tc, ["p2", "p1", "p0"])
    assert s1.to_canonical_json() == s2.to_canonical_json()
    assert s1.to_canonical_bytes() == s2.to_canonical_bytes()
    assert s1.digital_decay_signature_hash == s2.digital_decay_signature_hash
