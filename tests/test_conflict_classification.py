from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.conflict_classification import (
    ConflictReceipt,
    classify_decision_conflict,
)


def _payload(items: tuple[tuple[str, object], ...]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(items, key=lambda item: item[0]))


def _decision(payload: tuple[tuple[str, object], ...]) -> tuple[str, tuple[tuple[str, object], ...]]:
    return sha256_hex(payload), payload


def test_deterministic_replay_100_runs() -> None:
    p1 = _payload((('a', 1), ('b', 2)))
    p2 = _payload((('a', 1), ('b', 2), ('c', 3)))
    h1, _ = _decision(p1)
    h2, _ = _decision(p2)
    memory = tuple(sorted((sha256_hex('m1'), sha256_hex('m2'))))

    baseline = classify_decision_conflict(memory, h1, p1, h2, p2)
    for _ in range(100):
        replay = classify_decision_conflict(memory, h1, p1, h2, p2)
        assert replay == baseline
        assert replay.conflict_hash == baseline.conflict_hash


def test_identical_classification() -> None:
    p = _payload((('x', 7),))
    h, _ = _decision(p)
    receipt = classify_decision_conflict((sha256_hex('mem'),), h, p, h, p)
    assert receipt.comparison.conflict_type == 'IDENTICAL'


def test_equivalent_classification_same_payload_different_hashes() -> None:
    p = _payload((('k', 'v'), ('n', 1)))
    h1 = sha256_hex(p)
    h2 = 'f' * 64
    receipt = classify_decision_conflict((sha256_hex('mem'),), h1, p, h2, p)
    assert receipt.comparison.conflict_type == 'EQUIVALENT'


def test_dominated_classification_subset() -> None:
    p_small = _payload((('a', 1),))
    p_large = _payload((('a', 1), ('b', 2)))
    h1, _ = _decision(p_small)
    h2, _ = _decision(p_large)
    receipt = classify_decision_conflict((sha256_hex('mem'),), h1, p_small, h2, p_large)
    assert receipt.comparison.conflict_type == 'DOMINATED'


def test_inconsistent_classification() -> None:
    p1 = _payload((('a', 1), ('b', 9)))
    p2 = _payload((('a', 2), ('c', 3)))
    h1, _ = _decision(p1)
    h2, _ = _decision(p2)
    receipt = classify_decision_conflict((sha256_hex('mem'),), h1, p1, h2, p2)
    assert receipt.comparison.conflict_type == 'INCONSISTENT'


def test_symmetry_of_classification_and_receipt() -> None:
    p1 = _payload((('a', 1), ('b', 2)))
    p2 = _payload((('a', 1), ('b', 2), ('c', 3)))
    h1, _ = _decision(p1)
    h2, _ = _decision(p2)
    memory = tuple(sorted((sha256_hex('m1'), sha256_hex('m2'))))
    left = classify_decision_conflict(memory, h1, p1, h2, p2)
    right = classify_decision_conflict(memory, h2, p2, h1, p1)
    assert left == right


def test_invalid_hash_fails() -> None:
    p = _payload((('x', 1),))
    with pytest.raises(ValueError, match='INVALID_INPUT'):
        classify_decision_conflict((sha256_hex('mem'),), 'abc', p, 'f' * 64, p)


def test_payload_hash_mismatch_fails() -> None:
    p = _payload((('x', 1),))
    good_hash, _ = _decision(p)
    mismatch = _payload((('x', 2),))
    with pytest.raises(ValueError, match='INVALID_INPUT'):
        classify_decision_conflict((sha256_hex('mem'),), good_hash, mismatch, good_hash, p)


def test_canonical_identity_enforcement() -> None:
    p = _payload((('x', 1),))
    h, _ = _decision(p)
    unsorted_memory = (sha256_hex('a'), sha256_hex('a'))
    with pytest.raises(ValueError, match='INVALID_INPUT'):
        classify_decision_conflict(unsorted_memory, h, p, h, p)


def test_immutability() -> None:
    p = _payload((('x', 1),))
    h, _ = _decision(p)
    receipt = classify_decision_conflict((sha256_hex('mem'),), h, p, h, p)
    with pytest.raises(FrozenInstanceError):
        receipt.conflict_hash = '0' * 64


def test_hash_recomputation_stability() -> None:
    p1 = _payload((('a', 1),))
    p2 = _payload((('a', 1), ('b', 2)))
    h1, _ = _decision(p1)
    h2, _ = _decision(p2)
    memory = tuple(sorted((sha256_hex('m1'), sha256_hex('m2'))))
    receipt = classify_decision_conflict(memory, h1, p1, h2, p2)

    tampered_hash = '0' * 64
    with pytest.raises(ValueError, match='INVALID_INPUT'):
        ConflictReceipt(
            comparison=receipt.comparison,
            input_memory_hashes=receipt.input_memory_hashes,
            conflict_hash=tampered_hash,
        )
