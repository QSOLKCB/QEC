from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.canonical_identity import canonical_hash_identity
import re

CONFLICT_TYPE = frozenset({"IDENTICAL", "EQUIVALENT", "DOMINATED", "INCONSISTENT"})


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")


def _require_sha256_hex(value: str) -> str:
    if not isinstance(value, str) or _SHA256_HEX_RE.fullmatch(value) is None:
        raise _invalid_input()
    return value


def _canonical_payload(payload: tuple[tuple[str, Any], ...]) -> tuple[tuple[str, Any], ...]:
    if not isinstance(payload, tuple):
        raise _invalid_input()
    canonicalized: list[tuple[str, Any]] = []
    for item in payload:
        if not isinstance(item, tuple) or len(item) != 2:
            raise _invalid_input()
        key, value = item
        if not isinstance(key, str):
            raise _invalid_input()
        canonicalized.append((key, value))
    candidate = tuple(canonicalized)
    if candidate != tuple(sorted(candidate, key=lambda pair: pair[0])):
        raise _invalid_input()
    keys = tuple(key for key, _ in candidate)
    if len(keys) != len(set(keys)):
        raise _invalid_input()
    return candidate


def _payload_hash(payload: tuple[tuple[str, Any], ...]) -> str:
    return sha256_hex(payload)


def _classify(payload_a: tuple[tuple[str, Any], ...], payload_b: tuple[tuple[str, Any], ...], hash_a: str, hash_b: str) -> str:
    if hash_a == hash_b:
        return "IDENTICAL"
    if canonical_json(payload_a) == canonical_json(payload_b):
        return "EQUIVALENT"
    items_a = frozenset(payload_a)
    items_b = frozenset(payload_b)
    if items_a < items_b or items_b < items_a:
        return "DOMINATED"
    return "INCONSISTENT"


@dataclass(frozen=True)
class ConflictComparison:
    decision_a_hash: str
    decision_b_hash: str
    conflict_type: str

    def __post_init__(self) -> None:
        hash_a = _require_sha256_hex(self.decision_a_hash)
        hash_b = _require_sha256_hex(self.decision_b_hash)
        if hash_a > hash_b:
            hash_a, hash_b = hash_b, hash_a
        object.__setattr__(self, "decision_a_hash", hash_a)
        object.__setattr__(self, "decision_b_hash", hash_b)
        if self.conflict_type not in CONFLICT_TYPE:
            raise _invalid_input()


@dataclass(frozen=True)
class ConflictReceipt:
    comparison: ConflictComparison
    input_memory_hashes: tuple[str, ...]
    conflict_hash: str

    def __post_init__(self) -> None:
        canonical_memory = canonical_hash_identity(self.input_memory_hashes)
        object.__setattr__(self, "input_memory_hashes", canonical_memory)
        expected_hash = sha256_hex(
            {
                "comparison": {
                    "decision_a_hash": self.comparison.decision_a_hash,
                    "decision_b_hash": self.comparison.decision_b_hash,
                    "conflict_type": self.comparison.conflict_type,
                },
                "input_memory_hashes": self.input_memory_hashes,
            }
        )
        if self.conflict_hash != expected_hash:
            raise _invalid_input()


def classify_decision_conflict(
    input_memory_hashes: tuple[str, ...],
    decision_a_hash: str,
    decision_a_payload: tuple[tuple[str, Any], ...],
    decision_b_hash: str,
    decision_b_payload: tuple[tuple[str, Any], ...],
) -> ConflictReceipt:
    hash_a = _require_sha256_hex(decision_a_hash)
    hash_b = _require_sha256_hex(decision_b_hash)
    canonical_a = _canonical_payload(decision_a_payload)
    canonical_b = _canonical_payload(decision_b_payload)
    hash_matches_a = _payload_hash(canonical_a) == hash_a
    hash_matches_b = _payload_hash(canonical_b) == hash_b
    if not (hash_matches_a and hash_matches_b):
        if not (canonical_json(canonical_a) == canonical_json(canonical_b) and hash_a != hash_b):
            raise _invalid_input()

    if hash_a <= hash_b:
        left_hash, right_hash = hash_a, hash_b
        left_payload, right_payload = canonical_a, canonical_b
    else:
        left_hash, right_hash = hash_b, hash_a
        left_payload, right_payload = canonical_b, canonical_a
    conflict_type = _classify(left_payload, right_payload, left_hash, right_hash)
    comparison = ConflictComparison(
        decision_a_hash=left_hash,
        decision_b_hash=right_hash,
        conflict_type=conflict_type,
    )
    canonical_memory = canonical_hash_identity(input_memory_hashes)
    conflict_hash = sha256_hex(
        {
            "comparison": {
                "decision_a_hash": comparison.decision_a_hash,
                "decision_b_hash": comparison.decision_b_hash,
                "conflict_type": comparison.conflict_type,
            },
            "input_memory_hashes": canonical_memory,
        }
    )
    return ConflictReceipt(
        comparison=comparison,
        input_memory_hashes=canonical_memory,
        conflict_hash=conflict_hash,
    )
