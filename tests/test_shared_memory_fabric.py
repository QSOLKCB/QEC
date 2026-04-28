from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.shared_memory_fabric import SharedMemoryReceipt, merge_memory_receipts


@dataclass(frozen=True)
class _MemoryReceipt:
    source_agent_id: str
    memory_hash: str
    memory_payload: object


def _hash_payload(payload: object) -> str:
    return sha256_hex(payload)


def _sample_receipts() -> tuple[_MemoryReceipt, ...]:
    payloads = (
        {"x": 1, "z": [1, 2]},
        {"a": "alpha", "b": True},
        {"nested": {"k": 4}, "q": None},
    )
    return tuple(
        _MemoryReceipt(
            source_agent_id=f"agent-{idx}",
            memory_hash=_hash_payload(payload),
            memory_payload=payload,
        )
        for idx, payload in enumerate(payloads)
    )


def _receipt_hash_payload(state, input_hashes: tuple[str, ...]) -> dict[str, object]:
    return {
        "entries": [
            {
                "source_agent_id": entry.source_agent_id,
                "memory_hash": entry.memory_hash,
                "memory_payload": [[k, v] for k, v in entry.memory_payload],
            }
            for entry in state.entries
        ],
        "input_memory_hashes": list(input_hashes),
    }


def _recompute_hash(state, input_hashes: tuple[str, ...]) -> str:
    payload = _receipt_hash_payload(state, input_hashes)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_deterministic_replay() -> None:
    inputs = _sample_receipts()
    expected = merge_memory_receipts(inputs)

    for _ in range(100):
        result = merge_memory_receipts(inputs)
        assert result == expected


def test_order_independence() -> None:
    inputs = _sample_receipts()
    expected = merge_memory_receipts(inputs)

    for permuted in itertools.permutations(inputs):
        assert merge_memory_receipts(permuted) == expected


def test_deduplication_single_entry_per_hash() -> None:
    inputs = _sample_receipts()
    duplicated = inputs + (inputs[0],)

    merged = merge_memory_receipts(duplicated)
    assert len(merged.entries) == len(inputs)


def test_canonical_ordering() -> None:
    merged = merge_memory_receipts(tuple(reversed(_sample_receipts())))
    ordering = tuple((entry.memory_hash, entry.source_agent_id) for entry in merged.entries)
    assert ordering == tuple(sorted(ordering))


def test_hash_stability() -> None:
    merged = merge_memory_receipts(_sample_receipts())
    input_hashes = tuple(sorted(entry.memory_hash for entry in merged.entries))
    recomputed_hash = _recompute_hash(merged, input_hashes)

    receipt = SharedMemoryReceipt(
        shared_memory_state=merged,
        input_memory_hashes=input_hashes,
        shared_memory_hash=recomputed_hash,
    )
    assert receipt.shared_memory_hash == _recompute_hash(receipt.shared_memory_state, receipt.input_memory_hashes)


def test_mutation_protection() -> None:
    state = merge_memory_receipts(_sample_receipts())
    with pytest.raises(TypeError):
        state.entries[0].memory_payload["x"] = 1  # type: ignore[index]


def test_invalid_hash_validation() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        merge_memory_receipts((_MemoryReceipt(source_agent_id="a", memory_hash=123, memory_payload={"x": 1}),))

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        merge_memory_receipts((_MemoryReceipt(source_agent_id="a", memory_hash="abcd", memory_payload={"x": 1}),))


def test_invalid_non_canonical_payload() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        merge_memory_receipts((_MemoryReceipt(source_agent_id="a", memory_hash="0" * 64, memory_payload={1: "bad"}),))


def test_duplicate_inconsistent_entries_fail_fast() -> None:
    memory_hash = "1" * 64
    receipt_a = _MemoryReceipt(source_agent_id="a", memory_hash=memory_hash, memory_payload={"x": 1})
    receipt_b = _MemoryReceipt(source_agent_id="b", memory_hash=memory_hash, memory_payload={"x": 1})

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        merge_memory_receipts((receipt_a, receipt_b))
