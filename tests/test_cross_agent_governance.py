from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import itertools
import json

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.cross_agent_governance import AgentDecision, GovernanceReceipt, arbitrate_decisions
from qec.analysis.shared_memory_fabric import merge_memory_receipts


class _MemoryReceipt:
    def __init__(self, source_agent_id: str, memory_payload: object) -> None:
        self.source_agent_id = source_agent_id
        self.memory_payload = memory_payload
        self.memory_hash = sha256_hex(memory_payload)


def _build_shared_state():
    receipts = (
        _MemoryReceipt("agent-a", {"x": 1}),
        _MemoryReceipt("agent-b", {"x": 2}),
        _MemoryReceipt("agent-c", {"x": 3}),
    )
    shared_state = merge_memory_receipts(receipts)
    input_hashes = tuple(entry.memory_hash for entry in shared_state.entries)
    return shared_state, input_hashes, receipts


def _decision_payload(value: int) -> tuple[tuple[str, object], ...]:
    return (("priority", value), ("state", ("active", True)))


def _decision(agent_id: str, value: int) -> AgentDecision:
    payload = _decision_payload(value)
    return AgentDecision(
        agent_id=agent_id,
        decision_hash=sha256_hex(dict(payload)),
        decision_payload=payload,
    )


def _selected_hash_from_state(state) -> str:
    return state.decisions[0].decision_hash


def _receipt_hash_payload(state, input_hashes: tuple[str, ...], selected_decision_hash: str) -> dict[str, object]:
    return {
        "decisions": [
            {
                "agent_id": entry.agent_id,
                "decision_hash": entry.decision_hash,
            }
            for entry in state.decisions
        ],
        "input_memory_hashes": list(input_hashes),
        "selected_decision_hash": selected_decision_hash,
    }


def _recompute_hash(state, input_hashes: tuple[str, ...], selected_decision_hash: str) -> str:
    payload = _receipt_hash_payload(state, input_hashes, selected_decision_hash)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_deterministic_replay() -> None:
    shared_state, input_hashes, _ = _build_shared_state()
    decisions = (
        _decision("a", 4),
        _decision("b", 2),
        _decision("c", 9),
    )

    expected = arbitrate_decisions(shared_state, input_hashes, decisions)
    for _ in range(100):
        assert arbitrate_decisions(shared_state, input_hashes, decisions) == expected


def test_permutation_invariance() -> None:
    shared_state, input_hashes, _ = _build_shared_state()
    decisions = (
        _decision("a", 1),
        _decision("b", 2),
        _decision("c", 3),
    )
    expected = arbitrate_decisions(shared_state, input_hashes, decisions)

    for permuted in itertools.permutations(decisions):
        assert arbitrate_decisions(shared_state, input_hashes, permuted) == expected

    _, rebuilt_input_hashes, receipts = _build_shared_state()
    reversed_state = merge_memory_receipts(tuple(reversed(receipts)))
    assert tuple(entry.memory_hash for entry in reversed_state.entries) == rebuilt_input_hashes
    assert arbitrate_decisions(reversed_state, rebuilt_input_hashes, tuple(reversed(decisions))) == expected


def test_canonical_input_enforcement() -> None:
    shared_state, input_hashes, _ = _build_shared_state()
    decision = (_decision("a", 1),)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        arbitrate_decisions(shared_state, tuple(reversed(input_hashes)), decision)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        arbitrate_decisions(shared_state, (input_hashes[0], input_hashes[0]), decision)

    bad_hashes = tuple(sorted(("0" * 64,) + input_hashes[:-1]))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        arbitrate_decisions(shared_state, bad_hashes, decision)


def test_decision_integrity() -> None:
    shared_state, input_hashes, _ = _build_shared_state()

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        bad_payload = (("z", 1), ("a", 2))
        AgentDecision(agent_id="a", decision_hash=sha256_hex({"a": 2, "z": 1}), decision_payload=bad_payload)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AgentDecision(agent_id="a", decision_hash="f" * 64, decision_payload=(("a", 1),))

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        bad_decision = AgentDecision(agent_id="a", decision_hash=sha256_hex({"a": 1}), decision_payload=(("a", 1),))
        arbitrate_decisions(shared_state, input_hashes, (bad_decision, object()))


def test_deduplication_duplicate_decision_hash_fails() -> None:
    shared_state, input_hashes, _ = _build_shared_state()
    payload = _decision_payload(3)
    shared_hash = sha256_hex(dict(payload))
    decisions = (
        AgentDecision(agent_id="agent-x", decision_hash=shared_hash, decision_payload=payload),
        AgentDecision(agent_id="agent-y", decision_hash=shared_hash, decision_payload=payload),
    )

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        arbitrate_decisions(shared_state, input_hashes, decisions)


def test_selection_stability_first_sorted_selected() -> None:
    shared_state, input_hashes, _ = _build_shared_state()
    decisions = (
        _decision("z", 99),
        _decision("a", 1),
        _decision("m", 50),
    )

    governance_state = arbitrate_decisions(shared_state, input_hashes, decisions)
    expected_first = min(governance_state.decisions, key=lambda entry: (entry.decision_hash, entry.agent_id))
    assert _selected_hash_from_state(governance_state) == expected_first.decision_hash


def test_hash_stability() -> None:
    shared_state, input_hashes, _ = _build_shared_state()
    governance_state = arbitrate_decisions(shared_state, input_hashes, (_decision("a", 1), _decision("b", 2)))
    selected_hash = _selected_hash_from_state(governance_state)
    governance_hash = _recompute_hash(governance_state, input_hashes, selected_hash)

    receipt = GovernanceReceipt(
        governance_state=governance_state,
        input_memory_hashes=input_hashes,
        selected_decision_hash=selected_hash,
        governance_hash=governance_hash,
    )

    assert receipt.governance_hash == _recompute_hash(receipt.governance_state, receipt.input_memory_hashes, receipt.selected_decision_hash)


def test_governance_receipt_invalid_inputs() -> None:
    shared_state, input_hashes, _ = _build_shared_state()
    governance_state = arbitrate_decisions(shared_state, input_hashes, (_decision("a", 1), _decision("b", 2)))
    selected_hash = _selected_hash_from_state(governance_state)
    governance_hash = _recompute_hash(governance_state, input_hashes, selected_hash)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        GovernanceReceipt(
            governance_state=governance_state,
            input_memory_hashes=input_hashes,
            selected_decision_hash="0" * 64,
            governance_hash=governance_hash,
        )

    malformed_hashes = ("abc",) + input_hashes[1:]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        GovernanceReceipt(
            governance_state=governance_state,
            input_memory_hashes=malformed_hashes,
            selected_decision_hash=selected_hash,
            governance_hash=governance_hash,
        )

    modified_hashes = tuple(reversed(input_hashes))
    modified_governance_hash = _recompute_hash(governance_state, modified_hashes, selected_hash)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        GovernanceReceipt(
            governance_state=governance_state,
            input_memory_hashes=modified_hashes,
            selected_decision_hash=selected_hash,
            governance_hash=modified_governance_hash,
        )

    wrong_selected = governance_state.decisions[-1].decision_hash
    wrong_selected_hash = _recompute_hash(governance_state, input_hashes, wrong_selected)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        GovernanceReceipt(
            governance_state=governance_state,
            input_memory_hashes=input_hashes,
            selected_decision_hash=wrong_selected,
            governance_hash=wrong_selected_hash,
        )

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        GovernanceReceipt(
            governance_state=governance_state,
            input_memory_hashes=input_hashes,
            selected_decision_hash=selected_hash,
            governance_hash="f" * 64,
        )


def test_immutability() -> None:
    shared_state, input_hashes, _ = _build_shared_state()
    payload = (("config", {"mode": "safe"}),)
    decision = AgentDecision(
        agent_id="agent-immut",
        decision_hash=sha256_hex({"config": {"mode": "safe"}}),
        decision_payload=payload,
    )
    governance_state = arbitrate_decisions(shared_state, input_hashes, (decision,))

    with pytest.raises(FrozenInstanceError):
        decision.agent_id = "modified"  # type: ignore[misc]

    nested = dict(decision.decision_payload)["config"]
    with pytest.raises(TypeError):
        nested["mode"] = "unsafe"  # type: ignore[index]

    with pytest.raises(TypeError):
        nested |= {"extra": 1}  # type: ignore[operator]

    with pytest.raises(FrozenInstanceError):
        governance_state.decisions = ()  # type: ignore[misc]


def test_invalid_hash_format_fails_fast() -> None:
    shared_state, input_hashes, _ = _build_shared_state()
    decision = _decision("a", 1)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        arbitrate_decisions(shared_state, ("ABC",), (decision,))

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AgentDecision(agent_id="a", decision_hash="ABC", decision_payload=(("a", 1),))


def test_arbitrate_rejects_invalid_shared_memory_state() -> None:
    _, input_hashes, _ = _build_shared_state()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        arbitrate_decisions(object(), input_hashes, (_decision("a", 1),))  # type: ignore[arg-type]
