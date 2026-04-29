from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import permutations

import pytest

from qec.analysis.agent_specialization import AgentRole
from qec.analysis.inter_agent_protocol import AgentMessage, build_agent_message_state


def _identity() -> tuple[str, ...]:
    return (
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
    )


def _message(*, sender_id: str, receiver_id: str, sender_role: str, message_type: str, payload: tuple[tuple[str, object], ...]) -> AgentMessage:
    probe = {
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "sender_role": sender_role,
        "message_type": message_type,
        "payload": payload,
    }
    from qec.analysis.canonical_hashing import sha256_hex

    message_hash = sha256_hex(probe)
    return AgentMessage(
        sender_id=sender_id,
        receiver_id=receiver_id,
        sender_role=sender_role,
        message_type=message_type,
        payload=payload,
        message_hash=message_hash,
    )


def test_deterministic_replay_100_runs() -> None:
    messages = (
        _message(sender_id="a", receiver_id="b", sender_role=AgentRole.CONTROL, message_type="proposal", payload=(("x", 1),)),
        _message(sender_id="b", receiver_id="a", sender_role=AgentRole.VALIDATION, message_type="validation", payload=(("y", 2),)),
    )
    baseline = build_agent_message_state(_identity(), messages)
    for _ in range(100):
        assert build_agent_message_state(_identity(), messages) == baseline


def test_permutation_invariance_and_ordering_stability() -> None:
    messages = (
        _message(sender_id="c", receiver_id="a", sender_role=AgentRole.REPAIR, message_type="repair", payload=(("a", 3),)),
        _message(sender_id="a", receiver_id="c", sender_role=AgentRole.ADVERSARIAL, message_type="challenge", payload=(("b", 4),)),
        _message(sender_id="b", receiver_id="a", sender_role=AgentRole.COMPRESSION, message_type="confirmation", payload=(("c", 5),)),
    )
    expected = build_agent_message_state(_identity(), messages)
    for perm in permutations(messages):
        assert build_agent_message_state(_identity(), perm) == expected
    keys = tuple((m.message_hash, m.sender_id, m.receiver_id) for m in expected.message_state.messages)
    assert keys == tuple(sorted(keys))




def test_invalid_agent_ids_fail() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _message(sender_id="", receiver_id="b", sender_role=AgentRole.CONTROL, message_type="proposal", payload=(("x", 1),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _message(sender_id="a", receiver_id="", sender_role=AgentRole.CONTROL, message_type="proposal", payload=(("x", 1),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _message(sender_id=None, receiver_id="b", sender_role=AgentRole.CONTROL, message_type="proposal", payload=(("x", 1),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _message(sender_id="a", receiver_id=1, sender_role=AgentRole.CONTROL, message_type="proposal", payload=(("x", 1),))  # type: ignore[arg-type]


def test_agent_message_rejects_hash_from_non_canonical_payload() -> None:
    from qec.analysis.canonical_hashing import sha256_hex

    canonical_payload = (("a", 1), ("b", 2))
    non_canonical_probe = {
        "sender_id": "sender",
        "receiver_id": "receiver",
        "sender_role": AgentRole.CONTROL,
        "message_type": "proposal",
        "payload": {"b": 2, "a": 1},
    }
    bad_hash = sha256_hex(non_canonical_probe)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AgentMessage(
            sender_id="sender",
            receiver_id="receiver",
            sender_role=AgentRole.CONTROL,
            message_type="proposal",
            payload=canonical_payload,
            message_hash=bad_hash,
        )


def test_invalid_message_type_fails() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _message(sender_id="a", receiver_id="b", sender_role=AgentRole.CONTROL, message_type="unknown", payload=(("x", 1),))


def test_invalid_role_fails() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _message(sender_id="a", receiver_id="b", sender_role="observer", message_type="proposal", payload=(("x", 1),))


def test_malformed_payload_structure_fails() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _message(sender_id="a", receiver_id="b", sender_role=AgentRole.CONTROL, message_type="proposal", payload=(("x", 1, 2),))  # type: ignore[arg-type]


def test_non_tuple_payload_fails() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AgentMessage(
            sender_id="a",
            receiver_id="b",
            sender_role=AgentRole.CONTROL,
            message_type="proposal",
            payload=[("x", 1)],  # type: ignore[arg-type]
            message_hash="0" * 64,
        )


def test_hash_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AgentMessage(
            sender_id="a",
            receiver_id="b",
            sender_role=AgentRole.CONTROL,
            message_type="proposal",
            payload=(("x", 1),),
            message_hash="0" * 64,
        )


def test_duplicate_messages_fail() -> None:
    msg = _message(sender_id="a", receiver_id="b", sender_role=AgentRole.CONTROL, message_type="proposal", payload=(("x", 1),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_agent_message_state(_identity(), (msg, msg))


def test_identity_mismatch_fails() -> None:
    msg = _message(sender_id="a", receiver_id="b", sender_role=AgentRole.CONTROL, message_type="proposal", payload=(("x", 1),))
    bad_identity = (_identity()[1], _identity()[0])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_agent_message_state(bad_identity, (msg,))


def test_immutability() -> None:
    msg = _message(sender_id="a", receiver_id="b", sender_role=AgentRole.CONTROL, message_type="proposal", payload=(("x", 1),))
    receipt = build_agent_message_state(_identity(), (msg,))
    with pytest.raises(FrozenInstanceError):
        receipt.protocol_hash = "1" * 64


def test_nested_payload_is_immutable() -> None:
    from qec.analysis.canonical_hashing import sha256_hex

    payload = (("x", (("k", (1, 2)),)),)
    msg = AgentMessage(
        sender_id="a",
        receiver_id="b",
        sender_role=AgentRole.CONTROL,
        message_type="proposal",
        payload=(("x", {"k": [1, 2]}),),
        message_hash=sha256_hex(
            {
                "sender_id": "a",
                "receiver_id": "b",
                "sender_role": AgentRole.CONTROL,
                "message_type": "proposal",
                "payload": payload,
            }
        ),
    )
    nested = msg.payload[0][1]
    assert isinstance(nested, tuple)
    with pytest.raises((TypeError, AttributeError)):
        nested[0][1].append(3)  # type: ignore[union-attr]
