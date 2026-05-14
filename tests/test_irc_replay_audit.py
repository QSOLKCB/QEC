import json

import pytest

from qec.operator.irc_replay_audit import (
    IRCReplayAuditEvent,
    IRCReplayAuditReceipt,
    _MAX_AUDIT_EVENTS,
    build_irc_replay_audit_event,
    build_irc_replay_audit_receipt,
    get_allowed_irc_audit_event_statuses,
    normalize_audit_line,
    replay_irc_audit_from_interactions,
    validate_irc_replay_audit_event,
    validate_irc_replay_audit_receipt,
)


def _event(i: int, line: str = "PRIVMSG #qec :!help", out=(":qec-ircd NOTICE #qec :ok",)):
    return build_irc_replay_audit_event(i, "c1", line, out)


def test_event_and_receipt_hash_deterministic_and_canonical_stable():
    e1 = _event(0)
    e2 = _event(0)
    assert e1.event_hash == e2.event_hash
    r1 = build_irc_replay_audit_receipt((e1,))
    r2 = build_irc_replay_audit_receipt((e2,))
    assert r1.irc_replay_audit_hash == r2.irc_replay_audit_hash
    assert json.dumps(e1.to_dict(), sort_keys=True)
    assert json.dumps(r1.to_dict(), sort_keys=True)
    assert e1.to_canonical_json() == e2.to_canonical_json()
    assert e1.to_canonical_bytes() == e2.to_canonical_bytes()


def test_command_detection_and_normalization():
    e = _event(0, "PRIVMSG #qec :!HeLp")
    assert e.command_detected is True
    assert e.command_name == "help"
    e2 = _event(0, "PRIVMSG #qec :hello")
    assert e2.command_detected is False
    assert e2.command_name is None
    assert normalize_audit_line("PING :x\r\n") == "PING :x"


def test_bounds_and_invalid_inputs():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        normalize_audit_line("x\x00")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_irc_replay_audit_event(0, "c1", "PRIVMSG #qec :ok", ["a" * 513])
    with pytest.raises(ValueError, match="INVALID_EVENT_INDEX"):
        build_irc_replay_audit_event(True, "c1", "PRIVMSG #qec :ok", ())
    with pytest.raises(ValueError, match="INVALID_EVENT_INDEX"):
        build_irc_replay_audit_event(-1, "c1", "PRIVMSG #qec :ok", ())


def test_receipt_order_and_limits():
    e0 = _event(0)
    e1 = _event(1)
    with pytest.raises(ValueError, match="DUPLICATE_AUDIT_EVENT"):
        build_irc_replay_audit_receipt((e0, _event(0)))
    with pytest.raises(ValueError, match="AUDIT_EVENT_ORDER_MISMATCH"):
        build_irc_replay_audit_receipt((e1,))
    too_many = tuple(_event(i) for i in range(_MAX_AUDIT_EVENTS + 1))
    with pytest.raises(ValueError, match="AUDIT_EVENT_LIMIT_EXCEEDED"):
        build_irc_replay_audit_receipt(too_many)


def test_validators_and_hash_mismatch_checks():
    e = _event(0)
    assert validate_irc_replay_audit_event(e)
    d = e.to_dict(); d["output_lines"] = tuple(d["output_lines"]); bad_status = IRCReplayAuditEvent(**{**d, "event_status": "BAD"})
    with pytest.raises(ValueError, match="INVALID_EVENT_STATUS"):
        validate_irc_replay_audit_event(bad_status)
    bad_hash_fmt = IRCReplayAuditEvent(**{**d, "event_hash": "x"})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_irc_replay_audit_event(bad_hash_fmt)
    bad_hash = IRCReplayAuditEvent(**{**d, "event_hash": "0" * 64})
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_irc_replay_audit_event(bad_hash)

    r = build_irc_replay_audit_receipt((e,))
    assert validate_irc_replay_audit_receipt(r)
    rd = r.to_dict(); rd["events"] = tuple(r.events); bad_r_fmt = IRCReplayAuditReceipt(**{**rd, "irc_replay_audit_hash": "x"})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_irc_replay_audit_receipt(bad_r_fmt)
    bad_r = IRCReplayAuditReceipt(**{**rd, "irc_replay_audit_hash": "0" * 64})
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_irc_replay_audit_receipt(bad_r)


def test_counts_first_final_and_replay_determinism():
    e0 = build_irc_replay_audit_event(0, "c1", "PRIVMSG #qec :!help", (":qec-ircd NOTICE #qec :ok",))
    e1 = build_irc_replay_audit_event(1, "c1", "PRIVMSG #qec :hi", ("ERROR bad",))
    r = build_irc_replay_audit_receipt((e0, e1))
    assert r.event_count == 2
    assert r.command_event_count == 1
    assert r.error_event_count == 1
    assert r.first_event_hash == e0.event_hash
    assert r.final_event_hash == e1.event_hash

    interactions = [("c1", "PRIVMSG #qec :!help", (":qec-ircd NOTICE #qec :ok",)), ("c1", "PRIVMSG #qec :hi", ("ok",))]
    a = replay_irc_audit_from_interactions(interactions)
    b = replay_irc_audit_from_interactions(interactions)
    assert a.command_manifest_hash == b.command_manifest_hash
    assert a.irc_replay_audit_hash == b.irc_replay_audit_hash
    assert "IRC_EVENT_OK" in get_allowed_irc_audit_event_statuses()
