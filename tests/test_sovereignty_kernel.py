from __future__ import annotations

import pytest

from qec.analysis.sovereignty_kernel import (
    SovereignEventHistory,
    append_event,
    compute_merkle_root,
    export_history_canonical_bytes,
    generate_event_receipt,
    replay_history,
)


def _empty_history() -> SovereignEventHistory:
    return SovereignEventHistory(events=(), chain_root=compute_merkle_root(()))


def _build_history() -> SovereignEventHistory:
    history = _empty_history()
    history = append_event(history, {"kind": "INIT", "payload": {"alpha": 1, "beta": [3, 2, 1]}})
    history = append_event(history, {"kind": "STEP", "payload": {"gamma": "ok", "delta": 2.0}})
    return history


def test_repeated_run_determinism_and_merkle_root_stability() -> None:
    h1 = _build_history()
    h2 = _build_history()

    assert h1.events == h2.events
    assert h1.chain_root == h2.chain_root


def test_canonical_bytes_stability() -> None:
    history = _build_history()
    b1 = export_history_canonical_bytes(history)
    b2 = export_history_canonical_bytes(history)
    assert b1 == b2


def test_append_only_enforcement_and_immutability() -> None:
    history = _empty_history()
    history_1 = append_event(history, {"event": "A"})
    history_2 = append_event(history_1, {"event": "B"})

    assert len(history.events) == 0
    assert len(history_1.events) == 1
    assert len(history_2.events) == 2
    assert history_1.events[0].index == 0
    assert history_2.events[1].index == 1


def test_replay_fidelity_and_export_identity() -> None:
    history = _build_history()
    exported = export_history_canonical_bytes(history)

    replayed = replay_history(exported)
    replayed_bytes = export_history_canonical_bytes(replayed)

    assert replayed == history
    assert replayed_bytes == exported


def test_receipt_stability() -> None:
    h1 = _build_history()
    h2 = _build_history()

    r1 = generate_event_receipt(h1)
    r2 = generate_event_receipt(h2)

    assert r1 == r2


def test_fail_fast_invalid_input_handling() -> None:
    history = _empty_history()

    with pytest.raises(ValueError, match="payload keys must be strings"):
        append_event(history, {1: "bad"})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="payload must be a mapping object"):
        append_event(history, ["bad"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="events must be a JSON list"):
        replay_history(b'{"history_schema_version":1,"chain_root":"x","events":{}}')

    with pytest.raises(ValueError, match="non-finite float values are not permitted"):
        append_event(history, {"val": float("nan")})

    with pytest.raises(ValueError, match="non-finite float values are not permitted"):
        append_event(history, {"val": float("inf")})


def test_replay_rejects_tampered_payload() -> None:
    history = _build_history()
    exported = export_history_canonical_bytes(history)
    tampered = exported.replace(b'"INIT"', b'"INIX"')

    with pytest.raises(ValueError, match="event_hash replay mismatch"):
        replay_history(tampered)


def test_append_rejects_tampered_chain_root() -> None:
    history = _build_history()
    tampered_root = "a" * 64
    tampered_history = SovereignEventHistory(
        events=history.events,
        chain_root=tampered_root,
    )

    with pytest.raises(ValueError, match="history chain_root mismatch before append"):
        append_event(tampered_history, {"event": "C"})
