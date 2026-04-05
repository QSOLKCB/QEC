from __future__ import annotations

from dataclasses import replace
import json

import pytest

from qec.analysis.signed_provenance_artifact_chain import (
    ProvenanceChain,
    append_provenance_artifact,
    compute_provenance_root,
    export_provenance_bytes,
    generate_provenance_receipt,
    verify_provenance_chain,
)


def _empty_chain() -> ProvenanceChain:
    return ProvenanceChain(artifacts=(), chain_root=compute_provenance_root(()))


def _hex(char: str) -> str:
    return char * 64


def _build_chain() -> ProvenanceChain:
    chain = _empty_chain()
    chain = append_provenance_artifact(
        chain,
        {"kind": "INIT", "payload": {"a": 1, "b": [3, 2, 1]}},
        originating_sovereignty_event_hash=_hex("1"),
        originating_privilege_decision_hash=_hex("2"),
        signer_key_id="kernel-signer-v1",
    )
    chain = append_provenance_artifact(
        chain,
        {"kind": "STEP", "payload": {"ok": True, "weight": 2.0}},
        originating_sovereignty_event_hash=_hex("3"),
        originating_privilege_decision_hash=_hex("4"),
        signer_key_id="kernel-signer-v1",
    )
    return chain


def test_repeated_run_determinism_and_stable_root() -> None:
    c1 = _build_chain()
    c2 = _build_chain()

    assert c1.artifacts == c2.artifacts
    assert c1.chain_root == c2.chain_root
    assert verify_provenance_chain(c1)
    assert verify_provenance_chain(c2)


def test_identical_inputs_identical_provenance_bytes() -> None:
    c1 = _build_chain()
    c2 = _build_chain()

    b1 = export_provenance_bytes(c1)
    b2 = export_provenance_bytes(c2)

    assert b1 == b2


def test_append_only_chain_enforcement() -> None:
    chain = _build_chain()
    tampered_artifact = replace(chain.artifacts[1], index=0)
    tampered_chain = ProvenanceChain(
        artifacts=(chain.artifacts[0], tampered_artifact),
        chain_root=compute_provenance_root((chain.artifacts[0], tampered_artifact)),
    )

    with pytest.raises(ValueError, match="artifact index sequence is not append-only"):
        verify_provenance_chain(tampered_chain)


def test_tamper_rejection_on_payload_mutation() -> None:
    chain = _build_chain()
    tampered = replace(chain.artifacts[0], payload={"kind": "INIX", "payload": {"a": 1, "b": (3, 2, 1)}})
    tampered_chain = ProvenanceChain(
        artifacts=(tampered, chain.artifacts[1]),
        chain_root=compute_provenance_root((tampered, chain.artifacts[1])),
    )

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_provenance_chain(tampered_chain)


def test_receipt_stability() -> None:
    r1 = generate_provenance_receipt(_build_chain())
    r2 = generate_provenance_receipt(_build_chain())
    assert r1 == r2


def test_replay_fidelity_from_canonical_export() -> None:
    chain = _build_chain()
    canonical = export_provenance_bytes(chain)
    decoded = json.loads(canonical.decode("utf-8"))

    rebuilt = _empty_chain()
    for artifact in decoded["artifacts"]:
        rebuilt = append_provenance_artifact(
            rebuilt,
            artifact["payload"],
            originating_sovereignty_event_hash=artifact["originating_sovereignty_event_hash"],
            originating_privilege_decision_hash=artifact["originating_privilege_decision_hash"],
            signer_key_id=artifact["signer_key_id"],
            schema_version=artifact["schema_version"],
        )

    assert rebuilt == chain
    assert export_provenance_bytes(rebuilt) == canonical


def test_fail_fast_invalid_input_handling() -> None:
    chain = _empty_chain()

    with pytest.raises(ValueError, match="payload keys must be strings"):
        append_provenance_artifact(
            chain,
            {1: "bad"},  # type: ignore[arg-type]
            originating_sovereignty_event_hash=_hex("1"),
            originating_privilege_decision_hash=_hex("2"),
            signer_key_id="k",
        )

    with pytest.raises(ValueError, match="payload keys must be strings"):
        append_provenance_artifact(
            chain,
            {"ok": True, 1: "mixed"},  # type: ignore[arg-type]
            originating_sovereignty_event_hash=_hex("1"),
            originating_privilege_decision_hash=_hex("2"),
            signer_key_id="k",
        )

    with pytest.raises(ValueError, match="payload must be a mapping object"):
        append_provenance_artifact(
            chain,
            ["bad"],  # type: ignore[arg-type]
            originating_sovereignty_event_hash=_hex("1"),
            originating_privilege_decision_hash=_hex("2"),
            signer_key_id="k",
        )

    with pytest.raises(ValueError, match="originating_sovereignty_event_hash must be 64 hex characters"):
        append_provenance_artifact(
            chain,
            {"ok": True},
            originating_sovereignty_event_hash="123",
            originating_privilege_decision_hash=_hex("2"),
            signer_key_id="k",
        )

    with pytest.raises(ValueError, match="non-finite float values are not permitted"):
        append_provenance_artifact(
            chain,
            {"x": float("nan")},
            originating_sovereignty_event_hash=_hex("1"),
            originating_privilege_decision_hash=_hex("2"),
            signer_key_id="k",
        )
