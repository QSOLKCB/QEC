"""Security hardening tests for execution_proof fallback mode."""

from __future__ import annotations

from qec.controller import execution_proof as ep


def _verify_result() -> dict:
    return {
        "final_hash": "a" * 64,
        "steps": 3,
        "match": True,
        "mismatch_index": None,
    }


def test_fallback_signature_roundtrip_is_consistent(monkeypatch):
    monkeypatch.setattr(ep, "_CRYPTO_AVAILABLE", False)
    payload = ep.serialize_proof_payload(ep.build_proof_payload(_verify_result(), "tester"))
    sig = ep.sign_payload(payload, b"private")
    assert ep.verify_signature(payload, sig, b"public") is True


def test_create_execution_proof_marks_fallback_mode_unverified(monkeypatch):
    monkeypatch.setattr(ep, "_CRYPTO_AVAILABLE", False)
    proof = ep.create_execution_proof(
        verify_result=_verify_result(),
        signer_id="tester",
        private_key_pem=b"private",
        public_key_pem=b"public",
    )
    assert proof["verification_mode"] == "fallback_hash"
    assert proof["payload"]["verification_mode"] == "fallback_hash"
    assert proof["verified"] is False
