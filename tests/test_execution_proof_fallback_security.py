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
    assert ep.verify_compatibility_marker(payload, sig) is True
    assert ep.verify_signature(payload, sig, b"public") is False


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


def test_untrusted_signer_cannot_mark_proof_verified(monkeypatch):
    monkeypatch.setattr(ep, "_CRYPTO_AVAILABLE", False)
    result = ep.create_execution_proof(
        _verify_result(), "arbitrary", b"private", b"public",
        trusted_signers={"arbitrary": b"public"},
    )
    assert result["replay_valid"] is True
    assert result["signature_valid"] is False
    assert result["verified"] is False
    assert ep.authenticate_execution_proof(result, {"arbitrary": b"public"}) is False


def test_unsigned_proof_is_not_implicitly_authorized():
    result = ep.create_execution_proof(_verify_result(), "unsigned", None, None)
    assert result["verified"] is False
    assert result["verification_mode"] == "unsigned"
    assert result["signer_authorized"] is False
    assert ep.authenticate_execution_proof(result, {}) is False


def test_ed25519_requires_successful_replay_and_externally_pinned_signer():
    pytest = __import__("pytest")
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PrivateFormat, PublicFormat, NoEncryption,
    )
    private = Ed25519PrivateKey.generate()
    prv = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    pub = private.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    good = ep.create_execution_proof(
        _verify_result(), "operator", prv, pub,
        trusted_signers={"operator": pub},
    )
    assert good["signature_valid"] is True
    assert good["verified"] is True
    assert ep.authenticate_execution_proof(good, {"operator": pub})
    assert ep.authenticate_execution_proof(good, {}) is False
    assert ep.authenticate_execution_proof(good, {"operator": b"mismatch"}) is False

    bad = {**_verify_result(), "match": False, "mismatch_index": 0}
    rejected = ep.create_execution_proof(
        bad, "operator", prv, pub, trusted_signers={"operator": pub}
    )
    assert rejected["signature_valid"] is True
    assert rejected["replay_valid"] is False
    assert rejected["verified"] is False
    assert ep.authenticate_execution_proof(rejected, {"operator": pub}) is False
