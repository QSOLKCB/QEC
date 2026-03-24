"""Tests for signed execution proof layer (v81.1.0)."""

from __future__ import annotations

import copy
import json
import os
import tempfile

import pytest

from qec.controller.execution_proof import (
    build_proof_payload,
    create_execution_proof,
    serialize_proof_payload,
    sign_payload,
    verify_signature,
)

# ---------------------------------------------------------------------------
# Demo Ed25519 key pair (deterministic test fixture)
# ---------------------------------------------------------------------------

_PRIVATE_KEY_PEM = (
    b"-----BEGIN PRIVATE KEY-----\n"
    b"MC4CAQAwBQYDK2VwBCIEIO4Nngc2zhyTpxaDALLMVmUQ6OOjMk0eOgLjGnLLY2nN\n"
    b"-----END PRIVATE KEY-----\n"
)

_PUBLIC_KEY_PEM = (
    b"-----BEGIN PUBLIC KEY-----\n"
    b"MCowBQYDK2VwAyEAk5fIB0Cvc5fb2v0wizvCJjQFro2sald9OS1eyUO1soM=\n"
    b"-----END PUBLIC KEY-----\n"
)

# ---------------------------------------------------------------------------
# Sample verify_result (mimics replay_engine.verify_run output)
# ---------------------------------------------------------------------------

_VERIFY_RESULT = {
    "match": True,
    "final_hash": "a1b2c3d4e5f6" * 5 + "a1b2c3d4",
    "steps": 7,
    "mismatch_index": None,
}

_SIGNER_ID = "test-signer-001"


# ===== Test: Deterministic payload construction ============================


class TestBuildProofPayload:
    def test_basic_payload_structure(self):
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        assert payload["signer_id"] == _SIGNER_ID
        assert payload["final_hash"] == _VERIFY_RESULT["final_hash"]
        assert payload["steps"] == 7
        assert payload["match"] is True
        assert payload["mismatch_index"] is None
        assert payload["metadata"] == {}

    def test_payload_with_metadata(self):
        meta = {"experiment": "spectral_radius", "version": "81.1.0"}
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID, metadata=meta)
        assert payload["metadata"] == {"experiment": "spectral_radius", "version": "81.1.0"}

    def test_metadata_keys_sorted(self):
        meta = {"z_key": "last", "a_key": "first"}
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID, metadata=meta)
        keys = list(payload["metadata"].keys())
        assert keys == sorted(keys)

    def test_deterministic_repeated_calls(self):
        p1 = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        p2 = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        assert p1 == p2

    def test_missing_signer_id_raises(self):
        with pytest.raises(ValueError):
            build_proof_payload(_VERIFY_RESULT, "")

    def test_missing_verify_key_raises(self):
        incomplete = {"match": True, "final_hash": "abc"}
        with pytest.raises(KeyError):
            build_proof_payload(incomplete, _SIGNER_ID)

    def test_no_mutation_of_verify_result(self):
        vr = copy.deepcopy(_VERIFY_RESULT)
        original = copy.deepcopy(vr)
        build_proof_payload(vr, _SIGNER_ID, metadata={"x": 1})
        assert vr == original


# ===== Test: Stable serialization ==========================================


class TestSerializeProofPayload:
    def test_deterministic_bytes(self):
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        b1 = serialize_proof_payload(payload)
        b2 = serialize_proof_payload(payload)
        assert b1 == b2

    def test_sorted_keys_in_output(self):
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        raw = serialize_proof_payload(payload).decode("utf-8")
        parsed = json.loads(raw)
        assert list(parsed.keys()) == sorted(parsed.keys())

    def test_identical_payloads_identical_bytes(self):
        p1 = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID, metadata={"a": 1})
        p2 = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID, metadata={"a": 1})
        assert serialize_proof_payload(p1) == serialize_proof_payload(p2)


# ===== Test: Signing & verification ========================================


class TestSignAndVerify:
    def test_valid_signature_verifies(self):
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        payload_bytes = serialize_proof_payload(payload)
        sig = sign_payload(payload_bytes, _PRIVATE_KEY_PEM)
        assert isinstance(sig, str)
        assert verify_signature(payload_bytes, sig, _PUBLIC_KEY_PEM) is True

    def test_deterministic_signature(self):
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        payload_bytes = serialize_proof_payload(payload)
        sig1 = sign_payload(payload_bytes, _PRIVATE_KEY_PEM)
        sig2 = sign_payload(payload_bytes, _PRIVATE_KEY_PEM)
        assert sig1 == sig2

    def test_tampered_payload_fails_verification(self):
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        payload_bytes = serialize_proof_payload(payload)
        sig = sign_payload(payload_bytes, _PRIVATE_KEY_PEM)
        tampered = payload_bytes + b"tampered"
        assert verify_signature(tampered, sig, _PUBLIC_KEY_PEM) is False

    def test_tampered_signature_fails_verification(self):
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        payload_bytes = serialize_proof_payload(payload)
        sig = sign_payload(payload_bytes, _PRIVATE_KEY_PEM)
        bad_sig = "00" * 64  # wrong signature, valid hex
        assert verify_signature(payload_bytes, bad_sig, _PUBLIC_KEY_PEM) is False

    def test_wrong_key_fails_verification(self):
        """Sign with one key, verify with a different key's public half."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
            PublicFormat,
        )

        # Generate a second key pair deterministically is impossible,
        # but we only need *a different* public key here.
        other_key = Ed25519PrivateKey.generate()
        other_pub = other_key.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        payload = build_proof_payload(_VERIFY_RESULT, _SIGNER_ID)
        payload_bytes = serialize_proof_payload(payload)
        sig = sign_payload(payload_bytes, _PRIVATE_KEY_PEM)
        assert verify_signature(payload_bytes, sig, other_pub) is False


# ===== Test: create_execution_proof ========================================


class TestCreateExecutionProof:
    def test_proof_structure(self):
        proof = create_execution_proof(
            _VERIFY_RESULT, _SIGNER_ID, _PRIVATE_KEY_PEM, _PUBLIC_KEY_PEM
        )
        assert "payload" in proof
        assert "signature" in proof
        assert "verified" in proof
        assert proof["verified"] is True

    def test_proof_with_metadata(self):
        proof = create_execution_proof(
            _VERIFY_RESULT,
            _SIGNER_ID,
            _PRIVATE_KEY_PEM,
            _PUBLIC_KEY_PEM,
            metadata={"run": "alpha"},
        )
        assert proof["payload"]["metadata"] == {"run": "alpha"}

    def test_output_json_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proof = create_execution_proof(
                _VERIFY_RESULT,
                _SIGNER_ID,
                _PRIVATE_KEY_PEM,
                _PUBLIC_KEY_PEM,
                output_dir=tmpdir,
            )
            path = os.path.join(tmpdir, "execution_proof.json")
            assert os.path.isfile(path)
            with open(path) as f:
                on_disk = json.load(f)
            assert on_disk["signature"] == proof["signature"]
            assert on_disk["verified"] is True

    def test_no_output_dir_no_file(self):
        proof = create_execution_proof(
            _VERIFY_RESULT, _SIGNER_ID, _PRIVATE_KEY_PEM, _PUBLIC_KEY_PEM
        )
        assert proof["verified"] is True
        # No file written — just ensure no error

    def test_repeated_identical_proofs(self):
        p1 = create_execution_proof(
            _VERIFY_RESULT, _SIGNER_ID, _PRIVATE_KEY_PEM, _PUBLIC_KEY_PEM
        )
        p2 = create_execution_proof(
            _VERIFY_RESULT, _SIGNER_ID, _PRIVATE_KEY_PEM, _PUBLIC_KEY_PEM
        )
        assert p1["signature"] == p2["signature"]
        assert p1["payload"] == p2["payload"]
