"""Signed execution proof layer (v81.1.0).

Constructs canonical proof payloads from verified replay results,
signs them deterministically using Ed25519, and writes proof artifacts.

Ed25519 signatures are inherently deterministic — identical payloads
with the same private key always produce the same signature.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

_CRYPTO_AVAILABLE = False
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
        load_pem_private_key,
        load_pem_public_key,
    )

    _CRYPTO_AVAILABLE = True
except BaseException:  # pragma: no cover  # pyo3 panics are BaseException
    pass

# -- Step 1: Canonical proof payload ----------------------------------------


def build_proof_payload(
    verify_result: dict,
    signer_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """Build a canonical proof payload from a verified run result.

    Parameters
    ----------
    verify_result : dict
        Must contain ``final_hash``, ``steps``, ``match``, ``mismatch_index``.
    signer_id : str
        Identity of the signer.
    metadata : dict or None
        Optional caller-supplied metadata.  No automatic fields are added.

    Returns
    -------
    dict  with deterministic key order (sorted).
    """
    if not signer_id:
        raise ValueError("signer_id must be a non-empty string")
    for key in ("final_hash", "steps", "match", "mismatch_index"):
        if key not in verify_result:
            raise KeyError(f"verify_result missing required key: {key}")

    payload = {
        "final_hash": verify_result["final_hash"],
        "match": bool(verify_result["match"]),
        "metadata": dict(sorted(metadata.items())) if metadata else {},
        "mismatch_index": verify_result["mismatch_index"],
        "signer_id": str(signer_id),
        "steps": int(verify_result["steps"]),
    }
    return payload


# -- Step 2: Canonical serialization ----------------------------------------


def serialize_proof_payload(payload: dict) -> bytes:
    """Serialize a proof payload to deterministic UTF-8 bytes.

    Sorted keys, compact separators, no trailing newline.
    Identical payloads always produce identical bytes.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# -- Step 3: Signing --------------------------------------------------------


def sign_payload(payload_bytes: bytes, private_key_pem: bytes) -> str:
    """Sign *payload_bytes* with an Ed25519 private key.

    Returns the signature as a hex string.  Ed25519 signatures are
    deterministic — same key + same payload always yield the same signature.

    If the ``cryptography`` package is not usable, returns a deterministic
    SHA-256 hex digest of ``b"FALLBACK|" + payload_bytes`` instead.  This
    fallback value is a compatibility marker and is **not** a cryptographic
    signature.
    """
    if not _CRYPTO_AVAILABLE:
        return hashlib.sha256(b"FALLBACK|" + payload_bytes).hexdigest()
    key = load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Private key must be Ed25519")
    signature = key.sign(payload_bytes)
    return signature.hex()


# -- Step 4: Verification ---------------------------------------------------


def verify_signature(
    payload_bytes: bytes, signature: str, public_key_pem: bytes
) -> bool:
    """Verify an Ed25519 signature.

    Returns ``True`` for a valid signature, ``False`` otherwise.
    Does not raise for ordinary invalid signatures.
    """
    if not _CRYPTO_AVAILABLE:
        # The compatibility marker has no secret and is NOT authentication.
        return False
    key = load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("Public key must be Ed25519")
    try:
        key.verify(bytes.fromhex(signature), payload_bytes)
        return True
    except Exception:
        return False


def verify_compatibility_marker(payload_bytes: bytes, marker: str) -> bool:
    """Integrity-only v81 compatibility marker; never authenticates a signer."""
    import hmac
    if not isinstance(marker, str):
        return False
    expected = hashlib.sha256(b"FALLBACK|" + payload_bytes).hexdigest()
    return hmac.compare_digest(marker, expected)


def authenticate_execution_proof(
    proof: dict, trusted_signers: Mapping[str, bytes]
) -> bool:
    """Authenticate replay-successful Ed25519 proof under an external trust root.

    The trust mapping must come from the *verifier*, not the proof producer.
    No saved self-verification Boolean is accepted as evidence of authority.
    """
    if not isinstance(proof, dict) or not isinstance(trusted_signers, Mapping):
        return False
    payload = proof.get("signed_payload")
    envelope = proof.get("payload")
    if not isinstance(payload, dict) or not isinstance(envelope, dict):
        return False
    if (payload.get("match") is not True or
            payload.get("mismatch_index") is not None or
            proof.get("verification_mode") != "ed25519" or
            envelope != {**payload, "verification_mode": "ed25519"}):
        return False
    signer_id = payload.get("signer_id")
    trusted_key = trusted_signers.get(signer_id) if isinstance(signer_id, str) else None
    if not isinstance(trusted_key, bytes):
        return False
    try:
        return verify_signature(
            serialize_proof_payload(payload), proof.get("signature", ""), trusted_key
        )
    except (ValueError, TypeError):
        return False



def experiment_result_binding(
    result: dict, *, domain: str, identity: Any
) -> dict:
    """Canonical v1 binding for the complete scored experiment, excluding proof.

    A digest of *only* a replay receipt cannot authenticate the metrics and
    history next to it. Bind all non-proof result fields, including consensus,
    and the verifier-supplied experiment identity under a domain separator.
    """
    if not isinstance(result, dict) or not isinstance(domain, str) or not domain:
        raise ValueError("invalid experiment result or domain")
    required = ("features", "probe", "invariants", "trajectory",
                "verification", "consensus")
    if any(key not in result for key in required):
        raise ValueError("incomplete experiment result")
    body = {key: value for key, value in result.items() if key != "proof"}
    preimage = {"domain": domain, "identity": identity, "result": body}
    digest = hashlib.sha256(
        json.dumps(preimage, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {"domain": domain, "identity": identity, "result_hash": digest}


def authenticate_experiment_result(
    result: dict,
    trusted_signers: Mapping[str, bytes],
    *,
    domain: str,
    identity: Any,
) -> bool:
    """Require an authorized signature on the *same* result being scored.

    Older replay-only signatures are still verifiable as replay receipts via
    authenticate_execution_proof, but cannot authenticate experiment metrics.
    """
    if not isinstance(result, dict):
        return False
    proof = result.get("proof")
    if not authenticate_execution_proof(proof, trusted_signers):
        return False
    try:
        signed = proof["signed_payload"]
        verification = result["verification"]
        history = result["probe"]["history"]
        invariant_history = result["invariants"]["history"]
        if (not isinstance(verification, dict) or
                not isinstance(history, list) or
                not isinstance(invariant_history, list) or
                verification.get("match") is not True or
                verification.get("mismatch_index") is not None or
                type(verification.get("steps")) is not int or
                verification["steps"] != len(history)):
            return False
        for key in ("final_hash", "steps", "match", "mismatch_index"):
            if signed.get(key) != verification.get(key):
                return False
        # Replay success alone must match the supplied trace, not an unrelated
        # valid run. The historical hash-chain format remains untouched.
        from qec.controller.replay_engine import build_hash_chain
        if build_hash_chain(history)["final_hash"] != verification["final_hash"]:
            return False
        expected = experiment_result_binding(
            result, domain=domain, identity=identity
        )
        return signed.get("metadata", {}).get("experiment_binding_v1") == expected
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError, OverflowError):
        return False


# -- Step 5: Main entry ------------------------------------------------------


def create_execution_proof(
    verify_result: dict,
    signer_id: str,
    private_key_pem: bytes | None,
    public_key_pem: bytes | None,
    metadata: dict[str, Any] | None = None,
    output_dir: str | None = None,
    *,
    trusted_signers: Mapping[str, bytes] | None = None,
) -> dict:
    """Create a signed execution proof from a verified run result.

    Parameters
    ----------
    verify_result : dict
        Output of ``replay_engine.verify_run``.
    signer_id : str
        Identity of the signer.
    private_key_pem / public_key_pem : bytes
        PEM-encoded Ed25519 key pair.
    metadata : dict or None
        Optional caller-supplied metadata.
    output_dir : str or None
        If provided, write ``execution_proof.json`` to this directory.

    Returns
    -------
    dict  with keys ``payload``, ``signature``, ``verified``,
    ``verification_mode``.  The ``payload`` value includes a
    ``verification_mode`` field.  The ``signature`` covers the
    original payload **without** ``verification_mode``; consumers
    must strip that key before signature verification.
    """
    payload = build_proof_payload(verify_result, signer_id, metadata)
    payload_bytes = serialize_proof_payload(payload)
    replay_valid = (verify_result.get("match") is True and
                    verify_result.get("mismatch_index") is None)
    has_keys = isinstance(private_key_pem, bytes) and isinstance(public_key_pem, bytes)
    verification_mode = (
        "ed25519" if has_keys and _CRYPTO_AVAILABLE
        else "fallback_hash" if has_keys else "unsigned"
    )
    signature = sign_payload(payload_bytes, private_key_pem) if has_keys else ""
    signature_valid = (
        verify_signature(payload_bytes, signature, public_key_pem)
        if verification_mode == "ed25519" else False
    )
    payload_with_mode = {**payload, "verification_mode": verification_mode}
    proof = {
        "payload": payload_with_mode,
        "signed_payload": payload,
        "signature": signature,
        "verified": False,
        "verification_mode": verification_mode,
        "signature_valid": signature_valid,
        "replay_valid": replay_valid,
        "signer_authorized": False,
    }
    signer_authorized = (
        trusted_signers is not None and
        isinstance(trusted_signers.get(signer_id), bytes) and
        trusted_signers[signer_id] == public_key_pem
    )
    proof["signer_authorized"] = bool(signer_authorized)
    proof["verified"] = bool(
        replay_valid and signature_valid and signer_authorized
        and authenticate_execution_proof(proof, trusted_signers)
    ) if trusted_signers is not None else False

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "execution_proof.json")
        with open(path, "w") as f:
            json.dump(proof, f, sort_keys=True, indent=2)

    return proof
