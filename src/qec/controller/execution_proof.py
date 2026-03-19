"""Signed execution proof layer (v81.1.0).

Constructs canonical proof payloads from verified replay results,
signs them deterministically using Ed25519, and writes proof artifacts.

Ed25519 signatures are inherently deterministic — identical payloads
with the same private key always produce the same signature.
"""

from __future__ import annotations

import json
import os
from typing import Any

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
except Exception:  # pragma: no cover
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

    Raises ``RuntimeError`` if the ``cryptography`` package is not usable.
    """
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError(
            "cryptography package is required for signing but is not available"
        )
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
        raise RuntimeError(
            "cryptography package is required for verification but is not available"
        )
    key = load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("Public key must be Ed25519")
    try:
        key.verify(bytes.fromhex(signature), payload_bytes)
        return True
    except Exception:
        return False


# -- Step 5: Main entry ------------------------------------------------------


def create_execution_proof(
    verify_result: dict,
    signer_id: str,
    private_key_pem: bytes,
    public_key_pem: bytes,
    metadata: dict[str, Any] | None = None,
    output_dir: str | None = None,
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
    dict  with keys ``payload``, ``signature``, ``verified``.
    """
    payload = build_proof_payload(verify_result, signer_id, metadata)
    payload_bytes = serialize_proof_payload(payload)
    signature = sign_payload(payload_bytes, private_key_pem)
    verified = verify_signature(payload_bytes, signature, public_key_pem)

    proof = {
        "payload": payload,
        "signature": signature,
        "verified": verified,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "execution_proof.json")
        with open(path, "w") as f:
            json.dump(proof, f, sort_keys=True, indent=2)

    return proof
