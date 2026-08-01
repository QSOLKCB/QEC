"""Hash-bound build attestations for pinned NEXUS binaries."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from qec.sonify.canonical import canonical_sha256
from .source import source_profile

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class NexusAttestationError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise NexusAttestationError(f"NEXUS binary does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_attestation(
    *,
    profile: str,
    binary: Path,
    toolchain: str,
) -> dict[str, object]:
    if not isinstance(toolchain, str) or not toolchain.strip():
        raise NexusAttestationError("toolchain must be non-empty text")
    source = source_profile(profile)
    payload: dict[str, object] = {
        "schema": "qec.nexus-build-attestation.v1",
        "qec_version": "170.2.0",
        "source": source.as_dict(),
        "binary": {
            "name": binary.name,
            "sha256": file_sha256(binary),
        },
        "build": {
            "toolchain": toolchain,
            "source_checkout_pinned": True,
            "reproducible_binary_claim": False,
        },
        "claim_boundary": {
            "attests": "declared_source_checkout_and_observed_binary_bytes",
            "does_not_attest": (
                "cross_environment_binary_reproducibility_or_physical_truth"
            ),
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def validate_build_attestation(
    attestation: dict[str, object],
    *,
    profile: str,
    binary: Path,
) -> dict[str, object]:
    if attestation.get("schema") != "qec.nexus-build-attestation.v1":
        raise NexusAttestationError(
            "unexpected NEXUS build-attestation schema"
        )
    observed = attestation.get("sha256")
    if not isinstance(observed, str) or not _SHA256_RE.fullmatch(observed):
        raise NexusAttestationError(
            "build attestation requires a full SHA-256 identity"
        )
    unsigned = dict(attestation)
    unsigned.pop("sha256", None)
    if canonical_sha256(unsigned) != observed:
        raise NexusAttestationError("NEXUS build-attestation hash mismatch")
    source = attestation.get("source")
    expected_source = source_profile(profile).as_dict()
    if source != expected_source:
        raise NexusAttestationError(
            "build attestation source identity does not match profile"
        )
    binary_record = attestation.get("binary")
    if not isinstance(binary_record, dict):
        raise NexusAttestationError(
            "build attestation binary record is missing"
        )
    digest = binary_record.get("sha256")
    if digest != file_sha256(binary):
        raise NexusAttestationError(
            "NEXUS binary bytes do not match build attestation"
        )
    return {
        "valid": True,
        "sha256": observed,
        "binary_sha256": digest,
    }
