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


def _claim_boundary() -> dict[str, str]:
    return {
        "attests": "declared_source_checkout_and_observed_binary_bytes",
        "does_not_attest": (
            "cross_environment_binary_reproducibility_or_physical_truth"
        ),
    }


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise NexusAttestationError(f"NEXUS binary does not exist: {path}")
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise NexusAttestationError(
            f"cannot read NEXUS binary: {path}: {exc}"
        ) from exc
    return hasher.hexdigest()


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
            "source_checkout_commit": source.commit,
            "source_checkout_binding": "caller_attested",
            "reproducible_binary_claim": False,
        },
        "claim_boundary": _claim_boundary(),
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def validate_build_attestation_record(
    attestation: dict[str, object],
    *,
    profile: str,
) -> dict[str, object]:
    if attestation.get("schema") != "qec.nexus-build-attestation.v1":
        raise NexusAttestationError(
            "unexpected NEXUS build-attestation schema"
        )
    if attestation.get("qec_version") != "170.2.0":
        raise NexusAttestationError(
            "unexpected QEC version in build attestation"
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
    build = attestation.get("build")
    if not isinstance(build, dict):
        raise NexusAttestationError(
            "build attestation build record is missing"
        )
    if not isinstance(build.get("toolchain"), str) or not str(
        build["toolchain"]
    ).strip():
        raise NexusAttestationError(
            "build attestation toolchain must be non-empty text"
        )
    if build.get("source_checkout_commit") != expected_source["commit"]:
        raise NexusAttestationError(
            "build attestation checkout commit does not match profile"
        )
    if build.get("source_checkout_binding") != "caller_attested":
        raise NexusAttestationError(
            "build attestation checkout binding is invalid"
        )
    if build.get("reproducible_binary_claim") is not False:
        raise NexusAttestationError(
            "build attestation may not claim reproducible binary bytes"
        )
    if attestation.get("claim_boundary") != _claim_boundary():
        raise NexusAttestationError(
            "build attestation claim boundary does not match contract"
        )
    binary_record = attestation.get("binary")
    if not isinstance(binary_record, dict):
        raise NexusAttestationError(
            "build attestation binary record is missing"
        )
    name = binary_record.get("name")
    if not isinstance(name, str) or not name.strip():
        raise NexusAttestationError(
            "build attestation binary name must be non-empty text"
        )
    digest = binary_record.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise NexusAttestationError(
            "build attestation binary requires a full SHA-256"
        )
    return {
        "valid": True,
        "sha256": observed,
        "binary_sha256": digest,
    }


def validate_build_attestation(
    attestation: dict[str, object],
    *,
    profile: str,
    binary: Path,
) -> dict[str, object]:
    result = validate_build_attestation_record(
        attestation,
        profile=profile,
    )
    if result["binary_sha256"] != file_sha256(binary):
        raise NexusAttestationError(
            "NEXUS binary bytes do not match build attestation"
        )
    return result
