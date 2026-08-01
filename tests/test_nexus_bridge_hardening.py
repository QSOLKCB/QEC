from __future__ import annotations

from pathlib import Path
import stat

import pytest

from qec.adapters.nexus import (
    NEXUS_V3,
    NexusAttestationError,
    NexusEvidenceError,
    NexusExecutionError,
    NexusInvocation,
    build_attestation,
    run_nexus,
    validate_build_attestation,
    validate_execution_receipt,
    validate_receipt_file,
)
from qec.sonify.canonical import canonical_json, canonical_sha256


def _fake_binary(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "nexus"
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _attestation(tmp_path: Path, binary: Path) -> Path:
    path = tmp_path / "attestation.json"
    path.write_text(
        canonical_json(
            build_attestation(
                profile="v4.0.0",
                binary=binary,
                toolchain="test",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _verify_receipt(tmp_path: Path) -> dict[str, object]:
    body = """cat <<'CSV'
metric,value
logical,16777216
rendered,1024
particles,512
max_radius_error,0
max_antipodal_error,0
max_sampling_gap_error,0
centre_error,0
orientation_before,-1
orientation_at,0
orientation_after,1
particle_first,0
particle_last,1022
CSV
"""
    binary = _fake_binary(tmp_path, body)
    return run_nexus(
        NexusInvocation(operation="verify"),
        binary=binary,
        attestation_path=_attestation(tmp_path, binary),
        output_dir=tmp_path / "out",
    )


def _resign(receipt: dict[str, object]) -> None:
    unsigned = dict(receipt)
    unsigned.pop("sha256", None)
    receipt["sha256"] = canonical_sha256(unsigned)


def test_invocation_defaults_use_distinct_config_instances() -> None:
    first = NexusInvocation(operation="verify")
    second = NexusInvocation(operation="verify")
    assert first.config == second.config
    assert first.config is not second.config


def test_build_attestation_rejects_modified_binary(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path, "printf x")
    record = build_attestation(
        profile="v4.0.0",
        binary=binary,
        toolchain="test",
    )
    binary.write_text("#!/bin/sh\nprintf y", encoding="utf-8")
    with pytest.raises(NexusAttestationError):
        validate_build_attestation(
            record,
            profile="v4.0.0",
            binary=binary,
        )


def test_receipt_source_profile_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    receipt = _verify_receipt(tmp_path)
    receipt["source"] = NEXUS_V3.as_dict()
    _resign(receipt)
    with pytest.raises(NexusEvidenceError):
        validate_execution_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("adapter_only", False),
        ("decoder_mutation", True),
        ("physical_claim", True),
        ("qutrit_or_gf3_claim", True),
        ("receipt_proves", "physical_truth"),
        ("receipt_does_not_prove", "nothing"),
    ],
)
def test_resigned_claim_boundary_tampering_fails_closed(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    receipt = _verify_receipt(tmp_path)
    boundary = receipt["claim_boundary"]
    assert isinstance(boundary, dict)
    boundary[field] = replacement
    _resign(receipt)
    with pytest.raises(
        NexusEvidenceError,
        match="claim boundary does not match",
    ):
        validate_execution_receipt(receipt)


@pytest.mark.parametrize(
    "contents",
    ["{not-json", "[]"],
)
def test_receipt_file_errors_use_execution_error_surface(
    tmp_path: Path,
    contents: str,
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(NexusExecutionError):
        validate_receipt_file(path)


def test_missing_receipt_file_uses_execution_error_surface(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        NexusExecutionError,
        match="cannot read NEXUS execution receipt",
    ):
        validate_receipt_file(tmp_path / "missing.json")
