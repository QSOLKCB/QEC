from __future__ import annotations

from pathlib import Path
import stat

import pytest

import qec.adapters.nexus.runner as nexus_runner
from qec.adapters.nexus import (
    NEXUS_V3,
    NexusAttestationError,
    NexusConfig,
    NexusEvidenceError,
    NexusExecutionError,
    NexusInvocation,
    build_attestation,
    build_execution_receipt,
    run_nexus,
    validate_build_attestation,
    validate_execution_receipt,
    validate_receipt_file,
)
from qec.sonify.canonical import canonical_json, canonical_sha256

_VERIFY_CSV = """metric,value
logical,16777216
rendered,1024
particles,512
workers,1
max_radius_error,0
max_antipodal_error,0
max_sampling_gap_error,0
centre_error,0
orientation_before,-1
orientation_at,0
orientation_after,1
particle_first,0
particle_last,1022
"""


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


def _verify_bundle(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    body = "cat <<'CSV'\n" + _VERIFY_CSV + "CSV\n"
    binary = _fake_binary(tmp_path, body)
    output = tmp_path / "out"
    receipt = run_nexus(
        NexusInvocation(operation="verify"),
        binary=binary,
        attestation_path=_attestation(tmp_path, binary),
        output_dir=output,
    )
    return receipt, output, binary


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
    receipt, _, _ = _verify_bundle(tmp_path)
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
    receipt, _, _ = _verify_bundle(tmp_path)
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


def test_bundle_validation_recomputes_bound_evidence(tmp_path: Path) -> None:
    _, output, _ = _verify_bundle(tmp_path)
    result = validate_receipt_file(output / "execution_receipt.json")
    assert result["bundle_verified"] is True


def test_resigned_forged_invariants_fail_bundle_validation(
    tmp_path: Path,
) -> None:
    receipt, output, _ = _verify_bundle(tmp_path)
    invariants = receipt["invariants"]
    assert isinstance(invariants, dict)
    invariants["row_count"] = 999
    _resign(receipt)
    (output / "execution_receipt.json").write_text(
        canonical_json(receipt) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        NexusEvidenceError,
        match="does not reproduce receipt invariants",
    ):
        validate_receipt_file(output / "execution_receipt.json")


def test_builder_derives_rows_from_bound_stdout() -> None:
    with pytest.raises(NexusEvidenceError):
        build_execution_receipt(
            NexusInvocation(operation="verify"),
            stdout_bytes=b"not-a-csv-table\n",
            binary_sha256="a" * 64,
            build_attestation_sha256="b" * 64,
        )


def test_nonfinite_or_malformed_coordinates_fail_every_trace_row(
    tmp_path: Path,
) -> None:
    body = """cat <<'CSV'
step,progress,x,y,radius,orientation
0,0,not-a-number,0,0.56,-1
1,0.5,0,0,0,0
2,1,0.56,0,0.56,1
CSV
"""
    binary = _fake_binary(tmp_path, body)
    with pytest.raises(NexusEvidenceError, match="x must be finite"):
        run_nexus(
            NexusInvocation(operation="trace", channel=0, steps=2),
            binary=binary,
            attestation_path=_attestation(tmp_path, binary),
            output_dir=tmp_path / "trace",
        )


def test_parallel_evidence_binds_requested_worker_count(
    tmp_path: Path,
) -> None:
    body = "cat <<'CSV'\n" + _VERIFY_CSV + "CSV\n"
    binary = _fake_binary(tmp_path, body)
    with pytest.raises(
        NexusEvidenceError,
        match="requested worker count does not match",
    ):
        run_nexus(
            NexusInvocation(operation="verify-parallel", workers=2),
            binary=binary,
            attestation_path=_attestation(tmp_path, binary),
            output_dir=tmp_path / "parallel",
        )


def test_fibonacci_policy_is_recomputed_from_channel(
    tmp_path: Path,
) -> None:
    body = """cat <<'CSV'
step,progress,x,y,radius,orientation,trit,trit_value,lane,fib_fraction,logical_index
0,0,0.28,-0.48497422611928565,0.56,-1,inbound,-1,lane-1,0,0
1,0.5,0,0,0,0,nexus,0,lane-1,0,0
2,1,-0.28,0.48497422611928565,0.56,1,outbound,1,lane-1,0,0
CSV
"""
    binary = _fake_binary(tmp_path, body)
    invocation = NexusInvocation(
        operation="fibonacci",
        channel=1,
        steps=2,
        config=NexusConfig(logical=3, rendered=3, particles=3),
    )
    with pytest.raises(
        NexusEvidenceError,
        match="logical index disagrees with phi policy",
    ):
        run_nexus(
            invocation,
            binary=binary,
            attestation_path=_attestation(tmp_path, binary),
            output_dir=tmp_path / "fibonacci",
        )


def test_subprocess_stdout_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = _fake_binary(
        tmp_path,
        "printf '1234567890123456789012345678901234567890'",
    )
    monkeypatch.setattr(nexus_runner, "MAX_STDOUT_BYTES", 32)
    with pytest.raises(NexusExecutionError, match="stdout exceeded"):
        run_nexus(
            NexusInvocation(operation="verify"),
            binary=binary,
            attestation_path=_attestation(tmp_path, binary),
            output_dir=tmp_path / "bounded",
        )


def test_executed_binary_is_private_validated_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, original = _verify_bundle(tmp_path)
    assert receipt["invariants"]["centre_exact"] is True
    attestation_path = _attestation(tmp_path, original)
    real_validate = nexus_runner.validate_build_attestation

    def validate_then_replace(
        attestation: dict[str, object],
        *,
        profile: str,
        binary: Path,
    ) -> dict[str, object]:
        result = real_validate(attestation, profile=profile, binary=binary)
        original.write_text(
            "#!/bin/sh\nprintf 'malicious replacement'",
            encoding="utf-8",
        )
        original.chmod(original.stat().st_mode | stat.S_IXUSR)
        return result

    monkeypatch.setattr(
        nexus_runner,
        "validate_build_attestation",
        validate_then_replace,
    )
    second = run_nexus(
        NexusInvocation(operation="verify"),
        binary=original,
        attestation_path=attestation_path,
        output_dir=tmp_path / "private-copy",
    )
    assert second["invariants"]["centre_exact"] is True
