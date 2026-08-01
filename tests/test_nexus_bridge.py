from __future__ import annotations

from pathlib import Path
import stat

import pytest

from qec.adapters.nexus import (
    NEXUS_V3,
    NEXUS_V4,
    NexusEvidenceError,
    NexusInvocation,
    build_attestation,
    run_nexus,
    validate_execution_receipt,
)
from qec.sonify.canonical import canonical_json


def _fake_binary(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "nexus"
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _attestation(
    tmp_path: Path,
    binary: Path,
    profile: str = "v4.0.0",
) -> Path:
    path = tmp_path / "attestation.json"
    path.write_text(
        canonical_json(
            build_attestation(
                profile=profile,
                binary=binary,
                toolchain="test",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_source_profiles_pin_published_v3_and_pending_v4() -> None:
    assert NEXUS_V3.doi == "10.5281/zenodo.21745329"
    assert (
        NEXUS_V3.commit
        == "e078b135322dc12a2565b9c512fc4ba75193dea7"
    )
    assert (
        NEXUS_V4.commit
        == "1e93a509a28144d70a17fa76b330ae042db7beab"
    )
    assert NEXUS_V4.doi is None
    assert NEXUS_V4.doi_status == "pending"


def test_v3_cannot_claim_v4_capabilities() -> None:
    with pytest.raises(ValueError, match="does not support ternary"):
        NexusInvocation(
            operation="ternary",
            profile="v3.0.0",
            channel=1,
            steps=2,
            base_frequency_hz="432",
        )


def test_invocation_argv_is_deterministic() -> None:
    invocation = NexusInvocation(
        operation="ternary",
        channel=17,
        steps=1000,
        base_frequency_hz="432",
    )
    argv = invocation.argv("/nexus")
    assert argv[:5] == ["/nexus", "ternary", "17", "1000", "432"]
    assert argv[-6:] == [
        "--radius",
        "0.56",
        "--phase",
        "0",
        "--turns",
        "1.5",
    ]


def test_verify_run_writes_hash_bound_bundle(tmp_path: Path) -> None:
    body = """cat <<'CSV'
metric,value
logical,16777216
rendered,1024
particles,512
workers,1
max_radius_error,1e-16
max_antipodal_error,0
max_sampling_gap_error,0
centre_error,0
orientation_before,-0.7
orientation_at,0
orientation_after,0.7
particle_first,0
particle_last,1022
CSV
"""
    binary = _fake_binary(tmp_path, body)
    receipt = run_nexus(
        NexusInvocation(operation="verify"),
        binary=binary,
        attestation_path=_attestation(tmp_path, binary),
        output_dir=tmp_path / "out",
    )
    assert receipt["invariants"]["centre_exact"] is True
    assert validate_execution_receipt(receipt)["valid"] is True
    assert (tmp_path / "out" / "execution_receipt.json").exists()


def test_ternary_semantics_are_recomputed(tmp_path: Path) -> None:
    body = """cat <<'CSV'
step,progress,x,y,radius,orientation,trit,trit_value,lane,frequency_hz,amplitude,pan
0,0,-0.56,0,0.56,-1,inbound,-1,lane-2,432,1,-1
1,0.5,0,0,0,0,nexus,0,lane-2,432,1,0
2,1,0.56,0,0.56,1,outbound,1,lane-2,432,1,1
CSV
"""
    binary = _fake_binary(tmp_path, body)
    invocation = NexusInvocation(
        operation="ternary",
        channel=17,
        steps=2,
        base_frequency_hz="432",
    )
    receipt = run_nexus(
        invocation,
        binary=binary,
        attestation_path=_attestation(tmp_path, binary),
        output_dir=tmp_path / "ternary",
    )
    assert receipt["invariants"]["triality_lane_verified"] is True


def test_receipt_rows_match_nexus_v4_contract(tmp_path: Path) -> None:
    body = """cat <<'CSV'
name,bytes,sha256
lane-0,1,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
lane-1,1,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
lane-2,1,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
all-lanes,3,dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
chain,4,eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
CSV
"""
    binary = _fake_binary(tmp_path, body)
    receipt = run_nexus(
        NexusInvocation(operation="receipt", samples=3),
        binary=binary,
        attestation_path=_attestation(tmp_path, binary),
        output_dir=tmp_path / "receipts",
    )
    assert receipt["invariants"]["chain_receipt"] is True


def test_tampered_receipt_fails_closed(tmp_path: Path) -> None:
    body = """cat <<'CSV'
name,bytes,sha256
lane-0,1,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
lane-1,1,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
lane-2,1,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
all-lanes,3,dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
chain,4,eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
CSV
"""
    binary = _fake_binary(tmp_path, body)
    receipt = run_nexus(
        NexusInvocation(operation="receipt", samples=3),
        binary=binary,
        attestation_path=_attestation(tmp_path, binary),
        output_dir=tmp_path / "receipts",
    )
    receipt["claim_boundary"]["physical_claim"] = True
    with pytest.raises(NexusEvidenceError):
        validate_execution_receipt(receipt)
