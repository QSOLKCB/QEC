from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from qec.adapters.nexus import (
    EXPECTED_ARCHIVE_SHA256,
    NEXUS_EXECUTION_CONTRACT_VERSION,
    NEXUS_REPLICATION_RECEIPT_VERSION,
    NexusReplicationError,
    validate_qbraid_replication_archive,
    validate_replication_receipt,
)
from qec.sonify.canonical import canonical_sha256

ROOT = "nexus-v4-qbraid-rerun2-results-1e93a509"
SOURCE_COMMIT = "1e93a509a28144d70a17fa76b330ae042db7beab"
CONTROL_COMMIT = "a71e9f73aa000eb5ffb2138471c5d8f49a106ed7"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metrics(workers: int, *, elapsed: bool = False) -> bytes:
    rows = [
        "metric,value",
        "logical,1000003",
        "rendered,997",
        "particles,128",
        f"workers,{workers}",
    ]
    if elapsed:
        rows.append("elapsed_ns,123")
    rows.extend(
        [
            "max_radius_error,1.11022302462515654e-16",
            "max_antipodal_error,0.00000000000000000e0",
            "max_sampling_gap_error,1",
            "centre_error,0.00000000000000000e0",
            "orientation_before,-7.07106781186547462e-1",
            "orientation_at,0.00000000000000000e0",
            "orientation_after,7.07106781186547462e-1",
            "particle_first,0",
            "particle_last,989",
        ]
    )
    return ("\n".join(rows) + "\n").encode()


def _write_bundle(tmp_path: Path) -> tuple[Path, str]:
    results = {
        "run_status": "VALID",
        "protocol": "NEXUS-v4-qBraid-rerun2",
        "tested_source_commit": SOURCE_COMMIT,
        "control_document_commit": CONTROL_COMMIT,
        "package": {"name": "nexus", "version": "4.0.0"},
        "platform": {
            "os": "Ubuntu 24.04.4 LTS",
            "kernel": "6.8.0-1059-azure",
            "cpu_model": "AMD EPYC 7763 64-Core Processor",
            "online_logical_cpus": 16,
        },
    }
    provenance = {"tested_source_commit": SOURCE_COMMIT}
    performance = {
        "w1_scalar": {"median_ns_per_eval": 70.0},
        "w7_parallel_batch": {
            "median_ns_per_eval": 20.0,
            "speedup_vs_scalar": 3.5,
            "efficiency": 0.5,
            "observations": 5,
        },
    }
    files: dict[str, bytes] = {
        "RESULTS.json": json.dumps(results).encode(),
        "REPORT.md": b"synthetic evidence\n",
        "00-provenance/VERSION_PROVENANCE.json": json.dumps(provenance).encode(),
        "00-provenance/source-before.sha256": b"same\n",
        "00-provenance/source-after.sha256": b"same\n",
        "00-provenance/source-manifest-diff.txt": b"",
        "01-build/primary/FAILED.txt": b"required command failed: format-check\n",
        "02-correctness/verify-nondivisible.csv": _metrics(1),
        "03-parallel/observed-run.csv": _metrics(7, elapsed=True),
        "03-parallel/observed-run.exit-status": b"0\n",
        "03-parallel/thread-observation/samples.txt": b"Threads:\t8\n",
        "04-performance/performance-summary.json": json.dumps(performance).encode(),
        "02-correctness/ternary-overflow.exit-status": b"1\n",
        "02-correctness/ternary-overflow.stderr": b"frequency must be finite\n",
        "07-visual/d1.svg": (
            b'<svg data-structure="D1-fluxtube" data-mouth-count="2" '
            b'id="central-throat" />'
        ),
        "07-visual/d2.svg": (
            b'<svg data-structure="D2-twisted-fluxtube" '
            b'data-mouth-count="2" id="central-knot" />'
        ),
    }
    for name in ("format-check", "clippy", "build-release", "tests"):
        files[f"01-build/primary/{name}.exit-status"] = b"0\n"
    for name in ("install", "build", "tests"):
        files[f"01-build/msrv-1.82.0/{name}.exit-status"] = b"0\n"
    for workers in (1, 2, 4, 7):
        files[f"02-correctness/verify-parallel-w{workers}.csv"] = _metrics(
            workers,
            elapsed=True,
        )
    for index in (1, 2, 3):
        files[f"05-determinism/ternary-{index}.csv"] = b"same ternary\n"
        files[f"05-determinism/receipt-100000-{index}.csv"] = b"same receipt\n"
    files["05-determinism/receipt-100001.csv"] = b"different samples\n"
    files["05-determinism/receipt-phase-change.csv"] = b"different phase\n"

    manifest = "".join(
        f"{_sha(data)}  ./{name}\n"
        for name, data in sorted(files.items())
    ).encode()
    files["SHA256SUMS.txt"] = manifest
    files["00-provenance/manifest-verification.txt"] = b"all listed files: OK\n"

    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(f"{ROOT}/{name}", data)
    return archive, _sha(archive.read_bytes())


def test_execution_and_replication_versions_are_separate() -> None:
    assert NEXUS_EXECUTION_CONTRACT_VERSION == "170.2.0"
    assert NEXUS_REPLICATION_RECEIPT_VERSION == "170.2.1"


def test_qbraid_archive_builds_anomaly_preserving_receipt(
    tmp_path: Path,
) -> None:
    archive, digest = _write_bundle(tmp_path)
    receipt = validate_qbraid_replication_archive(
        archive,
        expected_archive_sha256=digest,
    )
    assert receipt["claims"]["usable_verified_evidence"] is True
    assert receipt["claims"]["blanket_valid_claim_accepted"] is False
    assert receipt["claims"]["whole_bundle_status"] == (
        "verified_with_declared_anomaly"
    )
    assert receipt["anomalies"][0]["code"] == (
        "stale_primary_failure_marker"
    )


def test_committed_receipt_is_canonical_and_valid() -> None:
    path = Path("docs/replications/nexus_v4_0_1_qbraid_receipt.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256
    assert receipt["sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    assert validate_replication_receipt(receipt)["valid"] is True


def test_resigned_blanket_valid_claim_is_rejected() -> None:
    path = Path("docs/replications/nexus_v4_0_1_qbraid_receipt.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["claims"]["blanket_valid_claim_accepted"] = True
    unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
    receipt["sha256"] = canonical_sha256(unsigned)
    with pytest.raises(
        NexusReplicationError,
        match="claim classification mismatch",
    ):
        validate_replication_receipt(receipt)


def test_archive_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    archive, _ = _write_bundle(tmp_path)
    with pytest.raises(
        NexusReplicationError,
        match="archive SHA-256 mismatch",
    ):
        validate_qbraid_replication_archive(archive)


def test_zip_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("root/../escape", b"bad")
    digest = _sha(archive.read_bytes())
    with pytest.raises(NexusReplicationError, match="unsafe ZIP path"):
        validate_qbraid_replication_archive(
            archive,
            expected_archive_sha256=digest,
        )
