"""Canonical validation for the published NEXUS v4.0.1 qBraid evidence."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from qec.sonify.canonical import canonical_sha256
from .version import QEC_NEXUS_BRIDGE_VERSION

RECEIPT_SCHEMA = "qec.nexus-qbraid-replication-receipt.v1"
ARCHIVE_FILENAME = "nexus-v4-qbraid-rerun2-results-1e93a509.zip"
EXPECTED_ARCHIVE_SHA256 = "659e493a1b80b391db99b79dd6ee4e7a9b23c1821ff11eadbc3c5c36b10660d8"
EXPECTED_RECEIPT_SHA256 = "6137a128f73a950f0da12f54df10090005d23dfb5e771897e7af70fd55468dcd"
PUBLICATION_DOI = "10.5281/zenodo.21751929"
PUBLICATION_VERSION = "4.0.1"
PROTOCOL = "NEXUS-v4-qBraid-rerun2"
SOURCE_PROFILE = "v4.0.0"
SOURCE_COMMIT = "1e93a509a28144d70a17fa76b330ae042db7beab"
CONTROL_COMMIT = "a71e9f73aa000eb5ffb2138471c5d8f49a106ed7"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_UNLISTED = {"SHA256SUMS.txt", "00-provenance/manifest-verification.txt"}


class NexusReplicationError(ValueError):
    """Raised when replication evidence fails closed."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError as exc:
        raise NexusReplicationError(f"cannot read replication archive: {exc}") from exc
    return h.hexdigest()


def _object(data: bytes, label: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise NexusReplicationError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_float=str,
            object_pairs_hook=unique,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NexusReplicationError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise NexusReplicationError(f"{label} must contain a JSON object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NexusReplicationError(f"{label} must be non-empty text")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise NexusReplicationError(f"{label} must be an exact integer >= {minimum}")
    return value


def _status(data: bytes, label: str) -> int:
    try:
        text = data.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise NexusReplicationError(f"{label} must be ASCII") from exc
    if not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise NexusReplicationError(f"{label} must be canonical status text")
    return int(text)


def _metrics(data: bytes, label: str) -> dict[str, str]:
    try:
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise NexusReplicationError(f"cannot parse {label}: {exc}") from exc
    if not rows or set(rows[0]) != {"metric", "value"}:
        raise NexusReplicationError(f"{label} must be metric,value CSV")
    result: dict[str, str] = {}
    for row in rows:
        key, value = row.get("metric"), row.get("value")
        if not key or value is None or key in result:
            raise NexusReplicationError(f"{label} contains invalid metrics")
        result[key] = value
    return result


def _index(zf: zipfile.ZipFile) -> tuple[str, dict[str, zipfile.ZipInfo]]:
    infos = zf.infolist()
    if len(infos) > 512:
        raise NexusReplicationError("archive contains too many members")
    roots: set[str] = set()
    members: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise NexusReplicationError(f"unsafe ZIP path: {info.filename}")
        roots.add(path.parts[0])
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_IFMT(mode) == stat.S_IFLNK or info.flag_bits & 1:
            raise NexusReplicationError(f"unsafe ZIP member: {info.filename}")
        total += info.file_size
        if info.file_size > 64 * 1024 * 1024 or total > 128 * 1024 * 1024:
            raise NexusReplicationError("archive exceeds bounded size contract")
        if not info.is_dir():
            name = PurePosixPath(*path.parts[1:]).as_posix()
            if not name or name in members:
                raise NexusReplicationError(f"duplicate ZIP member: {info.filename}")
            members[name] = info
    if len(roots) != 1:
        raise NexusReplicationError("archive must contain one top-level directory")
    return next(iter(roots)), members


def _read(zf: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], name: str) -> bytes:
    if name not in members:
        raise NexusReplicationError(f"archive is missing {name}")
    return zf.read(members[name])


def _manifest(zf: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo]) -> dict[str, object]:
    raw = _read(zf, members, "SHA256SUMS.txt")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise NexusReplicationError("SHA256SUMS.txt must be UTF-8") from exc
    expected: dict[str, str] = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        name = name.removeprefix("./")
        path = PurePosixPath(name)
        if (
            not separator
            or not _SHA256.fullmatch(digest)
            or path.is_absolute()
            or ".." in path.parts
            or name in expected
        ):
            raise NexusReplicationError("invalid SHA256SUMS.txt entry")
        expected[name] = digest
    if not expected:
        raise NexusReplicationError("SHA256SUMS.txt may not be empty")
    for name, digest in expected.items():
        if _sha(_read(zf, members, name)) != digest:
            raise NexusReplicationError(f"manifest mismatch for {name}")
    unlisted = sorted(set(members) - set(expected))
    if set(unlisted) - _ALLOWED_UNLISTED:
        raise NexusReplicationError("archive contains unexpected unmanifested files")
    return {
        "manifest_sha256": _sha(raw),
        "listed_files": len(expected),
        "verified_files": len(expected),
        "allowed_unlisted_files": unlisted,
    }


def _performance(summary: dict[str, object]) -> dict[str, object]:
    scalar = summary.get("w1_scalar")
    parallel = summary.get("w7_parallel_batch")
    if not isinstance(scalar, dict) or not isinstance(parallel, dict):
        raise NexusReplicationError("performance summary lacks required modes")
    scalar_ns = _text(scalar.get("median_ns_per_eval"), "scalar median")
    parallel_ns = _text(parallel.get("median_ns_per_eval"), "parallel median")
    speedup = _text(parallel.get("speedup_vs_scalar"), "speedup")
    efficiency = _text(parallel.get("efficiency"), "efficiency")
    observations = _integer(parallel.get("observations"), "observations", 1)
    if not math.isclose(float(scalar_ns) / float(parallel_ns), float(speedup), rel_tol=1e-12):
        raise NexusReplicationError("speedup disagrees with measured medians")
    if not math.isclose(float(speedup) / 7, float(efficiency), rel_tol=1e-12):
        raise NexusReplicationError("efficiency disagrees with speedup")
    return {
        "observations_per_mode": observations,
        "scalar_median_ns_per_eval": scalar_ns,
        "parallel_7_median_ns_per_eval": parallel_ns,
        "speedup_at_7_workers": speedup,
        "efficiency_at_7_workers": efficiency,
    }


def validate_qbraid_replication_archive(
    archive_path: Path,
    *,
    expected_archive_sha256: str = EXPECTED_ARCHIVE_SHA256,
) -> dict[str, object]:
    if expected_archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise NexusReplicationError(
            "expected archive hash must equal the pinned publication SHA-256"
        )
    if not _SHA256.fullmatch(expected_archive_sha256):
        raise NexusReplicationError("expected archive hash must be full SHA-256")
    archive_sha = _file_sha(archive_path)
    if archive_sha != expected_archive_sha256:
        raise NexusReplicationError("replication archive SHA-256 mismatch")
    try:
        zf = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise NexusReplicationError(f"cannot open replication archive: {exc}") from exc
    with zf:
        root, members = _index(zf)
        manifest = _manifest(zf, members)
        results_raw = _read(zf, members, "RESULTS.json")
        provenance_raw = _read(zf, members, "00-provenance/VERSION_PROVENANCE.json")
        results = _object(results_raw, "RESULTS.json")
        provenance = _object(provenance_raw, "VERSION_PROVENANCE.json")
        if results.get("protocol") != PROTOCOL:
            raise NexusReplicationError("unexpected qBraid protocol")
        if results.get("tested_source_commit") != SOURCE_COMMIT:
            raise NexusReplicationError("tested source commit is not pinned NEXUS v4")
        if provenance.get("tested_source_commit") != SOURCE_COMMIT:
            raise NexusReplicationError("provenance commit disagrees with results")
        if results.get("control_document_commit") != CONTROL_COMMIT:
            raise NexusReplicationError("control document commit mismatch")
        package = results.get("package")
        if (
            not isinstance(package, dict)
            or package.get("name") != "nexus"
            or package.get("version") != "4.0.0"
        ):
            raise NexusReplicationError("unexpected frozen package identity")
        before = _read(zf, members, "00-provenance/source-before.sha256")
        after = _read(zf, members, "00-provenance/source-after.sha256")
        source_diff = _read(zf, members, "00-provenance/source-manifest-diff.txt")
        if before != after or source_diff.strip():
            raise NexusReplicationError("source tree changed during replication")
        primary = {
            name: _status(
                _read(zf, members, f"01-build/primary/{name}.exit-status"),
                name,
            )
            for name in ("format-check", "clippy", "build-release", "tests")
        }
        msrv = {
            name: _status(
                _read(zf, members, f"01-build/msrv-1.82.0/{name}.exit-status"),
                name,
            )
            for name in ("install", "build", "tests")
        }
        if any(primary.values()) or any(msrv.values()):
            raise NexusReplicationError("a required build or MSRV command failed")
        baseline = _metrics(
            _read(zf, members, "02-correctness/verify-nondivisible.csv"),
            "scalar verification",
        )
        ignored = {"workers", "elapsed_ns"}
        base_values = {k: v for k, v in baseline.items() if k not in ignored}
        for workers in (1, 2, 4, 7):
            candidate = _metrics(
                _read(zf, members, f"02-correctness/verify-parallel-w{workers}.csv"),
                f"parallel verification w{workers}",
            )
            values = {k: v for k, v in candidate.items() if k not in ignored}
            if candidate.get("workers") != str(workers) or values != base_values:
                raise NexusReplicationError("parallel verification differs from scalar baseline")
        observed = _metrics(
            _read(zf, members, "03-parallel/observed-run.csv"),
            "observed run",
        )
        observed_status = _status(
            _read(zf, members, "03-parallel/observed-run.exit-status"),
            "observed run",
        )
        if observed.get("workers") != "7" or observed_status:
            raise NexusReplicationError("seven-worker observation did not complete")
        try:
            samples = _read(
                zf,
                members,
                "03-parallel/thread-observation/samples.txt",
            ).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NexusReplicationError("thread samples must be UTF-8") from exc
        threads = [
            int(value)
            for value in re.findall(r"(?m)^Threads:\s*([0-9]+)$", samples)
        ]
        if not threads or max(threads) != 8:
            raise NexusReplicationError("thread samples do not show main plus seven workers")
        performance = _performance(
            _object(
                _read(zf, members, "04-performance/performance-summary.json"),
                "performance summary",
            )
        )
        ternary = [
            _sha(_read(zf, members, f"05-determinism/ternary-{index}.csv"))
            for index in (1, 2, 3)
        ]
        receipts = [
            _sha(
                _read(
                    zf,
                    members,
                    f"05-determinism/receipt-100000-{index}.csv",
                )
            )
            for index in (1, 2, 3)
        ]
        variants = {
            _sha(_read(zf, members, "05-determinism/receipt-100001.csv")),
            _sha(_read(zf, members, "05-determinism/receipt-phase-change.csv")),
        }
        if (
            len(set(ternary)) != 1
            or len(set(receipts)) != 1
            or receipts[0] in variants
            or len(variants) != 2
        ):
            raise NexusReplicationError("determinism or receipt sensitivity check failed")
        overflow = _status(
            _read(zf, members, "02-correctness/ternary-overflow.exit-status"),
            "ternary overflow",
        )
        overflow_error = _read(
            zf,
            members,
            "02-correctness/ternary-overflow.stderr",
        )
        if overflow == 0 or b"frequency must be finite" not in overflow_error:
            raise NexusReplicationError("ternary overflow was not rejected")
        visual_markers = (
            (
                "07-visual/d1.svg",
                (b"D1-fluxtube", b'data-mouth-count="2"', b"central-throat"),
            ),
            (
                "07-visual/d2.svg",
                (b"D2-twisted-fluxtube", b'data-mouth-count="2"', b"central-knot"),
            ),
        )
        for name, markers in visual_markers:
            data = _read(zf, members, name)
            if any(marker not in data for marker in markers):
                raise NexusReplicationError(f"visual contract failed for {name}")
        failed = _read(zf, members, "01-build/primary/FAILED.txt")
        anomalies: list[dict[str, object]] = []
        if failed.strip():
            anomalies.append(
                {
                    "code": "stale_primary_failure_marker",
                    "path": "01-build/primary/FAILED.txt",
                    "content_sha256": _sha(failed),
                    "reported_text": failed.decode(
                        "utf-8",
                        errors="replace",
                    ).strip(),
                    "contradicts": "all primary exit-status files are zero",
                    "effect": "blanket VALID/no-anomalies claim is not accepted",
                }
            )
        platform = results.get("platform")
        if not isinstance(platform, dict):
            raise NexusReplicationError("platform record is missing")
        receipt: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "qec_version": QEC_NEXUS_BRIDGE_VERSION,
            "publication": {
                "version": PUBLICATION_VERSION,
                "doi": PUBLICATION_DOI,
                "classification": "replication_evidence_release",
            },
            "source": {
                "profile": SOURCE_PROFILE,
                "package_version": "4.0.0",
                "commit": SOURCE_COMMIT,
            },
            "protocol": PROTOCOL,
            "archive": {
                "filename": ARCHIVE_FILENAME,
                "external_sidecar_match": True,
                "sha256": archive_sha,
                "top_level_directory": root,
                "member_files": len(members),
                **manifest,
                "results_sha256": _sha(results_raw),
                "provenance_sha256": _sha(provenance_raw),
                "report_sha256": _sha(_read(zf, members, "REPORT.md")),
            },
            "environment": {
                "platform": "qBraid",
                "os": _text(platform.get("os"), "operating system"),
                "kernel": _text(platform.get("kernel"), "kernel"),
                "cpu_model": _text(platform.get("cpu_model"), "CPU model"),
                "online_logical_cpus": _integer(
                    platform.get("online_logical_cpus"),
                    "logical CPUs",
                    1,
                ),
                "effective_worker_capacity": 7,
            },
            "verification": {
                "archive_integrity": "verified",
                "source_tree_unchanged": True,
                "primary_exit_statuses": primary,
                "msrv_1_82_0_exit_statuses": msrv,
                "parallel_scalar_equivalence": True,
                "ternary_deterministic": True,
                "receipt_deterministic_and_input_sensitive": True,
                "ternary_overflow_rejected": True,
                "visual_contracts_verified": True,
            },
            "thread_observation": {
                "requested_workers": 7,
                "max_threads_observed": 8,
                "interpretation": "one main thread plus seven worker threads observed",
                "multicore_execution_classification": (
                    "supported_by_thread_observation_and_speedup"
                ),
            },
            "performance": performance,
            "claims": {
                "reported_run_status": _text(
                    results.get("run_status"),
                    "reported run status",
                ),
                "usable_verified_evidence": True,
                "blanket_valid_claim_accepted": not anomalies,
                "whole_bundle_status": (
                    "verified"
                    if not anomalies
                    else "verified_with_declared_anomaly"
                ),
                "universal_performance_claim": False,
                "physical_or_quantum_claim": False,
            },
            "anomalies": anomalies,
            "claim_boundary": {
                "proves": (
                    "archive identity, listed-artifact integrity, and declared "
                    "implementation observations"
                ),
                "does_not_prove": (
                    "universal performance, physical truth, qutrit physics, "
                    "or quantum advantage"
                ),
            },
        }
        receipt["sha256"] = canonical_sha256(receipt)
        return receipt


def validate_replication_receipt(receipt: dict[str, object]) -> dict[str, object]:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise NexusReplicationError("unexpected replication receipt schema")
    observed = receipt.get("sha256")
    if not isinstance(observed, str) or not _SHA256.fullmatch(observed):
        raise NexusReplicationError("replication receipt requires a full SHA-256")
    unsigned = dict(receipt)
    unsigned.pop("sha256", None)
    if canonical_sha256(unsigned) != observed:
        raise NexusReplicationError("replication receipt hash mismatch")
    if receipt.get("qec_version") != QEC_NEXUS_BRIDGE_VERSION:
        raise NexusReplicationError("unexpected QEC version in replication receipt")
    if receipt.get("publication") != {
        "version": PUBLICATION_VERSION,
        "doi": PUBLICATION_DOI,
        "classification": "replication_evidence_release",
    }:
        raise NexusReplicationError("replication publication identity mismatch")
    if receipt.get("source") != {
        "profile": SOURCE_PROFILE,
        "package_version": "4.0.0",
        "commit": SOURCE_COMMIT,
    }:
        raise NexusReplicationError("replication source identity mismatch")
    archive = receipt.get("archive")
    claims = receipt.get("claims")
    anomalies = receipt.get("anomalies")
    if (
        not isinstance(archive, dict)
        or archive.get("sha256") != EXPECTED_ARCHIVE_SHA256
    ):
        raise NexusReplicationError("replication archive identity mismatch")
    if not isinstance(claims, dict) or not isinstance(anomalies, list) or not anomalies:
        raise NexusReplicationError("replication claim or anomaly block is missing")
    if (
        claims.get("usable_verified_evidence") is not True
        or claims.get("blanket_valid_claim_accepted") is not False
        or claims.get("whole_bundle_status")
        != "verified_with_declared_anomaly"
    ):
        raise NexusReplicationError("replication claim classification mismatch")
    expected_boundary = {
        "proves": (
            "archive identity, listed-artifact integrity, and declared "
            "implementation observations"
        ),
        "does_not_prove": (
            "universal performance, physical truth, qutrit physics, "
            "or quantum advantage"
        ),
    }
    if receipt.get("claim_boundary") != expected_boundary:
        raise NexusReplicationError("replication claim boundary mismatch")
    # A self-hash proves only internal consistency. Pin the published receipt
    # identity so re-signed changes to any evidence-bearing field fail closed.
    if observed != EXPECTED_RECEIPT_SHA256:
        raise NexusReplicationError("replication receipt identity mismatch")
    return {
        "valid": True,
        "sha256": observed,
        "status": claims["whole_bundle_status"],
    }
