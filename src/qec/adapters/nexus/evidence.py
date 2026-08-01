"""NEXUS CSV parsing, invariant verification, and QEC parent receipts."""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import re
from typing import Iterable

from qec.sonify.canonical import canonical_sha256
from .contract import NexusInvocation

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class NexusEvidenceError(ValueError):
    pass


def parse_csv(text: str) -> list[dict[str, str]]:
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except csv.Error as exc:
        raise NexusEvidenceError("invalid NEXUS CSV") from exc
    if not rows or not rows[0]:
        raise NexusEvidenceError(
            "NEXUS output must contain a non-empty CSV table"
        )
    if any(None in row for row in rows):
        raise NexusEvidenceError("NEXUS CSV contains excess columns")
    return rows


def _decimal(value: str, field: str) -> Decimal:
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise NexusEvidenceError(
            f"{field} must be finite decimal text"
        ) from exc
    if not number.is_finite():
        raise NexusEvidenceError(f"{field} must be finite")
    return number


def _exact_int(value: str, field: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(
        r"-?(0|[1-9][0-9]*)",
        value,
    ):
        raise NexusEvidenceError(
            f"{field} must be canonical integer text"
        )
    return int(value)


def _require_fields(row: dict[str, str], fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise NexusEvidenceError(
            f"missing NEXUS CSV fields: {', '.join(missing)}"
        )


def verify_evidence(
    invocation: NexusInvocation,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    operation = invocation.operation
    if operation in {"verify", "verify-parallel"}:
        return _verify_metrics(invocation, rows)
    if operation == "trace":
        return _verify_trace(invocation, rows, mode="trace")
    if operation == "fibonacci":
        return _verify_trace(invocation, rows, mode="fibonacci")
    if operation == "ternary":
        return _verify_trace(invocation, rows, mode="ternary")
    if operation == "receipt":
        return _verify_receipts(rows)
    raise NexusEvidenceError(f"no evidence verifier for {operation}")


def _verify_metrics(
    invocation: NexusInvocation,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    _require_fields(rows[0], ("metric", "value"))
    metrics: dict[str, str] = {}
    for row in rows:
        key = row["metric"]
        if not key or key in metrics:
            raise NexusEvidenceError(
                "verification metrics must have unique non-empty names"
            )
        metrics[key] = row["value"]
    required = (
        "logical",
        "rendered",
        "particles",
        "max_radius_error",
        "max_antipodal_error",
        "max_sampling_gap_error",
        "centre_error",
        "orientation_before",
        "orientation_at",
        "orientation_after",
        "particle_first",
        "particle_last",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise NexusEvidenceError(
            f"missing verification metrics: {', '.join(missing)}"
        )
    if _exact_int(metrics["logical"], "logical") != invocation.config.logical:
        raise NexusEvidenceError(
            "logical cardinality does not match invocation"
        )
    if _exact_int(
        metrics["rendered"],
        "rendered",
    ) != invocation.config.rendered:
        raise NexusEvidenceError(
            "rendered cardinality does not match invocation"
        )
    if _exact_int(
        metrics["particles"],
        "particles",
    ) != invocation.config.particles:
        raise NexusEvidenceError("particle count does not match invocation")
    centre = _decimal(metrics["centre_error"], "centre_error")
    antipodal = _decimal(
        metrics["max_antipodal_error"],
        "max_antipodal_error",
    )
    before = _decimal(metrics["orientation_before"], "orientation_before")
    at = _decimal(metrics["orientation_at"], "orientation_at")
    after = _decimal(metrics["orientation_after"], "orientation_after")
    gap = _exact_int(
        metrics["max_sampling_gap_error"],
        "max_sampling_gap_error",
    )
    radius_error = _decimal(
        metrics["max_radius_error"],
        "max_radius_error",
    )
    if centre != 0 or antipodal != 0:
        raise NexusEvidenceError(
            "exact centre and antipodal invariants must be zero"
        )
    if not (before < 0 and at == 0 and after > 0):
        raise NexusEvidenceError(
            "orientation must change sign exactly at the centre"
        )
    if radius_error < 0 or gap < 0 or gap > 1:
        raise NexusEvidenceError(
            "radius and floor-sampling residuals are outside contract"
        )
    return {
        "centre_exact": True,
        "antipodal_exact": True,
        "orientation_sign_change": True,
        "floor_sampling_gap_bounded": True,
        "row_count": len(rows),
    }


def _verify_trace(
    invocation: NexusInvocation,
    rows: list[dict[str, str]],
    *,
    mode: str,
) -> dict[str, object]:
    ternary = mode in {"ternary", "fibonacci"}
    fields = ["step", "progress", "x", "y", "radius", "orientation"]
    if ternary:
        fields.extend(("trit", "trit_value", "lane"))
    if mode == "fibonacci":
        fields.extend(("fib_fraction", "logical_index"))
    if mode == "ternary":
        fields.extend(("frequency_hz", "amplitude", "pan"))
    _require_fields(rows[0], fields)
    expected_rows = invocation.steps + 1  # type: ignore[operator]
    if len(rows) != expected_rows:
        raise NexusEvidenceError(
            "trace row count does not match steps + 1"
        )
    expected_lane = (
        f"lane-{invocation.channel % 3}"
        if invocation.channel is not None
        else None
    )
    centre_seen = False
    for index, row in enumerate(rows):
        if _exact_int(row["step"], "step") != index:
            raise NexusEvidenceError(
                "trace steps must be contiguous and ordered"
            )
        progress = _decimal(row["progress"], "progress")
        expected_progress = Decimal(index) / Decimal(
            invocation.steps  # type: ignore[arg-type]
        )
        if abs(progress - expected_progress) > Decimal("1e-15"):
            raise NexusEvidenceError(
                "trace progress disagrees with ordered step index"
            )
        if progress < 0 or progress > 1:
            raise NexusEvidenceError(
                "trace progress must remain within [0, 1]"
            )
        radius = _decimal(row["radius"], "radius")
        _decimal(row["orientation"], "orientation")
        if radius < 0:
            raise NexusEvidenceError("trace radius must be non-negative")
        if progress == Decimal("0.5"):
            centre_seen = True
            if any(
                _decimal(row[field], field) != 0
                for field in ("x", "y", "radius", "orientation")
            ):
                raise NexusEvidenceError(
                    "the NEXUS centre row must be the exact origin"
                )
        if ternary:
            if progress < Decimal("0.5"):
                expected = ("inbound", -1)
            elif progress == Decimal("0.5"):
                expected = ("nexus", 0)
            else:
                expected = ("outbound", 1)
            if (
                row["trit"] != expected[0]
                or _exact_int(row["trit_value"], "trit_value")
                != expected[1]
            ):
                raise NexusEvidenceError(
                    "ternary transfer state disagrees with progress"
                )
            if row["lane"] != expected_lane:
                raise NexusEvidenceError(
                    "triality lane disagrees with rendered channel"
                )
        if mode == "fibonacci":
            logical_index = _exact_int(
                row["logical_index"],
                "logical_index",
            )
            if not 0 <= logical_index < invocation.config.logical:
                raise NexusEvidenceError(
                    "Fibonacci logical index is outside logical cardinality"
                )
            fraction = _decimal(row["fib_fraction"], "fib_fraction")
            expected_fraction = Decimal(logical_index) / Decimal(
                invocation.config.logical
            )
            if abs(fraction - expected_fraction) > Decimal("1e-15"):
                raise NexusEvidenceError(
                    "Fibonacci fraction disagrees with logical index"
                )
        if mode == "ternary":
            frequency = _decimal(row["frequency_hz"], "frequency_hz")
            amplitude = _decimal(row["amplitude"], "amplitude")
            pan = _decimal(row["pan"], "pan")
            if (
                frequency <= 0
                or not 0 <= amplitude <= 1
                or not -1 <= pan <= 1
            ):
                raise NexusEvidenceError(
                    "sonification controls are outside declared bounds"
                )
            if progress == Decimal("0.5") and (
                amplitude != 1 or pan != 0
            ):
                raise NexusEvidenceError(
                    "nexus sonification event must be full-amplitude and centred"
                )
    if (
        invocation.steps % 2 == 0  # type: ignore[operator]
        and not centre_seen
    ):
        raise NexusEvidenceError(
            "even-step trace must contain the exact centre"
        )
    return {
        "ordered": True,
        "row_count": len(rows),
        "centre_exact": centre_seen,
        "ternary_classification_verified": ternary,
        "triality_lane_verified": ternary,
        "sampling_mode": (
            "fibonacci-phi" if mode == "fibonacci" else "uniform-floor"
        ),
        "sonification_controls_verified": mode == "ternary",
    }


def _verify_receipts(rows: list[dict[str, str]]) -> dict[str, object]:
    _require_fields(rows[0], ("name", "bytes", "sha256"))
    expected = ["lane-0", "lane-1", "lane-2", "all-lanes", "chain"]
    names = [row["name"] for row in rows]
    if names != expected:
        raise NexusEvidenceError(
            "receipt rows must be lane-0, lane-1, lane-2, all-lanes, chain"
        )
    for row in rows:
        if _exact_int(row["bytes"], "bytes") < 0:
            raise NexusEvidenceError(
                "receipt byte counts must be non-negative"
            )
        if not _SHA256_RE.fullmatch(row["sha256"]):
            raise NexusEvidenceError(
                "receipt digest must be a full lowercase SHA-256"
            )
    return {
        "per_lane_receipts": True,
        "all_lane_receipt": True,
        "chain_receipt": True,
        "full_sha256_only": True,
        "row_count": len(rows),
    }


def build_execution_receipt(
    invocation: NexusInvocation,
    *,
    rows: list[dict[str, str]],
    stdout_bytes: bytes,
    binary_sha256: str,
    build_attestation_sha256: str,
) -> dict[str, object]:
    if not _SHA256_RE.fullmatch(binary_sha256):
        raise NexusEvidenceError(
            "binary_sha256 must be a full lowercase SHA-256"
        )
    if not _SHA256_RE.fullmatch(build_attestation_sha256):
        raise NexusEvidenceError(
            "build_attestation_sha256 must be a full lowercase SHA-256"
        )
    invariants = verify_evidence(invocation, rows)
    payload: dict[str, object] = {
        "schema": "qec.nexus-execution-receipt.v1",
        "qec_version": "170.2.0",
        "source": invocation.source.as_dict(),
        "invocation": invocation.as_dict(),
        "artifacts": {
            "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "binary_sha256": binary_sha256,
            "csv_rows": len(rows),
        },
        "build_attestation_sha256": build_attestation_sha256,
        "invariants": invariants,
        "claim_boundary": {
            "adapter_only": True,
            "decoder_mutation": False,
            "physical_claim": False,
            "qutrit_or_gf3_claim": False,
            "receipt_proves": (
                "byte_identity_and_declared_execution_invariants"
            ),
            "receipt_does_not_prove": (
                "physical_truth_or_quantum_advantage"
            ),
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def validate_execution_receipt(
    receipt: dict[str, object],
) -> dict[str, object]:
    if receipt.get("schema") != "qec.nexus-execution-receipt.v1":
        raise NexusEvidenceError(
            "unexpected NEXUS execution receipt schema"
        )
    observed = receipt.get("sha256")
    if not isinstance(observed, str) or not _SHA256_RE.fullmatch(observed):
        raise NexusEvidenceError(
            "receipt sha256 must be a full lowercase digest"
        )
    unsigned = dict(receipt)
    unsigned.pop("sha256", None)
    expected = canonical_sha256(unsigned)
    if observed != expected:
        raise NexusEvidenceError("NEXUS execution receipt hash mismatch")
    if receipt.get("qec_version") != "170.2.0":
        raise NexusEvidenceError("unexpected QEC version in NEXUS receipt")
    attestation_sha = receipt.get("build_attestation_sha256")
    if not isinstance(attestation_sha, str) or not _SHA256_RE.fullmatch(
        attestation_sha
    ):
        raise NexusEvidenceError(
            "receipt must bind a full build-attestation SHA-256"
        )
    invocation = receipt.get("invocation")
    source = receipt.get("source")
    if not isinstance(invocation, dict) or not isinstance(
        invocation.get("profile"),
        str,
    ):
        raise NexusEvidenceError("receipt invocation profile is missing")
    from .source import source_profile

    if source != source_profile(invocation["profile"]).as_dict():
        raise NexusEvidenceError(
            "receipt source identity does not match its profile"
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise NexusEvidenceError(
            "receipt artifact identity block is missing"
        )
    for name in ("stdout_sha256", "binary_sha256"):
        value = artifacts.get(name)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise NexusEvidenceError(
                f"receipt {name} must be a full SHA-256"
            )
    boundary = receipt.get("claim_boundary")
    if not isinstance(boundary, dict) or boundary.get(
        "physical_claim"
    ) is not False:
        raise NexusEvidenceError(
            "NEXUS receipt must preserve the no-physical-claim boundary"
        )
    return {"valid": True, "sha256": observed}
