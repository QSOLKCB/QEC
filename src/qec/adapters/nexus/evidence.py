"""NEXUS CSV parsing, invariant verification, and QEC parent receipts."""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import math
import re
from typing import Iterable

from qec.sonify.canonical import canonical_sha256
from .attestation import validate_build_attestation_record
from .contract import NexusInvocation

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_TOLERANCE = 1e-12
_AUDIO_TOLERANCE = 5e-8
_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_TAU = 2.0 * math.pi


class NexusEvidenceError(ValueError):
    pass


def _claim_boundary() -> dict[str, object]:
    return {
        "adapter_only": True,
        "decoder_mutation": False,
        "physical_claim": False,
        "qutrit_or_gf3_claim": False,
        "receipt_proves": "byte_identity_and_declared_execution_invariants",
        "receipt_does_not_prove": "physical_truth_or_quantum_advantage",
    }


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


def _rows_from_stdout(stdout_bytes: bytes) -> list[dict[str, str]]:
    try:
        text = stdout_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NexusEvidenceError("NEXUS stdout must be UTF-8 CSV") from exc
    return parse_csv(text)


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


def _close(observed: Decimal, expected: float, tolerance: float) -> bool:
    return math.isclose(
        float(observed),
        expected,
        rel_tol=tolerance,
        abs_tol=tolerance,
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
        "workers",
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
    requested_workers = 1
    if invocation.operation == "verify-parallel":
        requested_workers = invocation.workers  # type: ignore[assignment]
    reported_workers = _exact_int(metrics["workers"], "workers")
    if reported_workers != requested_workers:
        raise NexusEvidenceError(
            "reported requested worker count does not match invocation"
        )
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
        "requested_workers_bound": True,
        "reported_requested_workers": reported_workers,
        "effective_workers_claim": False,
        "row_count": len(rows),
    }


def _uniform_logical_index(invocation: NexusInvocation) -> int:
    channel = invocation.channel  # type: ignore[assignment]
    return channel * invocation.config.logical // invocation.config.rendered


def _fibonacci_logical_index(invocation: NexusInvocation) -> int:
    channel = invocation.channel  # type: ignore[assignment]
    phi_fraction = (channel / _PHI) % 1.0
    return min(
        int(math.floor(phi_fraction * invocation.config.logical)),
        invocation.config.logical - 1,
    )


def _channel_angle(invocation: NexusInvocation, logical_index: int) -> float:
    phase = float(invocation.config.phase)
    fraction = logical_index / invocation.config.logical
    return phase + _TAU * fraction


def _expected_geometry(
    invocation: NexusInvocation,
    progress: float,
    *,
    logical_index: int,
) -> tuple[float, float, float, float]:
    radius = float(invocation.config.radius)
    turns = float(invocation.config.turns)
    angle = _channel_angle(invocation, logical_index)
    if progress == 0.0:
        x = -radius * math.cos(angle)
        y = -radius * math.sin(angle)
    elif progress == 0.5:
        x = 0.0
        y = 0.0
    elif progress == 1.0:
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
    elif progress < 0.5:
        local = progress * 2.0
        local_radius = radius * (1.0 - local)
        local_angle = angle + math.pi + turns * _TAU * local
        x = local_radius * math.cos(local_angle)
        y = local_radius * math.sin(local_angle)
    else:
        local = (progress - 0.5) * 2.0
        local_radius = radius * local
        local_angle = angle - turns * _TAU * (1.0 - local)
        x = local_radius * math.cos(local_angle)
        y = local_radius * math.sin(local_angle)
    observed_radius = math.hypot(x, y)
    orientation = math.sin(math.pi * (progress - 0.5))
    return x, y, observed_radius, orientation


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
    logical_index = (
        _fibonacci_logical_index(invocation)
        if mode == "fibonacci"
        else _uniform_logical_index(invocation)
    )
    expected_fibonacci_fraction = Decimal(logical_index) / Decimal(
        invocation.config.logical
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
        x = _decimal(row["x"], "x")
        y = _decimal(row["y"], "y")
        radius = _decimal(row["radius"], "radius")
        orientation = _decimal(row["orientation"], "orientation")
        if radius < 0:
            raise NexusEvidenceError("trace radius must be non-negative")
        expected_geometry = _expected_geometry(
            invocation,
            float(expected_progress),
            logical_index=logical_index,
        )
        for observed, expected, field in zip(
            (x, y, radius, orientation),
            expected_geometry,
            ("x", "y", "radius", "orientation"),
        ):
            if not _close(observed, expected, _FLOAT_TOLERANCE):
                raise NexusEvidenceError(
                    f"trace {field} disagrees with declared geometry"
                )
        if progress == Decimal("0.5"):
            centre_seen = True
            if any(value != 0 for value in (x, y, radius, orientation)):
                raise NexusEvidenceError(
                    "the NEXUS centre row must be the exact origin"
                )
        if ternary:
            if progress < Decimal("0.5"):
                expected_state = ("inbound", -1)
            elif progress == Decimal("0.5"):
                expected_state = ("nexus", 0)
            else:
                expected_state = ("outbound", 1)
            if (
                row["trit"] != expected_state[0]
                or _exact_int(row["trit_value"], "trit_value")
                != expected_state[1]
            ):
                raise NexusEvidenceError(
                    "ternary transfer state disagrees with progress"
                )
            if row["lane"] != expected_lane:
                raise NexusEvidenceError(
                    "triality lane disagrees with rendered channel"
                )
        if mode == "fibonacci":
            reported_index = _exact_int(
                row["logical_index"],
                "logical_index",
            )
            if reported_index != logical_index:
                raise NexusEvidenceError(
                    "Fibonacci logical index disagrees with phi policy"
                )
            fraction = _decimal(row["fib_fraction"], "fib_fraction")
            if (
                abs(fraction - expected_fibonacci_fraction)
                > Decimal("1e-15")
            ):
                raise NexusEvidenceError(
                    "Fibonacci fraction disagrees with phi-selected index"
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
            lane_semitones = (0.0, 4.0, 7.0)[
                invocation.channel % 3  # type: ignore[operator]
            ]
            direction_semitones = {-1: -2.0, 0: 0.0, 1: 2.0}[
                expected_state[1]
            ]
            expected_frequency = float(
                invocation.base_frequency_hz  # type: ignore[arg-type]
            ) * 2.0 ** ((lane_semitones + direction_semitones) / 12.0)
            normalized_radius = min(
                max(float(radius) / float(invocation.config.radius), 0.0),
                1.0,
            )
            expected_amplitude = (
                1.0
                if expected_state[1] == 0
                else 0.2 + 0.8 * normalized_radius
            )
            expected_pan = min(
                max(float(x) / float(invocation.config.radius), -1.0),
                1.0,
            )
            for observed, expected, field in (
                (frequency, expected_frequency, "frequency_hz"),
                (amplitude, expected_amplitude, "amplitude"),
                (pan, expected_pan, "pan"),
            ):
                if not _close(observed, expected, _AUDIO_TOLERANCE):
                    raise NexusEvidenceError(
                        f"sonification {field} disagrees with mapping"
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
        "geometry_recomputed": True,
        "ternary_classification_verified": ternary,
        "triality_lane_verified": ternary,
        "sampling_mode": (
            "fibonacci-phi" if mode == "fibonacci" else "uniform-floor"
        ),
        "fibonacci_policy_recomputed": mode == "fibonacci",
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
    rows = _rows_from_stdout(stdout_bytes)
    invariants = verify_evidence(invocation, rows)
    source = invocation.source.as_dict()
    request = invocation.as_dict()
    payload: dict[str, object] = {
        "schema": "qec.nexus-execution-receipt.v1",
        "qec_version": "170.2.0",
        "source": source,
        "invocation": request,
        "artifacts": {
            "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "binary_sha256": binary_sha256,
            "request_sha256": canonical_sha256(request),
            "source_identity_sha256": canonical_sha256(source),
            "csv_rows": len(rows),
        },
        "build_attestation_sha256": build_attestation_sha256,
        "invariants": invariants,
        "claim_boundary": _claim_boundary(),
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
    try:
        invocation = NexusInvocation.from_dict(receipt.get("invocation"))
    except ValueError as exc:
        raise NexusEvidenceError("invalid invocation in NEXUS receipt") from exc
    source = receipt.get("source")
    if source != invocation.source.as_dict():
        raise NexusEvidenceError(
            "receipt source identity does not match its profile"
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise NexusEvidenceError(
            "receipt artifact identity block is missing"
        )
    for name in (
        "stdout_sha256",
        "binary_sha256",
        "request_sha256",
        "source_identity_sha256",
    ):
        value = artifacts.get(name)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise NexusEvidenceError(
                f"receipt {name} must be a full SHA-256"
            )
    if artifacts["request_sha256"] != canonical_sha256(invocation.as_dict()):
        raise NexusEvidenceError("receipt request identity mismatch")
    if artifacts["source_identity_sha256"] != canonical_sha256(
        invocation.source.as_dict()
    ):
        raise NexusEvidenceError("receipt source identity hash mismatch")
    csv_rows = artifacts.get("csv_rows")
    if type(csv_rows) is not int or csv_rows <= 0:
        raise NexusEvidenceError("receipt csv_rows must be a positive integer")
    if not isinstance(receipt.get("invariants"), dict):
        raise NexusEvidenceError("receipt invariant block is missing")
    if receipt.get("claim_boundary") != _claim_boundary():
        raise NexusEvidenceError(
            "NEXUS receipt claim boundary does not match the adapter contract"
        )
    return {"valid": True, "sha256": observed}


def validate_execution_bundle(
    receipt: dict[str, object],
    *,
    request: dict[str, object],
    source_identity: dict[str, object],
    build_attestation: dict[str, object],
    stdout_bytes: bytes,
) -> dict[str, object]:
    result = validate_execution_receipt(receipt)
    try:
        invocation = NexusInvocation.from_dict(request)
    except ValueError as exc:
        raise NexusEvidenceError("invalid stored NEXUS request") from exc
    if receipt.get("invocation") != invocation.as_dict():
        raise NexusEvidenceError(
            "stored request does not match execution receipt"
        )
    if source_identity != invocation.source.as_dict():
        raise NexusEvidenceError(
            "stored source identity does not match request profile"
        )
    if receipt.get("source") != source_identity:
        raise NexusEvidenceError(
            "stored source identity does not match execution receipt"
        )
    try:
        attestation_result = validate_build_attestation_record(
            build_attestation,
            profile=invocation.profile,
        )
    except ValueError as exc:
        raise NexusEvidenceError("invalid stored build attestation") from exc
    if receipt.get("build_attestation_sha256") != attestation_result["sha256"]:
        raise NexusEvidenceError(
            "stored build attestation does not match execution receipt"
        )
    artifacts = receipt["artifacts"]
    assert isinstance(artifacts, dict)
    if artifacts.get("binary_sha256") != attestation_result["binary_sha256"]:
        raise NexusEvidenceError(
            "receipt binary identity does not match build attestation"
        )
    rows = _rows_from_stdout(stdout_bytes)
    stdout_sha256 = hashlib.sha256(stdout_bytes).hexdigest()
    if artifacts.get("stdout_sha256") != stdout_sha256:
        raise NexusEvidenceError(
            "stored NEXUS CSV does not match execution receipt"
        )
    if artifacts.get("csv_rows") != len(rows):
        raise NexusEvidenceError(
            "stored NEXUS CSV row count does not match execution receipt"
        )
    if artifacts.get("request_sha256") != canonical_sha256(request):
        raise NexusEvidenceError(
            "stored request hash does not match execution receipt"
        )
    if artifacts.get("source_identity_sha256") != canonical_sha256(
        source_identity
    ):
        raise NexusEvidenceError(
            "stored source hash does not match execution receipt"
        )
    invariants = verify_evidence(invocation, rows)
    if receipt.get("invariants") != invariants:
        raise NexusEvidenceError(
            "stored NEXUS evidence does not reproduce receipt invariants"
        )
    return {
        "valid": True,
        "bundle_verified": True,
        "sha256": result["sha256"],
    }
