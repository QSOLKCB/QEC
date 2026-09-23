# SPDX-License-Identifier: MPL-2.0
"""v172.1 exact-coordinate common control over a v172.0 matrix snapshot.

A marker seals the complete request, evaluates only its declared coordinate,
then returns a plan or a rejection. It cannot actuate or reserve links. Every
receipt embeds the inputs needed to replay that computation without trusting
its claimed outcome. No ambient state, decoder calls, or timing participate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType

from qec.sonify.canonical import canonical_json, canonical_sha256, validate_sha256

from .core import CrossbarMatrix

MARKER_CONTRACT_VERSION = "172.1"
REGISTER_SCHEMA = "qec.crossbar-marker-input-register.v1"
PROGRAM_SCHEMA = "qec.crossbar-marker-program.v1"
RECEIPT_SCHEMA = "qec.crossbar-common-control-receipt.v1"
MARKER_VALIDATION_SCHEMA = "qec.crossbar-common-control-validation.v1"
POLICY = "exact-coordinate-idle-only.v1"
MAX_PAYLOAD_BYTES = 1_048_576
MARKER_CLAIM_BOUNDARY = MappingProxyType({
    "classical_software_model_only": True,
    "marker_authority": "compute_one_declared_coordinate_plan",
    "payload_mutation_permitted": False,
    "decoder_output_mutation_permitted": False,
    "decoder_output_identity_is_caller_declared": True,
    "reservation_present": False,
    "connection_commit_present": False,
    "multi_stage_search_present": False,
    "continuity_proof_present": False,
    "receipt_proves": "replay_of_single_matrix_common_control",
    "receipt_does_not_prove": "physical_connection_or_decoder_correctness",
})


def _text(value: object) -> None:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError("INVALID_INPUT")
    # Reject unpaired surrogates before canonical UTF-8 hashing.
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("INVALID_INPUT") from exc


def _seal(schema: str, **fields: object) -> dict[str, object]:
    unsigned = {"schema": schema, "contract_version": MARKER_CONTRACT_VERSION, **fields}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _exact(actual: object, expected: dict[str, object]) -> None:
    """Compare canonical bytes, so e.g. True cannot impersonate integer 1."""
    try:
        if not isinstance(actual, dict) or canonical_json(actual) != canonical_json(expected):
            raise ValueError("INVALID_INPUT")
    except (TypeError, UnicodeError) as exc:
        raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class CrossbarRequest:
    """Immutable intent and opaque payload; decoder identity is a declared reference."""

    request_id: str
    horizontal_link_id: str
    vertical_link_id: str
    payload: bytes
    decoder_output_sha256: str

    def __post_init__(self) -> None:
        for text in (self.request_id, self.horizontal_link_id, self.vertical_link_id):
            _text(text)
        if type(self.payload) is not bytes or len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("INVALID_INPUT")
        try:
            validate_sha256(self.decoder_output_sha256, "decoder_output_sha256")
        except (ValueError, TypeError) as exc:
            raise ValueError("INVALID_INPUT") from exc

    def as_dict(self) -> dict[str, object]:
        return _seal(
            REGISTER_SCHEMA,
            request_id=self.request_id,
            horizontal_link_id=self.horizontal_link_id,
            vertical_link_id=self.vertical_link_id,
            payload_hex=self.payload.hex(),
            payload_sha256=hashlib.sha256(self.payload).hexdigest(),
            payload_length=len(self.payload),
            decoder_output_sha256=self.decoder_output_sha256,
            sealed=True,
        )

    @classmethod
    def from_dict(cls, value: object) -> CrossbarRequest:
        if not isinstance(value, dict):
            raise ValueError("INVALID_INPUT")
        try:
            payload_hex = value["payload_hex"]
            if type(payload_hex) is not str or len(payload_hex) > MAX_PAYLOAD_BYTES * 2:
                raise ValueError("INVALID_INPUT")
            request = cls(
                value["request_id"], value["horizontal_link_id"], value["vertical_link_id"],
                bytes.fromhex(payload_hex), value["decoder_output_sha256"],
            )
            _exact(value, request.as_dict())
            return request
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def _matrix(manifest: object) -> CrossbarMatrix:
    try:
        return CrossbarMatrix.from_dict(manifest)
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _request(request: CrossbarRequest) -> dict[str, object]:
    if type(request) is not CrossbarRequest:
        raise ValueError("INVALID_INPUT")
    request.__post_init__()
    return request.as_dict()


def _program(matrix_hash: str, register: dict[str, object], marker_id: str) -> dict[str, object]:
    _text(marker_id)
    return _seal(
        PROGRAM_SCHEMA,
        marker_id=marker_id,
        crossbar_matrix_receipt_hash=matrix_hash,
        input_register_sha256=register["sha256"],
        policy=POLICY,
        max_coordinate_evaluations=1,
        fallback_permitted=False,
        claim_boundary=dict(MARKER_CLAIM_BOUNDARY),
    )


def compile_marker_program(
    matrix_manifest: object, request: CrossbarRequest, *, marker_id: str = "marker-0"
) -> dict[str, object]:
    """Seal a programme bound to exactly one complete request and matrix state."""
    matrix = _matrix(matrix_manifest)
    return _program(matrix.sha256(), _request(request), marker_id)


def execute_marker_program(
    matrix_manifest: object, request: CrossbarRequest, program: object
) -> dict[str, object]:
    """Validate sealed control intent, compute one plan, and release the marker.

    A selected plan is not a committed connection. Valid requests with unknown
    endpoints or non-idle links yield reproducible rejection receipts. Malformed
    inputs and changed programme authority fail with INVALID_INPUT.
    """
    matrix = _matrix(matrix_manifest)
    register = _request(request)
    if not isinstance(program, dict):
        raise ValueError("INVALID_INPUT")
    matrix_hash = matrix.sha256()
    expected_program = _program(matrix_hash, register, program.get("marker_id"))
    _exact(program, expected_program)

    # The root binds all inputs before any control decision. Event order is
    # logical and fixed; hashes link observations to this exact invocation.
    root = canonical_sha256({
        "schema": "qec.crossbar-marker-event-root.v1",
        "contract_version": MARKER_CONTRACT_VERSION,
        "matrix_sha256": matrix_hash,
        "register_sha256": register["sha256"],
        "program_sha256": expected_program["sha256"],
    })
    events: list[dict[str, object]] = []

    def event(kind: str, **details: object) -> None:
        unsigned = {
            "sequence": len(events), "kind": kind, "details": details,
            "previous_sha256": events[-1]["sha256"] if events else root,
        }
        events.append({**unsigned, "sha256": canonical_sha256(unsigned)})

    event("register_sealed", input_register_sha256=register["sha256"])
    event("program_verified", program_sha256=expected_program["sha256"])
    horizontal = next((link for link in matrix.horizontal_links
                       if link.link_id == request.horizontal_link_id), None)
    vertical = next((link for link in matrix.vertical_links
                     if link.link_id == request.vertical_link_id), None)
    event(
        "endpoints_checked",
        horizontal_link_id=request.horizontal_link_id,
        vertical_link_id=request.vertical_link_id,
        horizontal_state=None if horizontal is None else horizontal.state,
        vertical_state=None if vertical is None else vertical.state,
    )
    selected = None
    evaluations = 0
    # Explicit precedence is part of the contract; no traversal-order ties.
    if horizontal is None:
        reason = "unknown_horizontal_link"
    elif vertical is None:
        reason = "unknown_vertical_link"
    else:
        evaluations = 1
        coordinate = matrix.coordinate(horizontal.link_id, vertical.link_id).as_dict()
        event("coordinate_evaluated", coordinate=coordinate)
        if horizontal.state != "idle":
            reason = "horizontal_" + horizontal.state
        elif vertical.state != "idle":
            reason = "vertical_" + vertical.state
        else:
            reason = "exact_coordinate_idle"
            selected = coordinate
    outcome = "plan_selected" if selected is not None else "rejected"
    closure_plan = [] if selected is None else [selected]
    event(outcome, reason=reason, closure_plan=closure_plan)
    event("marker_released", marker_id=expected_program["marker_id"])
    return _seal(
        RECEIPT_SCHEMA,
        matrix_manifest=matrix.as_dict(),
        input_register=register,
        marker_program=expected_program,
        crossbar_matrix_receipt_hash=matrix_hash,
        payload_sha256=register["payload_sha256"],
        decoder_output_sha256=request.decoder_output_sha256,
        event_root_sha256=root,
        events=events,
        outcome=outcome,
        reason=reason,
        coordinate_evaluations=evaluations,
        closure_plan=closure_plan,
        matrix_after_sha256=matrix_hash,
        marker_released=True,
        claim_boundary=dict(MARKER_CLAIM_BOUNDARY),
    )


def validate_common_control_receipt(
    receipt: object,
    *,
    expected_matrix_sha256: str | None = None,
    expected_request_sha256: str | None = None,
    expected_program_sha256: str | None = None,
) -> dict[str, object]:
    """Recompute the full receipt; optional trusted hashes pin external inputs.

    Self-contained replay establishes consistency, not independent provenance.
    Callers comparing a receipt to a known run should supply the trusted hashes.
    """
    if not isinstance(receipt, dict):
        raise ValueError("INVALID_INPUT")
    try:
        request = CrossbarRequest.from_dict(receipt["input_register"])
        replay = execute_marker_program(receipt["matrix_manifest"], request, receipt["marker_program"])
        _exact(receipt, replay)
        bindings = {
            "matrix": (expected_matrix_sha256, replay["crossbar_matrix_receipt_hash"]),
            "request": (expected_request_sha256, replay["input_register"]["sha256"]),
            "program": (expected_program_sha256, replay["marker_program"]["sha256"]),
        }
        for expected, actual in bindings.values():
            if expected is not None and validate_sha256(expected) != actual:
                raise ValueError("INVALID_INPUT")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
    return _seal(
        MARKER_VALIDATION_SCHEMA,
        crossbar_common_control_receipt_hash=replay["sha256"],
        crossbar_matrix_receipt_hash=replay["crossbar_matrix_receipt_hash"],
        input_register_sha256=replay["input_register"]["sha256"],
        marker_program_sha256=replay["marker_program"]["sha256"],
        outcome=replay["outcome"],
        replay_verified=True,
        payload_identity_verified=True,
        declared_decoder_identity_verified=True,
        bounded_authority_verified=True,
        marker_release_verified=True,
        external_bindings_verified={key: expected is not None for key, (expected, _) in bindings.items()},
        all_passed=True,
    )
