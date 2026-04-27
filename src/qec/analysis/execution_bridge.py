"""v149.3 — Deterministic execution bridge (simulated actuation + validation)."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import math
from typing import Final

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.hardware_alignment_layer import HardwareAlignmentReceipt

EXECUTION_BRIDGE_MODULE_VERSION: Final[str] = "v149.3"

_ALLOWED_ALIGNMENT_STATUSES: Final[tuple[str, ...]] = (
    "ALIGNED",
    "DEGRADED_ALIGNMENT",
    "REPLAY_UNSUPPORTED",
    "CAPABILITY_MISMATCH",
    "PRIORITY_EXCEEDED",
    "UNSTABLE_HARDWARE",
)
_ALLOWED_ACTUATION_STATUSES: Final[tuple[str, ...]] = (
    "NO_OP",
    "SIMULATED_SUCCESS",
    "SIMULATED_DEGRADED",
    "SIMULATED_BLOCKED",
)
_ALLOWED_VALIDATION_STATUSES: Final[tuple[str, ...]] = (
    "VALID",
    "INVALID",
    "INCONSISTENT",
    "UNDEFINED",
)
_ALLOWED_EFFECT_CLASSES: Final[tuple[str, ...]] = ("none", "aligned", "degraded", "unstable")

_BLOCKED_ALIGNMENT_STATUSES: Final[set[str]] = {
    "REPLAY_UNSUPPORTED",
    "CAPABILITY_MISMATCH",
    "PRIORITY_EXCEEDED",
}

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _round_public_metric(value: float) -> float:
    return float(round(float(value), 12))


def _require_canonical_token(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty canonical string")
    token = value.strip()
    if not token or token != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return token


def _require_sha256_hex(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid SHA-256 hex") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    return value


def _require_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    return number


def _require_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class SimulatedActuationRequest:
    signal_id: str
    hardware_id: str
    alignment_status: str
    selected_capability: str
    mapping_score: float
    input_hash: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.signal_id, name="signal_id")
        _require_canonical_token(self.hardware_id, name="hardware_id")
        if self.alignment_status not in _ALLOWED_ALIGNMENT_STATUSES:
            raise ValueError("alignment_status is not supported")
        _require_canonical_token(self.selected_capability, name="selected_capability")
        object.__setattr__(self, "mapping_score", _round_public_metric(_require_probability(self.mapping_score, name="mapping_score")))
        _require_sha256_hex(self.input_hash, name="input_hash")

        zero_score_statuses = {"REPLAY_UNSUPPORTED", "CAPABILITY_MISMATCH", "PRIORITY_EXCEEDED"}
        if self.alignment_status in zero_score_statuses and self.mapping_score != 0.0:
            raise ValueError("mapping_score must be 0.0 for blocked hard-failure statuses")

        none_capability_statuses = {"REPLAY_UNSUPPORTED", "CAPABILITY_MISMATCH"}
        if self.alignment_status in none_capability_statuses and self.selected_capability != "NONE":
            raise ValueError("selected_capability must be NONE for replay/capability blocked statuses")
        if self.alignment_status not in none_capability_statuses and self.selected_capability == "NONE":
            raise ValueError("selected_capability must be a capability token for this alignment_status")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "signal_id": self.signal_id,
            "hardware_id": self.hardware_id,
            "alignment_status": self.alignment_status,
            "selected_capability": self.selected_capability,
            "mapping_score": _round_public_metric(float(self.mapping_score)),
            "input_hash": self.input_hash,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class SimulatedActuationResult:
    signal_id: str
    hardware_id: str
    actuation_status: str
    effect_strength: float
    effect_class: str
    simulation_hash: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.signal_id, name="signal_id")
        _require_canonical_token(self.hardware_id, name="hardware_id")
        if self.actuation_status not in _ALLOWED_ACTUATION_STATUSES:
            raise ValueError("actuation_status is not supported")
        object.__setattr__(self, "effect_strength", _round_public_metric(_require_probability(self.effect_strength, name="effect_strength")))
        if self.effect_class not in _ALLOWED_EFFECT_CLASSES:
            raise ValueError("effect_class is not supported")
        if self.actuation_status in {"SIMULATED_BLOCKED", "NO_OP"}:
            if self.effect_strength != 0.0:
                raise ValueError("effect_strength must be 0.0 for blocked or no-op statuses")
            if self.effect_class != "none":
                raise ValueError("effect_class must be none for blocked or no-op statuses")
        elif self.actuation_status == "SIMULATED_SUCCESS":
            if self.effect_class != "aligned":
                raise ValueError("effect_class must be aligned for SIMULATED_SUCCESS")
        elif self.actuation_status == "SIMULATED_DEGRADED":
            if self.effect_class not in {"degraded", "unstable"}:
                raise ValueError("effect_class must be degraded or unstable for SIMULATED_DEGRADED")

        computed_simulation_hash = sha256_hex(self._simulation_payload())
        provided_simulation_hash = _require_sha256_hex(self.simulation_hash, name="simulation_hash")
        if provided_simulation_hash != computed_simulation_hash:
            raise ValueError("simulation_hash must be derived from canonical payload")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _simulation_payload(self) -> dict[str, _JSONValue]:
        return {
            "signal_id": self.signal_id,
            "hardware_id": self.hardware_id,
            "actuation_status": self.actuation_status,
            "effect_strength": _round_public_metric(float(self.effect_strength)),
            "effect_class": self.effect_class,
        }

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {**self._simulation_payload(), "simulation_hash": self.simulation_hash}

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class ExecutionValidationResult:
    signal_id: str
    hardware_id: str
    validation_status: str
    consistency_score: float
    validation_reason: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.signal_id, name="signal_id")
        _require_canonical_token(self.hardware_id, name="hardware_id")
        if self.validation_status not in _ALLOWED_VALIDATION_STATUSES:
            raise ValueError("validation_status is not supported")
        object.__setattr__(self, "consistency_score", _round_public_metric(_require_probability(self.consistency_score, name="consistency_score")))
        _require_canonical_token(self.validation_reason, name="validation_reason")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "signal_id": self.signal_id,
            "hardware_id": self.hardware_id,
            "validation_status": self.validation_status,
            "consistency_score": _round_public_metric(float(self.consistency_score)),
            "validation_reason": self.validation_reason,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class ExecutionBridgeReceipt:
    module_version: str
    request_count: int
    actuation_results: tuple[SimulatedActuationResult, ...]
    validation_results: tuple[ExecutionValidationResult, ...]
    overall_consistency_score: float
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        if self.module_version != EXECUTION_BRIDGE_MODULE_VERSION:
            raise ValueError("module_version must match EXECUTION_BRIDGE_MODULE_VERSION")
        _require_non_negative_int(self.request_count, name="request_count")

        if not isinstance(self.actuation_results, tuple):
            raise ValueError("actuation_results must be tuple")
        if not isinstance(self.validation_results, tuple):
            raise ValueError("validation_results must be tuple")

        for item in self.actuation_results:
            if not isinstance(item, SimulatedActuationResult):
                raise ValueError("actuation_results must contain SimulatedActuationResult")
        for item in self.validation_results:
            if not isinstance(item, ExecutionValidationResult):
                raise ValueError("validation_results must contain ExecutionValidationResult")

        sorted_actuation = tuple(sorted(self.actuation_results, key=lambda v: (v.signal_id, v.hardware_id)))
        if sorted_actuation != self.actuation_results:
            raise ValueError("actuation_results must be sorted by (signal_id, hardware_id)")

        sorted_validation = tuple(sorted(self.validation_results, key=lambda v: (v.signal_id, v.hardware_id)))
        if sorted_validation != self.validation_results:
            raise ValueError("validation_results must be sorted by (signal_id, hardware_id)")

        if len(sorted_actuation) != self.request_count:
            raise ValueError("request_count must match actuation_results length")
        if len(sorted_validation) != self.request_count:
            raise ValueError("request_count must match validation_results length")

        actuation_pairs = tuple((item.signal_id, item.hardware_id) for item in sorted_actuation)
        validation_pairs = tuple((item.signal_id, item.hardware_id) for item in sorted_validation)
        if len(set(actuation_pairs)) != len(actuation_pairs):
            raise ValueError("duplicate (signal_id, hardware_id) in actuation_results is not allowed")
        if len(set(validation_pairs)) != len(validation_pairs):
            raise ValueError("duplicate (signal_id, hardware_id) in validation_results is not allowed")
        if actuation_pairs != validation_pairs:
            raise ValueError("actuation_results and validation_results must have identical pair ordering")

        object.__setattr__(
            self,
            "overall_consistency_score",
            _round_public_metric(_require_probability(self.overall_consistency_score, name="overall_consistency_score")),
        )
        computed_score = (
            _round_public_metric(sum(v.consistency_score for v in sorted_validation) / len(sorted_validation))
            if sorted_validation
            else 0.0
        )
        if float(self.overall_consistency_score) != computed_score:
            raise ValueError("overall_consistency_score must equal mean validation consistency_score")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "module_version": self.module_version,
            "request_count": int(self.request_count),
            "actuation_results": tuple(item.to_dict() for item in self.actuation_results),
            "validation_results": tuple(item.to_dict() for item in self.validation_results),
            "overall_consistency_score": _round_public_metric(float(self.overall_consistency_score)),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def _actuation_for_request(request: SimulatedActuationRequest) -> SimulatedActuationResult:
    if request.alignment_status in _BLOCKED_ALIGNMENT_STATUSES:
        actuation_status = "SIMULATED_BLOCKED"
        effect_strength = 0.0
        effect_class = "none"
    elif request.alignment_status == "UNSTABLE_HARDWARE":
        actuation_status = "SIMULATED_DEGRADED"
        effect_strength = request.mapping_score
        effect_class = "unstable"
    elif request.alignment_status == "DEGRADED_ALIGNMENT":
        actuation_status = "SIMULATED_DEGRADED"
        effect_strength = request.mapping_score
        effect_class = "degraded"
    elif request.alignment_status == "ALIGNED":
        actuation_status = "SIMULATED_SUCCESS"
        effect_strength = request.mapping_score
        effect_class = "aligned"
    else:
        actuation_status = "NO_OP"
        effect_strength = 0.0
        effect_class = "none"

    simulation_hash = sha256_hex(
        {
            "signal_id": request.signal_id,
            "hardware_id": request.hardware_id,
            "actuation_status": actuation_status,
            "effect_strength": _round_public_metric(effect_strength),
            "effect_class": effect_class,
        }
    )
    return SimulatedActuationResult(
        signal_id=request.signal_id,
        hardware_id=request.hardware_id,
        actuation_status=actuation_status,
        effect_strength=effect_strength,
        effect_class=effect_class,
        simulation_hash=simulation_hash,
    )


def _validate_execution(
    request: SimulatedActuationRequest,
    actuation: SimulatedActuationResult,
) -> ExecutionValidationResult:
    expected_by_alignment = {
        "REPLAY_UNSUPPORTED": "SIMULATED_BLOCKED",
        "CAPABILITY_MISMATCH": "SIMULATED_BLOCKED",
        "PRIORITY_EXCEEDED": "SIMULATED_BLOCKED",
        "ALIGNED": "SIMULATED_SUCCESS",
        "DEGRADED_ALIGNMENT": "SIMULATED_DEGRADED",
        "UNSTABLE_HARDWARE": "SIMULATED_DEGRADED",
    }

    if request.alignment_status not in expected_by_alignment:
        return ExecutionValidationResult(
            signal_id=request.signal_id,
            hardware_id=request.hardware_id,
            validation_status="INVALID",
            consistency_score=0.0,
            validation_reason="invalid_alignment_status",
        )

    expected_status = expected_by_alignment[request.alignment_status]
    if actuation.actuation_status == expected_status:
        return ExecutionValidationResult(
            signal_id=request.signal_id,
            hardware_id=request.hardware_id,
            validation_status="VALID",
            consistency_score=actuation.effect_strength,
            validation_reason="status_mapping_valid",
        )

    return ExecutionValidationResult(
        signal_id=request.signal_id,
        hardware_id=request.hardware_id,
        validation_status="INCONSISTENT",
        consistency_score=0.0,
        validation_reason="status_mapping_mismatch",
    )


def simulate_execution_bridge(
    alignment_receipt: HardwareAlignmentReceipt,
) -> ExecutionBridgeReceipt:
    if not isinstance(alignment_receipt, HardwareAlignmentReceipt):
        raise ValueError("alignment_receipt must be HardwareAlignmentReceipt")

    sorted_decisions = tuple(sorted(alignment_receipt.alignment_decisions, key=lambda d: (d.signal_id, d.hardware_id)))

    actuation_results: list[SimulatedActuationResult] = []
    validation_results: list[ExecutionValidationResult] = []

    for decision in sorted_decisions:
        request = SimulatedActuationRequest(
            signal_id=decision.signal_id,
            hardware_id=decision.hardware_id,
            alignment_status=decision.alignment_status,
            selected_capability=decision.selected_capability,
            mapping_score=decision.mapping_score,
            input_hash=decision.stable_hash(),
        )
        actuation = _actuation_for_request(request)
        validation = _validate_execution(request, actuation)
        actuation_results.append(actuation)
        validation_results.append(validation)

    actuation_tuple = tuple(actuation_results)
    validation_tuple = tuple(validation_results)
    overall_score = (
        _round_public_metric(sum(item.consistency_score for item in validation_tuple) / len(validation_tuple))
        if validation_tuple
        else 0.0
    )

    return ExecutionBridgeReceipt(
        module_version=EXECUTION_BRIDGE_MODULE_VERSION,
        request_count=len(sorted_decisions),
        actuation_results=actuation_tuple,
        validation_results=validation_tuple,
        overall_consistency_score=overall_score,
    )


__all__ = [
    "EXECUTION_BRIDGE_MODULE_VERSION",
    "SimulatedActuationRequest",
    "SimulatedActuationResult",
    "ExecutionValidationResult",
    "ExecutionBridgeReceipt",
    "simulate_execution_bridge",
]
