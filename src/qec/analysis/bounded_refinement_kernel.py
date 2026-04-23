"""v144.4 — deterministic bounded refinement kernel (BRK)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import string

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.deterministic_transition_policy import TransitionPolicyReceipt

MAX_ITERATIONS = 8
STEP_SIZE = 0.25
CONVERGENCE_THRESHOLD = 0.01
REFINEMENT_DIMENSION = 4

CLASSIFICATION_CONVERGED = "converged"
CLASSIFICATION_BOUNDED = "bounded"
CLASSIFICATION_NO_IMPROVEMENT = "no_improvement"

_ALLOWED_CLASSIFICATIONS = frozenset(
    {CLASSIFICATION_CONVERGED, CLASSIFICATION_BOUNDED, CLASSIFICATION_NO_IMPROVEMENT}
)

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _round12(value: float) -> float:
    return round(float(value), 12)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _normalize_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    if normalized != value:
        raise ValueError(f"{field_name} must not include leading/trailing whitespace")
    return normalized


def _validate_sha256_hex(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be 64-char hex string")
    if any(ch not in string.hexdigits for ch in value) or value.lower() != value:
        raise ValueError(f"{field_name} must be 64-char hex string")
    return value


def _validate_vector(value: tuple[float, ...], field_name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_name} entries must be numeric in [0,1]")
        numeric = float(item)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{field_name} entries must be finite and in [0,1]")
        normalized.append(numeric)
    return tuple(normalized)


def _vector_payload(vector: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(_round12(item) for item in vector)


def _vector_from_signature(ordering_signature: str, salt: str) -> tuple[float, ...]:
    digest = hashlib.sha256(f"{salt}:{ordering_signature}".encode("utf-8")).digest()
    values: list[float] = []
    denominator = float((1 << 64) - 1)
    for index in range(REFINEMENT_DIMENSION):
        start = index * 8
        chunk = digest[start : start + 8]
        values.append(int.from_bytes(chunk, "big") / denominator)
    return tuple(values)


def _deterministic_gradient(
    vector: tuple[float, ...],
    target_vector: tuple[float, ...],
    bias_vector: tuple[float, ...],
) -> tuple[float, ...]:
    gradient: list[float] = []
    for value, target, bias in zip(vector, target_vector, bias_vector):
        direction = 1.0 if value >= 0.5 else -1.0
        gradient.append((value - target) + (direction * bias))
    return tuple(gradient)


@dataclass(frozen=True)
class RefinementState:
    ordering_signature: str
    initial_vector: tuple[float, ...]
    dimension: int
    stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ordering_signature",
            _normalize_string(self.ordering_signature, "ordering_signature"),
        )
        object.__setattr__(self, "initial_vector", _validate_vector(self.initial_vector, "initial_vector"))
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension < 1:
            raise ValueError("dimension must be int >= 1")
        if len(self.initial_vector) != self.dimension:
            raise ValueError("vector length must equal dimension")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical refinement state payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "ordering_signature": self.ordering_signature,
            "initial_vector": _vector_payload(self.initial_vector),
            "dimension": self.dimension,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


@dataclass(frozen=True)
class RefinementStep:
    iteration: int
    input_vector: tuple[float, ...]
    output_vector: tuple[float, ...]
    delta_norm: float
    stable_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be int >= 0")
        object.__setattr__(self, "input_vector", _validate_vector(self.input_vector, "input_vector"))
        object.__setattr__(self, "output_vector", _validate_vector(self.output_vector, "output_vector"))
        if len(self.input_vector) != len(self.output_vector):
            raise ValueError("step vectors must have equal dimension")
        if isinstance(self.delta_norm, bool) or not isinstance(self.delta_norm, (int, float)):
            raise ValueError("delta_norm must be numeric in [0,1]")
        delta = float(self.delta_norm)
        if not math.isfinite(delta) or not 0.0 <= delta <= 1.0:
            raise ValueError("delta_norm must be finite and within [0,1]")
        object.__setattr__(self, "delta_norm", delta)
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical refinement step payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "iteration": self.iteration,
            "input_vector": _vector_payload(self.input_vector),
            "output_vector": _vector_payload(self.output_vector),
            "delta_norm": _round12(self.delta_norm),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


@dataclass(frozen=True)
class RefinementReceipt:
    input_policy_hash: str
    steps: tuple[RefinementStep, ...]
    final_vector: tuple[float, ...]
    iteration_count: int
    converged: bool
    convergence_metric: float
    classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_policy_hash",
            _validate_sha256_hex(_normalize_string(self.input_policy_hash, "input_policy_hash"), "input_policy_hash"),
        )
        if not isinstance(self.steps, tuple) or any(not isinstance(item, RefinementStep) for item in self.steps):
            raise ValueError("steps must be tuple[RefinementStep, ...]")
        if not self.steps:
            raise ValueError("steps must not be empty")
        object.__setattr__(self, "final_vector", _validate_vector(self.final_vector, "final_vector"))
        if isinstance(self.iteration_count, bool) or not isinstance(self.iteration_count, int) or self.iteration_count < 1:
            raise ValueError("iteration_count must be int >= 1")
        if self.iteration_count != len(self.steps):
            raise ValueError("iteration_count must match number of steps")
        if not isinstance(self.converged, bool):
            raise ValueError("converged must be bool")
        if isinstance(self.convergence_metric, bool) or not isinstance(self.convergence_metric, (int, float)):
            raise ValueError("convergence_metric must be numeric in [0,1]")
        metric = float(self.convergence_metric)
        if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise ValueError("convergence_metric must be finite and within [0,1]")
        object.__setattr__(self, "convergence_metric", metric)
        if self.classification not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError("classification is invalid")
        if len(self.final_vector) != len(self.steps[-1].output_vector):
            raise ValueError("final_vector dimension mismatch")
        if self.final_vector != self.steps[-1].output_vector:
            raise ValueError("final_vector must equal last output_vector")
        expected_iterations = tuple(range(len(self.steps)))
        actual_iterations = tuple(step.iteration for step in self.steps)
        if actual_iterations != expected_iterations:
            raise ValueError("invalid iteration state")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical refinement payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "input_policy_hash": self.input_policy_hash,
            "steps": tuple(item.to_dict() for item in self.steps),
            "final_vector": _vector_payload(self.final_vector),
            "iteration_count": self.iteration_count,
            "converged": self.converged,
            "convergence_metric": _round12(self.convergence_metric),
            "classification": self.classification,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


def _derive_refinement_state(ordering_signature: str) -> RefinementState:
    initial_vector = _vector_from_signature(ordering_signature, "initial")
    payload = {
        "ordering_signature": ordering_signature,
        "initial_vector": _vector_payload(initial_vector),
        "dimension": REFINEMENT_DIMENSION,
    }
    return RefinementState(
        ordering_signature=ordering_signature,
        initial_vector=initial_vector,
        dimension=REFINEMENT_DIMENSION,
        stable_hash=sha256_hex(payload),
    )


def refine_transition_policy(receipt: TransitionPolicyReceipt) -> RefinementReceipt:
    if not isinstance(receipt, TransitionPolicyReceipt):
        raise ValueError("receipt must be a TransitionPolicyReceipt")
    if receipt.stable_hash != receipt.computed_stable_hash():
        raise ValueError("receipt stable_hash is invalid")
    if receipt.selected_decision.stable_hash != receipt.selected_decision.computed_stable_hash():
        raise ValueError("selected_decision stable_hash is invalid")
    _validate_sha256_hex(receipt.input_receipt_hash, "input_receipt_hash")

    ordering_signature = _normalize_string(
        receipt.selected_decision.selected_ordering_signature,
        "selected_ordering_signature",
    )

    state = _derive_refinement_state(ordering_signature)
    target_vector = _vector_from_signature(ordering_signature, "target")
    bias_basis = _vector_from_signature(ordering_signature, "bias")
    bias_vector = tuple(0.2 + (basis * 0.6) for basis in bias_basis)

    current_vector = state.initial_vector
    steps: list[RefinementStep] = []
    converged = False
    final_delta = 1.0

    for iteration in range(MAX_ITERATIONS):
        gradient = _deterministic_gradient(current_vector, target_vector, bias_vector)
        next_vector = tuple(_clamp01(value - (STEP_SIZE * grad)) for value, grad in zip(current_vector, gradient))
        raw_delta = sum(abs(new - old) for new, old in zip(next_vector, current_vector)) / state.dimension
        delta_norm = _round12(raw_delta)

        step_payload = {
            "iteration": iteration,
            "input_vector": _vector_payload(current_vector),
            "output_vector": _vector_payload(next_vector),
            "delta_norm": delta_norm,
        }
        step = RefinementStep(
            iteration=iteration,
            input_vector=current_vector,
            output_vector=next_vector,
            delta_norm=delta_norm,
            stable_hash=sha256_hex(step_payload),
        )
        steps.append(step)
        current_vector = next_vector
        final_delta = delta_norm

        if delta_norm <= CONVERGENCE_THRESHOLD:
            converged = True
            break

    convergence_metric = _round12(_clamp01(1.0 - final_delta))
    if converged:
        classification = CLASSIFICATION_CONVERGED
    elif final_delta < 0.1:
        classification = CLASSIFICATION_BOUNDED
    else:
        classification = CLASSIFICATION_NO_IMPROVEMENT

    receipt_payload = {
        "input_policy_hash": receipt.stable_hash,
        "steps": tuple(step.to_dict() for step in steps),
        "final_vector": _vector_payload(current_vector),
        "iteration_count": len(steps),
        "converged": converged,
        "convergence_metric": convergence_metric,
        "classification": classification,
    }
    return RefinementReceipt(
        input_policy_hash=receipt.stable_hash,
        steps=tuple(steps),
        final_vector=current_vector,
        iteration_count=len(steps),
        converged=converged,
        convergence_metric=convergence_metric,
        classification=classification,
        stable_hash=sha256_hex(receipt_payload),
    )


__all__ = [
    "CLASSIFICATION_BOUNDED",
    "CLASSIFICATION_CONVERGED",
    "CLASSIFICATION_NO_IMPROVEMENT",
    "CONVERGENCE_THRESHOLD",
    "MAX_ITERATIONS",
    "REFINEMENT_DIMENSION",
    "RefinementReceipt",
    "RefinementState",
    "RefinementStep",
    "STEP_SIZE",
    "refine_transition_policy",
]
