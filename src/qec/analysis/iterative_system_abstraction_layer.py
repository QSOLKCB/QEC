"""v142.0 — Iterative System Abstraction Layer.

Deterministic Layer-4 abstraction for iterative state snapshots, transitions,
and replay-safe execution receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex

ITERATIVE_SYSTEM_ABSTRACTION_LAYER_VERSION = "v142.0"
_CONTROL_MODE = "iterative_abstraction_advisory"
_VALID_TRANSITION_LABELS: tuple[str, ...] = ("advance", "stabilize", "converge", "stall")


class _FrozenDict(dict[str, Any]):
    """Minimal immutable dict for deterministic payload sealing."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("state_payload is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable



def _float_in_unit_interval(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{field_name} must be finite")
    if output < 0.0 or output > 1.0:
        raise ValueError(f"{field_name} must be in [0,1]")
    return output



def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


@dataclass(frozen=True)
class IterativeStateSnapshot:
    step_index: int
    state_id: str
    state_payload: dict[str, Any]
    convergence_metric: float
    active: bool

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("step_index must be int >= 0")
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("state_id must be a non-empty str")
        if not isinstance(self.state_payload, dict):
            raise ValueError("state_payload must be dict")
        try:
            canonical_payload = canonicalize_json(self.state_payload)
        except CanonicalHashingError as exc:
            raise ValueError("malformed payload") from exc
        if not isinstance(canonical_payload, dict):
            raise ValueError("malformed payload")
        object.__setattr__(self, "state_payload", _FrozenDict(canonical_payload))
        object.__setattr__(self, "convergence_metric", _float_in_unit_interval(self.convergence_metric, "convergence_metric"))
        if not isinstance(self.active, bool):
            raise ValueError("active must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "state_id": self.state_id,
            "state_payload": dict(self.state_payload),
            "convergence_metric": self.convergence_metric,
            "active": self.active,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class IterativeTransition:
    from_state_id: str
    to_state_id: str
    delta_magnitude: float
    transition_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.from_state_id, str) or not self.from_state_id:
            raise ValueError("from_state_id must be a non-empty str")
        if not isinstance(self.to_state_id, str) or not self.to_state_id:
            raise ValueError("to_state_id must be a non-empty str")
        object.__setattr__(self, "delta_magnitude", _float_in_unit_interval(self.delta_magnitude, "delta_magnitude"))
        if self.transition_label not in _VALID_TRANSITION_LABELS:
            raise ValueError("invalid transition label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state_id": self.from_state_id,
            "to_state_id": self.to_state_id,
            "delta_magnitude": self.delta_magnitude,
            "transition_label": self.transition_label,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class IterativeExecutionTrace:
    snapshots: tuple[IterativeStateSnapshot, ...]
    transitions: tuple[IterativeTransition, ...]
    total_steps: int
    final_state_id: str
    mean_convergence: float
    converged: bool

    def __post_init__(self) -> None:
        if not isinstance(self.snapshots, tuple):
            raise ValueError("snapshots must be tuple[IterativeStateSnapshot, ...]")
        for snapshot in self.snapshots:
            if not isinstance(snapshot, IterativeStateSnapshot):
                raise ValueError("snapshots must be tuple[IterativeStateSnapshot, ...]")

        if not isinstance(self.transitions, tuple):
            raise ValueError("transitions must be tuple[IterativeTransition, ...]")
        for transition in self.transitions:
            if not isinstance(transition, IterativeTransition):
                raise ValueError("transitions must be tuple[IterativeTransition, ...]")

        if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int) or self.total_steps < 0:
            raise ValueError("total_steps must be int >= 0")
        if self.total_steps != len(self.snapshots):
            raise ValueError("total_steps must equal len(snapshots)")

        if not isinstance(self.final_state_id, str):
            raise ValueError("final_state_id must be str")
        if self.snapshots and self.final_state_id != self.snapshots[-1].state_id:
            raise ValueError("final_state_id must equal snapshots[-1].state_id")
        if not self.snapshots and self.final_state_id != "":
            raise ValueError("final_state_id must be empty when snapshots is empty")

        if len(self.transitions) != max(self.total_steps - 1, 0):
            raise ValueError("transitions length must be max(total_steps - 1, 0)")

        object.__setattr__(self, "mean_convergence", _float_in_unit_interval(self.mean_convergence, "mean_convergence"))
        if not isinstance(self.converged, bool):
            raise ValueError("converged must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshots": tuple(snapshot.to_dict() for snapshot in self.snapshots),
            "transitions": tuple(transition.to_dict() for transition in self.transitions),
            "total_steps": self.total_steps,
            "final_state_id": self.final_state_id,
            "mean_convergence": self.mean_convergence,
            "converged": self.converged,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class IterativeExecutionReceipt:
    version: str
    trace: IterativeExecutionTrace
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.trace, IterativeExecutionTrace):
            raise ValueError("trace must be IterativeExecutionTrace")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")
        object.__setattr__(self, "stable_hash", self.stable_hash_value())

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "trace": self.trace.to_dict(),
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
        }

    def stable_hash_value(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def _transition_label(previous: IterativeStateSnapshot, current: IterativeStateSnapshot, delta_magnitude: float) -> str:
    if current.convergence_metric >= 0.999:
        return "converge"
    if delta_magnitude < 0.01:
        return "stabilize"
    if current.convergence_metric > previous.convergence_metric:
        return "advance"
    return "stall"


def evaluate_iterative_system_abstraction(
    snapshots: tuple[IterativeStateSnapshot, ...],
    *,
    version: str = ITERATIVE_SYSTEM_ABSTRACTION_LAYER_VERSION,
) -> IterativeExecutionReceipt:
    if not isinstance(snapshots, tuple):
        raise ValueError("snapshots must be tuple[IterativeStateSnapshot, ...]")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    for snapshot in snapshots:
        if not isinstance(snapshot, IterativeStateSnapshot):
            raise ValueError("invalid snapshot types")

    if snapshots:
        if snapshots[0].step_index != 0:
            raise ValueError("missing step 0")
        for expected_idx, snapshot in enumerate(snapshots):
            if snapshot.step_index != expected_idx:
                raise ValueError("invalid step ordering")

    transitions: list[IterativeTransition] = []
    for idx in range(len(snapshots) - 1):
        prev_snapshot = snapshots[idx]
        next_snapshot = snapshots[idx + 1]
        delta = _clamp01(abs(next_snapshot.convergence_metric - prev_snapshot.convergence_metric))
        transitions.append(
            IterativeTransition(
                from_state_id=prev_snapshot.state_id,
                to_state_id=next_snapshot.state_id,
                delta_magnitude=delta,
                transition_label=_transition_label(prev_snapshot, next_snapshot, delta),
            )
        )

    total_steps = len(snapshots)
    final_state_id = snapshots[-1].state_id if snapshots else ""
    mean_convergence = 0.0 if not snapshots else _clamp01(sum(s.convergence_metric for s in snapshots) / float(total_steps))
    converged = bool(snapshots) and snapshots[-1].convergence_metric >= 0.999

    trace = IterativeExecutionTrace(
        snapshots=snapshots,
        transitions=tuple(transitions),
        total_steps=total_steps,
        final_state_id=final_state_id,
        mean_convergence=mean_convergence,
        converged=converged,
    )

    return IterativeExecutionReceipt(
        version=version,
        trace=trace,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "ITERATIVE_SYSTEM_ABSTRACTION_LAYER_VERSION",
    "IterativeStateSnapshot",
    "IterativeTransition",
    "IterativeExecutionTrace",
    "IterativeExecutionReceipt",
    "evaluate_iterative_system_abstraction",
]
