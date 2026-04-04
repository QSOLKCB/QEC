"""Deterministic Layer-4 quantum noise balancing layer (v137.1.8).

This module provides a bounded perturbation balancing engine inspired by
quantum-noise choreography concepts while remaining fully deterministic and
replay-safe. It does not perform physical quantum simulation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple

QUANTUM_NOISE_BALANCING_LAYER_VERSION: str = "v137.1.8"
ROUND_DIGITS: int = 12
GENESIS_HASH: str = "0" * 64

# Theory invariants (explicitly preserved by this module)
BOUNDED_PERTURBATION_LAW: str = "BOUNDED_PERTURBATION_LAW"
DETERMINISTIC_NOISE_COMPENSATION: str = "DETERMINISTIC_NOISE_COMPENSATION"
FIDELITY_STABILITY_INVARIANT: str = "FIDELITY_STABILITY_INVARIANT"
REPLAY_SAFE_PERTURBATION_CHAIN: str = "REPLAY_SAFE_PERTURBATION_CHAIN"


@dataclass(frozen=True)
class NoisePerturbationSnapshot:
    perturbation_id: str
    noise_sources: Tuple[str, ...]
    perturbation_magnitude: float
    correction_pressure: float
    drift_signal: float
    compensation_factor: float
    fidelity_score: float
    replay_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation_id": self.perturbation_id,
            "noise_sources": list(self.noise_sources),
            "perturbation_magnitude": _round_float(self.perturbation_magnitude),
            "correction_pressure": _round_float(self.correction_pressure),
            "drift_signal": _round_float(self.drift_signal),
            "compensation_factor": _round_float(self.compensation_factor),
            "fidelity_score": _round_float(self.fidelity_score),
            "replay_hash": self.replay_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class NoiseCompensationModel:
    compensation_weights: Tuple[Tuple[str, float], ...]
    normalization_factor: float
    bounded: bool = True
    model_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "compensation_weights": [
                [source, _round_float(weight)]
                for source, weight in self.compensation_weights
            ],
            "normalization_factor": _round_float(self.normalization_factor),
            "bounded": self.bounded,
            "model_hash": self.model_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class FidelityStabilityReport:
    fidelity_score: float
    stability_score: float
    compensation_effectiveness: float
    balanced: bool
    decision_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity_score": _round_float(self.fidelity_score),
            "stability_score": _round_float(self.stability_score),
            "compensation_effectiveness": _round_float(
                self.compensation_effectiveness,
            ),
            "balanced": self.balanced,
            "decision_hash": self.decision_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PerturbationLedgerEntry:
    sequence_id: int
    perturbation_hash: str
    parent_hash: str
    fidelity_score: float
    stability_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "perturbation_hash": self.perturbation_hash,
            "parent_hash": self.parent_hash,
            "fidelity_score": _round_float(self.fidelity_score),
            "stability_score": _round_float(self.stability_score),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PerturbationLedger:
    entries: Tuple[PerturbationLedgerEntry, ...]
    head_hash: str
    chain_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "head_hash": self.head_hash,
            "chain_valid": self.chain_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _round_float(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def _clamp01(value: float) -> float:
    return _round_float(max(0.0, min(1.0, float(value))))


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def normalize_noise_inputs(
    noise_state: Mapping[str, float] | Iterable[tuple[str, float]],
) -> Tuple[Tuple[str, float], ...]:
    """Canonicalize and validate noise inputs with deterministic ordering."""
    if isinstance(noise_state, Mapping):
        items = tuple(noise_state.items())
    else:
        items = tuple(noise_state)

    normalized: list[tuple[str, float]] = []
    for source, value in items:
        if not isinstance(source, str) or source == "":
            raise ValueError("noise source names must be non-empty strings")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"noise value must be finite for source={source!r}")
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"noise value must be in [0, 1] for source={source!r}")
        normalized.append((source, _round_float(numeric)))

    normalized.sort(key=lambda pair: pair[0])
    for idx in range(1, len(normalized)):
        if normalized[idx - 1][0] == normalized[idx][0]:
            raise ValueError("duplicate noise source names are not allowed")
    return tuple(normalized)


def compute_bounded_perturbation_metrics(
    normalized_noise_inputs: Tuple[Tuple[str, float], ...],
) -> tuple[float, float, float]:
    """Compute bounded perturbation magnitude, correction pressure, and drift."""
    if not normalized_noise_inputs:
        return (0.0, 0.0, 0.0)

    values = tuple(value for _, value in normalized_noise_inputs)
    count = float(len(values))
    mean_value = sum(values) / count
    spread = max(values) - min(values)

    drift_components = [
        abs(values[i] - values[i - 1]) for i in range(1, len(values))
    ]
    drift_signal = max(drift_components) if drift_components else 0.0

    perturbation_magnitude = _clamp01(mean_value)
    correction_pressure = _clamp01(spread)
    drift_signal = _clamp01(drift_signal)
    return (perturbation_magnitude, correction_pressure, drift_signal)


def build_noise_compensation_model(
    normalized_noise_inputs: Tuple[Tuple[str, float], ...],
) -> NoiseCompensationModel:
    """Build deterministic compensation weights from explicit normalized inputs."""
    n = len(normalized_noise_inputs)
    if n == 0:
        payload = {
            "compensation_weights": [],
            "normalization_factor": 1.0,
            "bounded": True,
        }
        return NoiseCompensationModel(
            compensation_weights=(),
            normalization_factor=1.0,
            bounded=True,
            model_hash=_hash_sha256(payload),
        )

    raw_weights = tuple(
        (_round_float((idx + 1) * value), source)
        for idx, (source, value) in enumerate(normalized_noise_inputs)
    )
    normalization_factor = sum(weight for weight, _ in raw_weights)
    if normalization_factor <= 0.0:
        normalization_factor = float(n)
        normalized_weights = tuple((source, _round_float(1.0 / n)) for _, source in raw_weights)
    else:
        normalized_weights = tuple(
            (source, _round_float(weight / normalization_factor))
            for weight, source in raw_weights
        )

    payload = {
        "compensation_weights": [[s, w] for s, w in normalized_weights],
        "normalization_factor": _round_float(normalization_factor),
        "bounded": True,
    }
    return NoiseCompensationModel(
        compensation_weights=normalized_weights,
        normalization_factor=_round_float(normalization_factor),
        bounded=True,
        model_hash=_hash_sha256(payload),
    )


def compute_noise_compensation(
    normalized_noise_inputs: Tuple[Tuple[str, float], ...],
    compensation_model: NoiseCompensationModel,
) -> float:
    """Compute deterministic bounded compensation factor in [0, 1]."""
    values_by_source = {source: value for source, value in normalized_noise_inputs}
    weighted_sum = 0.0
    for source, weight in compensation_model.compensation_weights:
        weighted_sum += values_by_source[source] * float(weight)
    return _clamp01(weighted_sum)


def compute_fidelity_stability_score(
    perturbation_magnitude: float,
    correction_pressure: float,
    drift_signal: float,
    compensation_factor: float,
) -> tuple[float, float, float]:
    """Compute bounded fidelity, stability, and compensation effectiveness."""
    perturbation_magnitude = _clamp01(perturbation_magnitude)
    correction_pressure = _clamp01(correction_pressure)
    drift_signal = _clamp01(drift_signal)
    compensation_factor = _clamp01(compensation_factor)

    fidelity_score = _clamp01(1.0 - abs(perturbation_magnitude - compensation_factor))
    base_stability = _clamp01(1.0 - abs(drift_signal - correction_pressure))
    compensation_effectiveness = _clamp01(1.0 - abs(correction_pressure - compensation_factor))
    stability_score = _clamp01(base_stability + 0.25 * compensation_effectiveness)
    return (fidelity_score, stability_score, compensation_effectiveness)


def _entry_digest(entry: PerturbationLedgerEntry) -> str:
    return _hash_sha256(entry.to_dict())


def append_perturbation_ledger_entry(
    ledger: PerturbationLedger | None,
    perturbation_hash: str,
    fidelity_score: float,
    stability_score: float,
) -> PerturbationLedger:
    """Append one replay-safe perturbation ledger entry with SHA-256 chaining."""
    current = ledger if ledger is not None else PerturbationLedger((), GENESIS_HASH, True)
    if not current.chain_valid:
        raise ValueError("cannot append to an invalid perturbation ledger")
    if not validate_perturbation_ledger(current):
        raise ValueError("cannot append to a malformed perturbation ledger")

    parent_hash = current.head_hash if current.entries else GENESIS_HASH
    entry = PerturbationLedgerEntry(
        sequence_id=len(current.entries),
        perturbation_hash=perturbation_hash,
        parent_hash=parent_hash,
        fidelity_score=_clamp01(fidelity_score),
        stability_score=_clamp01(stability_score),
    )
    entries = current.entries + (entry,)
    new_head_hash = _entry_digest(entry)
    updated = PerturbationLedger(entries=entries, head_hash=new_head_hash, chain_valid=True)
    return PerturbationLedger(
        entries=updated.entries,
        head_hash=updated.head_hash,
        chain_valid=validate_perturbation_ledger(updated),
    )


def validate_perturbation_ledger(ledger: PerturbationLedger) -> bool:
    """Deterministically recompute the full chain and validate parent links."""
    expected_parent = GENESIS_HASH
    expected_head = GENESIS_HASH
    for expected_sequence, entry in enumerate(ledger.entries):
        if entry.sequence_id != expected_sequence:
            return False
        if entry.parent_hash != expected_parent:
            return False
        digest = _entry_digest(entry)
        expected_parent = digest
        expected_head = digest

    if ledger.entries:
        return ledger.head_hash == expected_head
    return ledger.head_hash == GENESIS_HASH


def run_quantum_noise_balancing_layer(
    noise_state: Mapping[str, float] | Iterable[tuple[str, float]],
    prior_ledger: PerturbationLedger | None = None,
    perturbation_id: str | None = None,
) -> tuple[FidelityStabilityReport, PerturbationLedger]:
    """Run deterministic Layer-4 bounded perturbation balancing orchestration."""
    normalized = normalize_noise_inputs(noise_state)
    perturbation_magnitude, correction_pressure, drift_signal = (
        compute_bounded_perturbation_metrics(normalized)
    )
    model = build_noise_compensation_model(normalized)
    compensation_factor = compute_noise_compensation(normalized, model)

    fidelity_score, stability_score, compensation_effectiveness = (
        compute_fidelity_stability_score(
            perturbation_magnitude=perturbation_magnitude,
            correction_pressure=correction_pressure,
            drift_signal=drift_signal,
            compensation_factor=compensation_factor,
        )
    )

    stable_perturbation_id = (
        perturbation_id
        if perturbation_id is not None
        else _hash_sha256({"noise": [[s, v] for s, v in normalized]})
    )
    snapshot_seed = {
        "perturbation_id": stable_perturbation_id,
        "noise_sources": [source for source, _ in normalized],
        "perturbation_magnitude": perturbation_magnitude,
        "correction_pressure": correction_pressure,
        "drift_signal": drift_signal,
        "compensation_factor": compensation_factor,
        "fidelity_score": fidelity_score,
        "model_hash": model.model_hash,
    }
    replay_hash = _hash_sha256(snapshot_seed)
    snapshot = NoisePerturbationSnapshot(
        perturbation_id=stable_perturbation_id,
        noise_sources=tuple(source for source, _ in normalized),
        perturbation_magnitude=perturbation_magnitude,
        correction_pressure=correction_pressure,
        drift_signal=drift_signal,
        compensation_factor=compensation_factor,
        fidelity_score=fidelity_score,
        replay_hash=replay_hash,
    )

    decision_payload = {
        "fidelity_score": fidelity_score,
        "stability_score": stability_score,
        "compensation_effectiveness": compensation_effectiveness,
        "balanced": bool(fidelity_score >= 0.75 and stability_score >= 0.75),
        "snapshot_replay_hash": snapshot.replay_hash,
    }
    report = FidelityStabilityReport(
        fidelity_score=fidelity_score,
        stability_score=stability_score,
        compensation_effectiveness=compensation_effectiveness,
        balanced=bool(decision_payload["balanced"]),
        decision_hash=_hash_sha256(decision_payload),
    )

    updated_ledger = append_perturbation_ledger_entry(
        ledger=prior_ledger,
        perturbation_hash=snapshot.replay_hash,
        fidelity_score=report.fidelity_score,
        stability_score=report.stability_score,
    )
    return (report, updated_ledger)
