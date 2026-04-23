"""v144.2 — deterministic state-conditioned filter mesh scorer."""

from __future__ import annotations

from dataclasses import dataclass
import math

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

WEIGHT_INVARIANT_ALIGNMENT = 0.30
WEIGHT_HARDWARE_ALIGNMENT = 0.30
WEIGHT_RECURRENCE_AVOID = 0.20
WEIGHT_STABILITY = 0.20

STRONG_PREFERENCE_THRESHOLD = 0.75
WEAK_PREFERENCE_THRESHOLD = 0.60
PREFERENCE_MARGIN_THRESHOLD = 0.10

CLASSIFICATION_STABLE_PREFERENCE = "stable_preference"
CLASSIFICATION_WEAK_PREFERENCE = "weak_preference"
CLASSIFICATION_CONFLICTED = "conflicted"

_ALLOWED_CLASSIFICATIONS = frozenset(
    {CLASSIFICATION_STABLE_PREFERENCE, CLASSIFICATION_WEAK_PREFERENCE, CLASSIFICATION_CONFLICTED}
)

_DOMINANT_PRESSURE_ORDER = ("thermal_pressure", "latency_drift", "timing_skew", "power_pressure")

_REGIME_FILTER_HINTS: dict[str, tuple[str, ...]] = {
    "high_frequency": ("spectral", "phase", "timing"),
    "low_frequency": ("smoothing", "stability", "damping"),
    "resonant": ("resonance", "damping", "stability"),
}
_INVARIANT_FILTER_HINTS: dict[str, tuple[str, ...]] = {
    "parity": ("parity", "syndrome", "consistency"),
    "topological": ("topology", "geometry", "boundary"),
    "stabilizer": ("stabilizer", "consensus", "damping"),
}
_GEOMETRY_CONTROL_HINTS: dict[str, tuple[str, ...]] = {
    "surface": ("boundary", "lattice", "surface"),
    "color": ("color", "triad", "cluster"),
    "toric": ("cycle", "toric", "loop"),
}
_HARDWARE_PRIORITY_FILTERS: dict[str, tuple[str, ...]] = {
    "thermal_pressure": ("thermal", "cool", "damp", "stabilize"),
    "latency_drift": ("latency", "queue", "pipeline", "buffer"),
    "timing_skew": ("timing", "clock", "sync", "phase"),
    "power_pressure": ("power", "load", "budget", "throttle"),
}
_HARDWARE_CLASS_HINTS: dict[str, tuple[str, ...]] = {
    "superconducting": ("thermal", "timing", "stabilize"),
    "photonic": ("latency", "phase", "sync"),
    "iontrap": ("timing", "power", "stabilize"),
}
_RESONANCE_AMPLIFYING_HINTS = ("resonance", "feedback", "amplify", "loop")
_RESONANCE_DAMPING_HINTS = ("damp", "stabilize", "quench", "suppress")
_STABILITY_HINTS = ("stabilize", "consensus", "damp", "bound")


_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _normalize_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    if normalized != value:
        raise ValueError(f"{field_name} must not include leading/trailing whitespace")
    return normalized


def _validate_unit_interval(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric in [0,1]")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0,1]")
    return numeric


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _round12(value: float) -> float:
    return round(float(value), 12)


def _has_hint(filters: tuple[str, ...], hints: tuple[str, ...]) -> bool:
    lowered = tuple(token.lower() for token in filters)
    return any(hint in token for token in lowered for hint in hints)


def _signature_for_filters(input_filters: tuple[str, ...], control_filters: tuple[str, ...]) -> str:
    return canonical_json({"in": list(input_filters), "ctrl": list(control_filters)})


def _ordering_payload_without_hash(
    input_filters: tuple[str, ...], control_filters: tuple[str, ...], ordering_signature: str
) -> dict[str, _JSONValue]:
    return {
        "input_filters": input_filters,
        "control_filters": control_filters,
        "ordering_signature": ordering_signature,
    }


@dataclass(frozen=True)
class FilterMeshState:
    invariant_class: str
    geometry_class: str
    spectral_regime: str
    hardware_class: str
    recurrence_class: str
    thermal_pressure: float
    latency_drift: float
    timing_skew: float
    power_pressure: float
    consensus_instability: float

    def __post_init__(self) -> None:
        for field_name in (
            "invariant_class",
            "geometry_class",
            "spectral_regime",
            "hardware_class",
            "recurrence_class",
        ):
            object.__setattr__(self, field_name, _normalize_string(getattr(self, field_name), field_name))
        for field_name in (
            "thermal_pressure",
            "latency_drift",
            "timing_skew",
            "power_pressure",
            "consensus_instability",
        ):
            object.__setattr__(self, field_name, _validate_unit_interval(getattr(self, field_name), field_name))

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "invariant_class": self.invariant_class,
            "geometry_class": self.geometry_class,
            "spectral_regime": self.spectral_regime,
            "hardware_class": self.hardware_class,
            "recurrence_class": self.recurrence_class,
            "thermal_pressure": _round12(self.thermal_pressure),
            "latency_drift": _round12(self.latency_drift),
            "timing_skew": _round12(self.timing_skew),
            "power_pressure": _round12(self.power_pressure),
            "consensus_instability": _round12(self.consensus_instability),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class FilterOrdering:
    input_filters: tuple[str, ...]
    control_filters: tuple[str, ...]
    ordering_signature: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.input_filters, tuple) or not self.input_filters:
            raise ValueError("input_filters must be a non-empty tuple")
        if not isinstance(self.control_filters, tuple) or not self.control_filters:
            raise ValueError("control_filters must be a non-empty tuple")
        object.__setattr__(
            self,
            "input_filters",
            tuple(_normalize_string(filter_name, "input_filters item") for filter_name in self.input_filters),
        )
        object.__setattr__(
            self,
            "control_filters",
            tuple(_normalize_string(filter_name, "control_filters item") for filter_name in self.control_filters),
        )
        object.__setattr__(self, "ordering_signature", _normalize_string(self.ordering_signature, "ordering_signature"))
        expected_signature = self.deterministic_signature(self.input_filters, self.control_filters)
        if self.ordering_signature != expected_signature:
            raise ValueError("ordering_signature must deterministically reflect input_filters and control_filters")
        expected_hash = self.computed_stable_hash()
        if self.stable_hash != expected_hash:
            raise ValueError("stable_hash must match canonical ordering payload")

    @staticmethod
    def deterministic_signature(input_filters: tuple[str, ...], control_filters: tuple[str, ...]) -> str:
        return _signature_for_filters(input_filters, control_filters)

    @classmethod
    def build(cls, input_filters: tuple[str, ...], control_filters: tuple[str, ...]) -> "FilterOrdering":
        signature = cls.deterministic_signature(input_filters, control_filters)
        stable_hash = sha256_hex(_ordering_payload_without_hash(input_filters, control_filters, signature))
        return cls(
            input_filters=input_filters,
            control_filters=control_filters,
            ordering_signature=signature,
            stable_hash=stable_hash,
        )

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "input_filters": self.input_filters,
            "control_filters": self.control_filters,
            "ordering_signature": self.ordering_signature,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(_ordering_payload_without_hash(self.input_filters, self.control_filters, self.ordering_signature))


@dataclass(frozen=True)
class OrderingScore:
    ordering_signature: str
    invariant_alignment: float
    hardware_alignment: float
    recurrence_avoidance: float
    stability_alignment: float
    total_score: float
    rank: int
    stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordering_signature", _normalize_string(self.ordering_signature, "ordering_signature"))
        for field_name in (
            "invariant_alignment",
            "hardware_alignment",
            "recurrence_avoidance",
            "stability_alignment",
            "total_score",
        ):
            object.__setattr__(self, field_name, _validate_unit_interval(getattr(self, field_name), field_name))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be int >= 1")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical score payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "ordering_signature": self.ordering_signature,
            "invariant_alignment": _round12(self.invariant_alignment),
            "hardware_alignment": _round12(self.hardware_alignment),
            "recurrence_avoidance": _round12(self.recurrence_avoidance),
            "stability_alignment": _round12(self.stability_alignment),
            "total_score": _round12(self.total_score),
            "rank": self.rank,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            **self._payload_without_hash(),
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


@dataclass(frozen=True)
class FilterMeshReceipt:
    state: FilterMeshState
    candidate_count: int
    ordered_scores: tuple[OrderingScore, ...]
    dominant_ordering_signature: str
    dominant_score: float
    classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, FilterMeshState):
            raise ValueError("state must be a FilterMeshState")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if not isinstance(self.ordered_scores, tuple) or len(self.ordered_scores) != self.candidate_count:
            raise ValueError("ordered_scores must be tuple with candidate_count entries")
        if any(not isinstance(item, OrderingScore) for item in self.ordered_scores):
            raise ValueError("ordered_scores must contain OrderingScore entries")
        expected_sort = tuple(
            sorted(
                self.ordered_scores,
                key=lambda item: (-_round12(item.total_score), item.ordering_signature, item.stable_hash),
            )
        )
        if expected_sort != self.ordered_scores:
            raise ValueError("ordered_scores must be sorted canonically")
        for expected_rank, score in enumerate(self.ordered_scores, start=1):
            if score.rank != expected_rank:
                raise ValueError("invalid ranking state")
        top = self.ordered_scores[0]
        object.__setattr__(
            self,
            "dominant_ordering_signature",
            _normalize_string(self.dominant_ordering_signature, "dominant_ordering_signature"),
        )
        object.__setattr__(self, "dominant_score", _validate_unit_interval(self.dominant_score, "dominant_score"))
        if self.dominant_ordering_signature != top.ordering_signature:
            raise ValueError("dominant_ordering_signature must match top-ranked ordering")
        if self.dominant_score != top.total_score:
            raise ValueError("dominant_score must match top-ranked total_score")
        if self.classification not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError("classification is invalid")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical receipt payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "state": self.state.to_dict(),
            "candidate_count": self.candidate_count,
            "ordered_scores": tuple(score.to_dict() for score in self.ordered_scores),
            "dominant_ordering_signature": self.dominant_ordering_signature,
            "dominant_score": _round12(self.dominant_score),
            "classification": self.classification,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            **self._payload_without_hash(),
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


def _score_invariant_alignment(state: FilterMeshState, ordering: FilterOrdering) -> float:
    all_filters = ordering.input_filters + ordering.control_filters
    score = 0.5
    regime_hints = _REGIME_FILTER_HINTS.get(state.spectral_regime, ())
    invariant_hints = _INVARIANT_FILTER_HINTS.get(state.invariant_class, ())
    if _has_hint(all_filters, regime_hints + invariant_hints):
        score += 0.25
    geometry_hints = _GEOMETRY_CONTROL_HINTS.get(state.geometry_class, ())
    if _has_hint(ordering.control_filters, geometry_hints):
        score += 0.25
    return _clamp01(score)


def _dominant_pressure_metric(state: FilterMeshState) -> str:
    metrics = {
        "thermal_pressure": state.thermal_pressure,
        "latency_drift": state.latency_drift,
        "timing_skew": state.timing_skew,
        "power_pressure": state.power_pressure,
    }
    return max(_DOMINANT_PRESSURE_ORDER, key=lambda key: (metrics[key], -_DOMINANT_PRESSURE_ORDER.index(key)))


def _score_hardware_alignment(state: FilterMeshState, ordering: FilterOrdering) -> float:
    combined = ordering.input_filters + ordering.control_filters
    dominant_metric = _dominant_pressure_metric(state)
    hints = _HARDWARE_PRIORITY_FILTERS[dominant_metric]
    score = 0.4
    if combined and _has_hint((combined[0],), hints):
        score += 0.35
    elif len(combined) > 1 and _has_hint((combined[1],), hints):
        score += 0.2
    hardware_hints = _HARDWARE_CLASS_HINTS.get(state.hardware_class, ())
    if _has_hint(combined, hardware_hints):
        score += 0.15
    return _clamp01(score)


def _score_recurrence_avoidance(state: FilterMeshState, ordering: FilterOrdering) -> float:
    all_filters = ordering.input_filters + ordering.control_filters
    score = 0.6
    if state.recurrence_class == "oscillatory":
        if _has_hint(all_filters, _RESONANCE_AMPLIFYING_HINTS):
            score -= 0.4
        if _has_hint(all_filters, _RESONANCE_DAMPING_HINTS):
            score += 0.2
    else:
        if _has_hint(all_filters, _RESONANCE_DAMPING_HINTS):
            score += 0.1
    return _clamp01(score)


def _score_stability_alignment(state: FilterMeshState, ordering: FilterOrdering) -> float:
    combined = ordering.input_filters + ordering.control_filters
    score = 0.5
    if state.consensus_instability >= 0.6:
        if combined and _has_hint((combined[0],), _STABILITY_HINTS):
            score += 0.3
        elif len(combined) > 1 and _has_hint((combined[1],), _STABILITY_HINTS):
            score += 0.15
        else:
            score -= 0.2
    pressures = (state.thermal_pressure, state.latency_drift, state.timing_skew, state.power_pressure)
    if max(pressures) - min(pressures) <= 0.15 and len(combined) >= 2 and combined[0] != combined[1]:
        score += 0.2
    return _clamp01(score)


def _classify_scores(ordered_scores: tuple[OrderingScore, ...]) -> str:
    dominant_score = ordered_scores[0].total_score
    if len(ordered_scores) == 1:
        if dominant_score >= STRONG_PREFERENCE_THRESHOLD:
            return CLASSIFICATION_STABLE_PREFERENCE
        if dominant_score >= WEAK_PREFERENCE_THRESHOLD:
            return CLASSIFICATION_WEAK_PREFERENCE
        return CLASSIFICATION_CONFLICTED
    margin = dominant_score - ordered_scores[1].total_score
    if dominant_score >= STRONG_PREFERENCE_THRESHOLD and margin >= PREFERENCE_MARGIN_THRESHOLD:
        return CLASSIFICATION_STABLE_PREFERENCE
    if dominant_score >= WEAK_PREFERENCE_THRESHOLD:
        return CLASSIFICATION_WEAK_PREFERENCE
    return CLASSIFICATION_CONFLICTED


def score_filter_mesh(state: FilterMeshState, candidates: tuple[FilterOrdering, ...]) -> FilterMeshReceipt:
    if not isinstance(state, FilterMeshState):
        raise ValueError("state must be a FilterMeshState")
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError("candidates must be a non-empty tuple")
    if any(not isinstance(candidate, FilterOrdering) for candidate in candidates):
        raise ValueError("candidates must contain FilterOrdering entries")

    signatures = tuple(candidate.ordering_signature for candidate in candidates)
    if len(set(signatures)) != len(signatures):
        raise ValueError("duplicate ordering_signature values are not allowed")

    hash_to_sig: dict[str, str] = {}
    for candidate in candidates:
        if candidate.stable_hash in hash_to_sig and hash_to_sig[candidate.stable_hash] != candidate.ordering_signature:
            raise ValueError("duplicate stable_hash across distinct orderings is not allowed")
        hash_to_sig[candidate.stable_hash] = candidate.ordering_signature

    scored: list[tuple[float, str, str, float, float, float, float]] = []
    for candidate in candidates:
        invariant_alignment = _score_invariant_alignment(state, candidate)
        hardware_alignment = _score_hardware_alignment(state, candidate)
        recurrence_avoidance = _score_recurrence_avoidance(state, candidate)
        stability_alignment = _score_stability_alignment(state, candidate)
        total_score = _clamp01(
            (WEIGHT_INVARIANT_ALIGNMENT * invariant_alignment)
            + (WEIGHT_HARDWARE_ALIGNMENT * hardware_alignment)
            + (WEIGHT_RECURRENCE_AVOID * recurrence_avoidance)
            + (WEIGHT_STABILITY * stability_alignment)
        )
        scored.append(
            (
                total_score,
                candidate.ordering_signature,
                candidate.stable_hash,
                invariant_alignment,
                hardware_alignment,
                recurrence_avoidance,
                stability_alignment,
            )
        )

    scored.sort(key=lambda row: (-_round12(row[0]), row[1], row[2]))

    ordered_scores: list[OrderingScore] = []
    for rank, item in enumerate(scored, start=1):
        total_score, ordering_signature, _, invariant_alignment, hardware_alignment, recurrence_avoidance, stability_alignment = item
        score_payload = {
            "ordering_signature": ordering_signature,
            "invariant_alignment": _round12(invariant_alignment),
            "hardware_alignment": _round12(hardware_alignment),
            "recurrence_avoidance": _round12(recurrence_avoidance),
            "stability_alignment": _round12(stability_alignment),
            "total_score": _round12(total_score),
            "rank": rank,
        }
        ordered_scores.append(
            OrderingScore(
                ordering_signature=ordering_signature,
                invariant_alignment=invariant_alignment,
                hardware_alignment=hardware_alignment,
                recurrence_avoidance=recurrence_avoidance,
                stability_alignment=stability_alignment,
                total_score=total_score,
                rank=rank,
                stable_hash=sha256_hex(score_payload),
            )
        )

    ordered_scores_tuple = tuple(ordered_scores)
    classification = _classify_scores(ordered_scores_tuple)
    dominant_score = ordered_scores_tuple[0].total_score
    receipt_payload = {
        "state": state.to_dict(),
        "candidate_count": len(candidates),
        "ordered_scores": tuple(score.to_dict() for score in ordered_scores_tuple),
        "dominant_ordering_signature": ordered_scores_tuple[0].ordering_signature,
        "dominant_score": _round12(dominant_score),
        "classification": classification,
    }
    return FilterMeshReceipt(
        state=state,
        candidate_count=len(candidates),
        ordered_scores=ordered_scores_tuple,
        dominant_ordering_signature=ordered_scores_tuple[0].ordering_signature,
        dominant_score=dominant_score,
        classification=classification,
        stable_hash=sha256_hex(receipt_payload),
    )


__all__ = [
    "CLASSIFICATION_CONFLICTED",
    "CLASSIFICATION_STABLE_PREFERENCE",
    "CLASSIFICATION_WEAK_PREFERENCE",
    "FilterMeshReceipt",
    "FilterMeshState",
    "FilterOrdering",
    "OrderingScore",
    "PREFERENCE_MARGIN_THRESHOLD",
    "STRONG_PREFERENCE_THRESHOLD",
    "WEAK_PREFERENCE_THRESHOLD",
    "WEIGHT_HARDWARE_ALIGNMENT",
    "WEIGHT_INVARIANT_ALIGNMENT",
    "WEIGHT_RECURRENCE_AVOID",
    "WEIGHT_STABILITY",
    "score_filter_mesh",
]
