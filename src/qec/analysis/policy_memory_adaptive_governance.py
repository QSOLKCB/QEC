"""v147.5.0 — Policy Memory / Adaptive Governance."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Optional

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.closed_loop_simulation_kernel import round12, validate_sha256_hex, validate_unit_interval
from qec.analysis.retro_trace_control_kernel import RetroTraceControlReceipt
from qec.analysis.retro_trace_forecast_kernel import RetroTraceForecastReceipt
from qec.analysis.retro_trace_policy_sensitivity import RetroTracePolicySensitivityReceipt

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_RECOMMENDATIONS = (
    "MAINTAIN_POLICY",
    "TIGHTEN_POLICY",
    "RELAX_POLICY",
    "ESCALATE_GOVERNANCE",
    "REVIEW_MEMORY",
)
_ACTION_ORDER = ("HOLD", "ADJUST", "ESCALATE")
_FACTOR_ORDER = ("stability", "locality", "forecast", "divergence")
_FORECAST_RISK_BY_CLASS = {"STABLE": 0.15, "DRIFT": 0.55, "UNSTABLE": 0.9}


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return round12(value)


def _validate_receipt_hash(receipt: object, *, field_name: str) -> str:
    if not hasattr(receipt, "to_dict"):
        raise ValueError(f"{field_name} must provide to_dict")
    payload = dict(receipt.to_dict())
    if "stable_hash" not in payload:
        raise ValueError(f"{field_name} must include stable_hash")
    observed_hash = validate_sha256_hex(str(payload.pop("stable_hash")), f"{field_name}.stable_hash")
    computed_hash = sha256_hex(payload)
    if observed_hash != computed_hash:
        raise ValueError(f"{field_name} stable_hash mismatch")
    return observed_hash


def _sorted_counts(values: tuple[str, ...], expected: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    counts = {key: 0 for key in expected}
    for value in values:
        if value not in counts:
            raise ValueError("unexpected value in deterministic counts")
        counts[value] += 1
    return tuple((key, counts[key]) for key in expected if counts[key] > 0)


@dataclass(frozen=True)
class PolicyMemoryEntry:
    entry_index: int
    control_hash: str
    decision_action: str
    control_score: float
    confidence: float
    dominant_factor: str
    sensitivity_hash: Optional[str]
    forecast_hash: Optional[str]
    memory_weight: float
    _stable_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.entry_index, bool) or not isinstance(self.entry_index, int) or self.entry_index < 0:
            raise ValueError("entry_index must be non-negative int")
        object.__setattr__(self, "control_hash", validate_sha256_hex(self.control_hash, "control_hash"))
        if self.decision_action not in _ACTION_ORDER:
            raise ValueError("decision_action must be ADJUST|ESCALATE|HOLD")
        object.__setattr__(self, "control_score", _clamp01(validate_unit_interval(self.control_score, "control_score")))
        object.__setattr__(self, "confidence", _clamp01(validate_unit_interval(self.confidence, "confidence")))
        if self.dominant_factor not in _FACTOR_ORDER:
            raise ValueError("dominant_factor must be divergence|forecast|locality|stability")
        if self.sensitivity_hash is not None:
            object.__setattr__(self, "sensitivity_hash", validate_sha256_hex(self.sensitivity_hash, "sensitivity_hash"))
        if self.forecast_hash is not None:
            object.__setattr__(self, "forecast_hash", validate_sha256_hex(self.forecast_hash, "forecast_hash"))
        object.__setattr__(self, "memory_weight", _clamp01(validate_unit_interval(self.memory_weight, "memory_weight")))
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        payload: dict[str, _JSONValue] = {
            "entry_index": self.entry_index,
            "control_hash": self.control_hash,
            "decision_action": self.decision_action,
            "control_score": round12(self.control_score),
            "confidence": round12(self.confidence),
            "dominant_factor": self.dominant_factor,
            "memory_weight": round12(self.memory_weight),
        }
        if self.sensitivity_hash is not None:
            payload["sensitivity_hash"] = self.sensitivity_hash
        if self.forecast_hash is not None:
            payload["forecast_hash"] = self.forecast_hash
        return payload

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class PolicyMemoryLedger:
    entries: tuple[PolicyMemoryEntry, ...]
    ledger_size: int
    action_counts: tuple[tuple[str, int], ...]
    dominant_factor_counts: tuple[tuple[str, int], ...]
    average_control_score: float
    average_confidence: float
    action_entropy_proxy: float
    replay_stability_score: float
    _stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("entries must be non-empty tuple")
        if any(not isinstance(item, PolicyMemoryEntry) for item in self.entries):
            raise ValueError("entries must contain PolicyMemoryEntry")
        canonical_entries = tuple(sorted(self.entries, key=lambda item: (item.control_hash, item.entry_index)))
        if canonical_entries != self.entries:
            raise ValueError("entries must be canonically ordered by (control_hash, entry_index)")
        unique_hashes = tuple(item.control_hash for item in self.entries)
        if len(set(unique_hashes)) != len(unique_hashes):
            raise ValueError("duplicate or inconsistent hashes")

        if isinstance(self.ledger_size, bool) or not isinstance(self.ledger_size, int) or self.ledger_size != len(self.entries):
            raise ValueError("ledger count mismatch")

        if not isinstance(self.action_counts, tuple):
            raise ValueError("action_counts must be tuple")
        if not isinstance(self.dominant_factor_counts, tuple):
            raise ValueError("dominant_factor_counts must be tuple")
        for name, count in self.action_counts:
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("bool where int required")
            if name not in _ACTION_ORDER:
                raise ValueError("invalid action count label")
        for name, count in self.dominant_factor_counts:
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("bool where int required")
            if name not in _FACTOR_ORDER:
                raise ValueError("invalid dominant factor count label")

        expected_action_counts = _sorted_counts(tuple(item.decision_action for item in self.entries), _ACTION_ORDER)
        expected_factor_counts = _sorted_counts(tuple(item.dominant_factor for item in self.entries), _FACTOR_ORDER)
        if self.action_counts != expected_action_counts:
            raise ValueError("action-count mismatch")
        if self.dominant_factor_counts != expected_factor_counts:
            raise ValueError("dominant-factor-count mismatch")

        object.__setattr__(
            self,
            "average_control_score",
            _clamp01(validate_unit_interval(self.average_control_score, "average_control_score")),
        )
        object.__setattr__(
            self,
            "average_confidence",
            _clamp01(validate_unit_interval(self.average_confidence, "average_confidence")),
        )
        object.__setattr__(
            self,
            "action_entropy_proxy",
            _clamp01(validate_unit_interval(self.action_entropy_proxy, "action_entropy_proxy")),
        )
        object.__setattr__(
            self,
            "replay_stability_score",
            _clamp01(validate_unit_interval(self.replay_stability_score, "replay_stability_score")),
        )

        average_control_score = _clamp01(sum(item.control_score for item in self.entries) / float(len(self.entries)))
        average_confidence = _clamp01(sum(item.confidence for item in self.entries) / float(len(self.entries)))
        probs = tuple(count / float(len(self.entries)) for _, count in self.action_counts)
        entropy = -sum(prob * log2(prob) for prob in probs if prob > 0.0)
        max_entropy = log2(float(len(_ACTION_ORDER)))
        entropy_proxy = _clamp01(entropy / max_entropy if max_entropy > 0.0 else 0.0)
        replay_stability_score = _clamp01(1.0 - entropy_proxy)

        if round12(self.average_control_score) != round12(average_control_score):
            raise ValueError("average_control_score mismatch")
        if round12(self.average_confidence) != round12(average_confidence):
            raise ValueError("average_confidence mismatch")
        if round12(self.action_entropy_proxy) != round12(entropy_proxy):
            raise ValueError("action_entropy_proxy mismatch")
        if round12(self.replay_stability_score) != round12(replay_stability_score):
            raise ValueError("replay_stability_score mismatch")

        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "entries": tuple(item.to_dict() for item in self.entries),
            "ledger_size": self.ledger_size,
            "action_counts": tuple((name, count) for name, count in self.action_counts),
            "dominant_factor_counts": tuple((name, count) for name, count in self.dominant_factor_counts),
            "average_control_score": round12(self.average_control_score),
            "average_confidence": round12(self.average_confidence),
            "action_entropy_proxy": round12(self.action_entropy_proxy),
            "replay_stability_score": round12(self.replay_stability_score),
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
class AdaptiveGovernanceSignal:
    name: str
    value: float
    _stable_hash: str

    def __post_init__(self) -> None:
        expected = (
            "adjustment_pressure",
            "escalation_pressure",
            "forecast_instability_memory",
            "governance_confidence",
            "hold_stability",
            "policy_volatility",
        )
        if self.name not in expected:
            raise ValueError("signal name must be canonical")
        object.__setattr__(self, "value", _clamp01(validate_unit_interval(self.value, "value")))
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {"name": self.name, "value": round12(self.value)}

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class AdaptiveGovernanceRecommendation:
    label: str
    rationale: str
    rank: int
    _stable_hash: str

    def __post_init__(self) -> None:
        if self.label not in _RECOMMENDATIONS:
            raise ValueError("invalid recommendation labels")
        if not isinstance(self.rationale, str) or not self.rationale:
            raise ValueError("rationale must be non-empty string")
        expected_rank = {
            "MAINTAIN_POLICY": 0,
            "TIGHTEN_POLICY": 1,
            "RELAX_POLICY": 2,
            "ESCALATE_GOVERNANCE": 3,
            "REVIEW_MEMORY": 4,
        }[self.label]
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank != expected_rank:
            raise ValueError("bool where int required")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {"label": self.label, "rationale": self.rationale, "rank": self.rank}

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class PolicyMemoryGovernanceSummary:
    recommendation_label: str
    governance_confidence: float
    replay_stability_score: float
    entry_count: int
    _stable_hash: str

    def __post_init__(self) -> None:
        if self.recommendation_label not in _RECOMMENDATIONS:
            raise ValueError("invalid recommendation labels")
        object.__setattr__(
            self,
            "governance_confidence",
            _clamp01(validate_unit_interval(self.governance_confidence, "governance_confidence")),
        )
        object.__setattr__(
            self,
            "replay_stability_score",
            _clamp01(validate_unit_interval(self.replay_stability_score, "replay_stability_score")),
        )
        if isinstance(self.entry_count, bool) or not isinstance(self.entry_count, int) or self.entry_count < 1:
            raise ValueError("entry_count must be positive int")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "recommendation_label": self.recommendation_label,
            "governance_confidence": round12(self.governance_confidence),
            "replay_stability_score": round12(self.replay_stability_score),
            "entry_count": self.entry_count,
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
class PolicyMemoryGovernanceReceipt:
    ledger: PolicyMemoryLedger
    signals: tuple[AdaptiveGovernanceSignal, ...]
    recommendation: AdaptiveGovernanceRecommendation
    summary: PolicyMemoryGovernanceSummary
    _stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.ledger, PolicyMemoryLedger):
            raise ValueError("ledger must be PolicyMemoryLedger")
        if not isinstance(self.signals, tuple) or len(self.signals) != 6:
            raise ValueError("signals must be tuple of length 6")
        if any(not isinstance(item, AdaptiveGovernanceSignal) for item in self.signals):
            raise ValueError("signals contains invalid item")
        canonical_names = tuple(sorted(signal.name for signal in self.signals))
        if canonical_names != (
            "adjustment_pressure",
            "escalation_pressure",
            "forecast_instability_memory",
            "governance_confidence",
            "hold_stability",
            "policy_volatility",
        ):
            raise ValueError("signals must use canonical ordering")
        if self.signals != tuple(sorted(self.signals, key=lambda item: item.name)):
            raise ValueError("signals must use canonical ordering")
        if not isinstance(self.recommendation, AdaptiveGovernanceRecommendation):
            raise ValueError("recommendation must be AdaptiveGovernanceRecommendation")
        if not isinstance(self.summary, PolicyMemoryGovernanceSummary):
            raise ValueError("summary must be PolicyMemoryGovernanceSummary")

        governance_confidence = dict((signal.name, signal.value) for signal in self.signals)["governance_confidence"]
        if self.summary.recommendation_label != self.recommendation.label:
            raise ValueError("summary mismatch")
        if round12(self.summary.governance_confidence) != round12(governance_confidence):
            raise ValueError("summary mismatch")
        if round12(self.summary.replay_stability_score) != round12(self.ledger.replay_stability_score):
            raise ValueError("summary mismatch")
        if self.summary.entry_count != self.ledger.ledger_size:
            raise ValueError("ledger count mismatch")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "ledger": self.ledger.to_dict(),
            "signals": tuple(item.to_dict() for item in self.signals),
            "recommendation": self.recommendation.to_dict(),
            "summary": self.summary.to_dict(),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def _recommendation_from_signals(signals: dict[str, float]) -> str:
    escalation_pressure = signals["escalation_pressure"]
    governance_confidence = signals["governance_confidence"]
    adjustment_pressure = signals["adjustment_pressure"]
    hold_stability = signals["hold_stability"]
    policy_volatility = signals["policy_volatility"]
    if escalation_pressure >= 0.66 and governance_confidence >= 0.5:
        return "ESCALATE_GOVERNANCE"
    if adjustment_pressure >= 0.5:
        return "TIGHTEN_POLICY"
    if hold_stability >= 0.75 and policy_volatility < 0.25:
        return "MAINTAIN_POLICY"
    if governance_confidence < 0.33:
        return "REVIEW_MEMORY"
    return "RELAX_POLICY"


def build_policy_memory_governance(
    control_receipts: tuple[RetroTraceControlReceipt, ...],
    *,
    sensitivity_receipts: tuple[RetroTracePolicySensitivityReceipt, ...] = (),
    forecast_receipts: tuple[RetroTraceForecastReceipt, ...] = (),
) -> PolicyMemoryGovernanceReceipt:
    if not isinstance(control_receipts, tuple) or not control_receipts:
        raise ValueError("empty control_receipts")
    if not isinstance(sensitivity_receipts, tuple):
        raise ValueError("sensitivity_receipts must be tuple")
    if not isinstance(forecast_receipts, tuple):
        raise ValueError("forecast_receipts must be tuple")

    sensitivity_by_hash: dict[str, RetroTracePolicySensitivityReceipt] = {}
    for idx, sensitivity in enumerate(sensitivity_receipts):
        if not isinstance(sensitivity, RetroTracePolicySensitivityReceipt):
            raise ValueError("invalid receipt types")
        sensitivity_hash = _validate_receipt_hash(sensitivity, field_name=f"sensitivity_receipts[{idx}]")
        if sensitivity_hash in sensitivity_by_hash:
            raise ValueError("duplicate or inconsistent hashes")
        sensitivity_by_hash[sensitivity_hash] = sensitivity

    forecast_by_hash: dict[str, RetroTraceForecastReceipt] = {}
    for idx, forecast in enumerate(forecast_receipts):
        if not isinstance(forecast, RetroTraceForecastReceipt):
            raise ValueError("invalid receipt types")
        forecast_hash = _validate_receipt_hash(forecast, field_name=f"forecast_receipts[{idx}]")
        if forecast_hash in forecast_by_hash:
            raise ValueError("duplicate or inconsistent hashes")
        forecast_by_hash[forecast_hash] = forecast

    entries_unsorted: list[PolicyMemoryEntry] = []
    for idx, control in enumerate(control_receipts):
        if not isinstance(control, RetroTraceControlReceipt):
            raise ValueError("invalid receipt types")
        control_hash = _validate_receipt_hash(control, field_name=f"control_receipts[{idx}]")
        sensitivity_hash = control.sensitivity_hash
        forecast_hash = control.forecast_hash
        if sensitivity_receipts and sensitivity_hash not in sensitivity_by_hash:
            raise ValueError(f"missing sensitivity receipt for control_receipts[{idx}]")
        if forecast_receipts and forecast_hash not in forecast_by_hash:
            raise ValueError(f"missing forecast receipt for control_receipts[{idx}]")

        memory_weight = _clamp01((0.6 * control.decision.confidence) + (0.4 * control.decision.control_score))
        entry_payload = {
            "entry_index": idx,
            "control_hash": control_hash,
            "decision_action": control.decision.action.action,
            "control_score": round12(control.decision.control_score),
            "confidence": round12(control.decision.confidence),
            "dominant_factor": control.summary.dominant_factor,
            "sensitivity_hash": sensitivity_hash,
            "forecast_hash": forecast_hash,
            "memory_weight": round12(memory_weight),
        }
        entries_unsorted.append(
            PolicyMemoryEntry(
                entry_index=idx,
                control_hash=control_hash,
                decision_action=control.decision.action.action,
                control_score=control.decision.control_score,
                confidence=control.decision.confidence,
                dominant_factor=control.summary.dominant_factor,
                sensitivity_hash=sensitivity_hash,
                forecast_hash=forecast_hash,
                memory_weight=memory_weight,
                _stable_hash=sha256_hex(entry_payload),
            )
        )

    ordered_entries = tuple(sorted(entries_unsorted, key=lambda item: (item.control_hash, item.entry_index)))
    if len(set(entry.control_hash for entry in ordered_entries)) != len(ordered_entries):
        raise ValueError("duplicate or inconsistent hashes")

    action_counts = _sorted_counts(tuple(item.decision_action for item in ordered_entries), _ACTION_ORDER)
    dominant_factor_counts = _sorted_counts(tuple(item.dominant_factor for item in ordered_entries), _FACTOR_ORDER)
    avg_score = _clamp01(sum(item.control_score for item in ordered_entries) / float(len(ordered_entries)))
    avg_conf = _clamp01(sum(item.confidence for item in ordered_entries) / float(len(ordered_entries)))

    probs = tuple(count / float(len(ordered_entries)) for _, count in action_counts)
    entropy = -sum(prob * log2(prob) for prob in probs if prob > 0.0)
    entropy_proxy = _clamp01(entropy / log2(float(len(_ACTION_ORDER))))
    replay_stability = _clamp01(1.0 - entropy_proxy)

    ledger_payload = {
        "entries": tuple(item.to_dict() for item in ordered_entries),
        "ledger_size": len(ordered_entries),
        "action_counts": action_counts,
        "dominant_factor_counts": dominant_factor_counts,
        "average_control_score": round12(avg_score),
        "average_confidence": round12(avg_conf),
        "action_entropy_proxy": round12(entropy_proxy),
        "replay_stability_score": round12(replay_stability),
    }
    ledger = PolicyMemoryLedger(
        entries=ordered_entries,
        ledger_size=len(ordered_entries),
        action_counts=action_counts,
        dominant_factor_counts=dominant_factor_counts,
        average_control_score=avg_score,
        average_confidence=avg_conf,
        action_entropy_proxy=entropy_proxy,
        replay_stability_score=replay_stability,
        _stable_hash=sha256_hex(ledger_payload),
    )

    escalation_share = sum(entry.memory_weight for entry in ordered_entries if entry.decision_action == "ESCALATE") / float(
        len(ordered_entries)
    )
    adjustment_share = sum(entry.memory_weight for entry in ordered_entries if entry.decision_action == "ADJUST") / float(
        len(ordered_entries)
    )
    hold_share = sum(entry.memory_weight for entry in ordered_entries if entry.decision_action == "HOLD") / float(len(ordered_entries))

    forecast_risks = []
    for entry in ordered_entries:
        forecast = forecast_by_hash.get(entry.forecast_hash) if entry.forecast_hash is not None else None
        if forecast is not None:
            forecast_risks.append(_FORECAST_RISK_BY_CLASS[forecast.summary.collapse_risk_classification])
        else:
            forecast_risks.append(1.0 - entry.control_score)

    memory_density = _clamp01(float(len(ordered_entries)) / float(len(ordered_entries) + 1))
    forecast_instability_memory = _clamp01(sum(forecast_risks) / float(len(forecast_risks)))
    signal_values = {
        "adjustment_pressure": _clamp01((adjustment_share * memory_density) + (0.10 * ledger.action_entropy_proxy)),
        "escalation_pressure": _clamp01(
            (0.65 * forecast_instability_memory) + (0.25 * adjustment_share) + (0.10 * escalation_share)
        ),
        "forecast_instability_memory": forecast_instability_memory,
        "hold_stability": _clamp01(hold_share + (0.25 * ledger.replay_stability_score)),
        "policy_volatility": _clamp01((0.65 * ledger.action_entropy_proxy) + (0.35 * (1.0 - ledger.replay_stability_score))),
    }
    signal_values["governance_confidence"] = _clamp01(
        ledger.average_confidence
        * memory_density
        * (1.0 - 0.15 * forecast_instability_memory)
        * (1.0 - 0.5 * signal_values["policy_volatility"])
    )

    for name, value in signal_values.items():
        validate_unit_interval(value, name)

    signal_objects = []
    for name in sorted(signal_values.keys()):
        payload = {"name": name, "value": round12(signal_values[name])}
        signal_objects.append(AdaptiveGovernanceSignal(name=name, value=signal_values[name], _stable_hash=sha256_hex(payload)))

    recommendation_label = _recommendation_from_signals(signal_values)
    rationale = (
        f"recommendation={recommendation_label};"
        f"escalation={round12(signal_values['escalation_pressure'])};"
        f"adjustment={round12(signal_values['adjustment_pressure'])};"
        f"hold={round12(signal_values['hold_stability'])};"
        f"confidence={round12(signal_values['governance_confidence'])}"
    )
    rank = {
        "MAINTAIN_POLICY": 0,
        "TIGHTEN_POLICY": 1,
        "RELAX_POLICY": 2,
        "ESCALATE_GOVERNANCE": 3,
        "REVIEW_MEMORY": 4,
    }[recommendation_label]
    recommendation_payload = {"label": recommendation_label, "rationale": rationale, "rank": rank}
    recommendation = AdaptiveGovernanceRecommendation(
        label=recommendation_label,
        rationale=rationale,
        rank=rank,
        _stable_hash=sha256_hex(recommendation_payload),
    )

    summary_payload = {
        "recommendation_label": recommendation_label,
        "governance_confidence": round12(signal_values["governance_confidence"]),
        "replay_stability_score": round12(ledger.replay_stability_score),
        "entry_count": ledger.ledger_size,
    }
    summary = PolicyMemoryGovernanceSummary(
        recommendation_label=recommendation_label,
        governance_confidence=signal_values["governance_confidence"],
        replay_stability_score=ledger.replay_stability_score,
        entry_count=ledger.ledger_size,
        _stable_hash=sha256_hex(summary_payload),
    )

    receipt_payload = {
        "ledger": ledger.to_dict(),
        "signals": tuple(item.to_dict() for item in signal_objects),
        "recommendation": recommendation.to_dict(),
        "summary": summary.to_dict(),
    }
    return PolicyMemoryGovernanceReceipt(
        ledger=ledger,
        signals=tuple(signal_objects),
        recommendation=recommendation,
        summary=summary,
        _stable_hash=sha256_hex(receipt_payload),
    )


__all__ = [
    "AdaptiveGovernanceRecommendation",
    "AdaptiveGovernanceSignal",
    "PolicyMemoryEntry",
    "PolicyMemoryGovernanceReceipt",
    "PolicyMemoryGovernanceSummary",
    "PolicyMemoryLedger",
    "build_policy_memory_governance",
]
