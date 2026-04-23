"""v147.2.0 — Retro Trace Policy Sensitivity Kernel."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from itertools import combinations

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.closed_loop_simulation_kernel import round12, validate_sha256_hex, validate_unit_interval
from qec.analysis.governed_orchestration_layer import GovernancePolicy
from qec.analysis.retro_trace_intake_bridge import RetroTraceReceipt

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_MIN_POLICY_COUNT = 2
_CLASS_LOW = "LOW"
_CLASS_MODERATE = "MODERATE"
_CLASS_HIGH = "HIGH"


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return round12(value)


def _trace_features(retro_trace: RetroTraceReceipt) -> dict[str, float]:
    metrics = dict(retro_trace.trace_metrics)
    trace_length = float(retro_trace.trace_length)
    trace_length_norm = _clamp01(trace_length / float(retro_trace.trace_length + 1))
    timing_density = _clamp01(float(len(retro_trace.normalized_timing)) / trace_length) if retro_trace.trace_length else 0.0
    sparsity = _clamp01(float(metrics["input_sparsity"]))
    observability = _clamp01((float(metrics["trace_completeness"]) + float(metrics["timing_observability"])) / 2.0)
    ordering_integrity = _clamp01(float(metrics["event_order_integrity"]))
    return {
        "trace_length": trace_length_norm,
        "timing_density": timing_density,
        "sparsity": sparsity,
        "observability": observability,
        "ordering_integrity": ordering_integrity,
    }


def _alignment(observed: float, threshold: float) -> float:
    return _clamp01((observed - threshold + 1.0) / 2.0)


@dataclass(frozen=True)
class RetroTracePolicyRun:
    policy_hash: str
    strictness: float
    compatibility: float
    metrics: tuple[tuple[str, float], ...]
    _stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_hash", validate_sha256_hex(self.policy_hash, "policy_hash"))
        object.__setattr__(self, "strictness", validate_unit_interval(self.strictness, "strictness"))
        object.__setattr__(self, "compatibility", validate_unit_interval(self.compatibility, "compatibility"))
        if not isinstance(self.metrics, tuple):
            raise ValueError("metrics must be tuple")
        expected = ("ordering_integrity", "observability", "sparsity", "timing_density", "trace_length")
        keys = tuple(name for name, _ in self.metrics)
        if keys != expected:
            raise ValueError("metrics must use canonical deterministic key ordering")
        normalized = []
        for name, value in self.metrics:
            normalized.append((name, validate_unit_interval(value, f"metrics[{name}]")))
        object.__setattr__(self, "metrics", tuple(normalized))
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "policy_hash": self.policy_hash,
            "strictness": round12(self.strictness),
            "compatibility": round12(self.compatibility),
            "metrics": tuple((name, round12(value)) for name, value in self.metrics),
        }

    def stable_hash(self) -> str:
        return self._stable_hash

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class RetroTracePolicyComparison:
    left_policy_hash: str
    right_policy_hash: str
    strictness_delta: float
    compatibility_delta: float
    metric_distance: float
    sensitivity_score: float
    _stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_policy_hash", validate_sha256_hex(self.left_policy_hash, "left_policy_hash"))
        object.__setattr__(self, "right_policy_hash", validate_sha256_hex(self.right_policy_hash, "right_policy_hash"))
        if self.left_policy_hash > self.right_policy_hash:
            raise ValueError("policy comparison must use canonical hash ordering")
        object.__setattr__(self, "metric_distance", validate_unit_interval(self.metric_distance, "metric_distance"))
        object.__setattr__(self, "sensitivity_score", validate_unit_interval(self.sensitivity_score, "sensitivity_score"))
        for field in ("strictness_delta", "compatibility_delta"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be numeric")
            f64 = float(value)
            if not (-1.0 <= f64 <= 1.0):
                raise ValueError(f"{field} must be in [-1,1]")
            object.__setattr__(self, field, round12(f64))
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "left_policy_hash": self.left_policy_hash,
            "right_policy_hash": self.right_policy_hash,
            "strictness_delta": round12(self.strictness_delta),
            "compatibility_delta": round12(self.compatibility_delta),
            "metric_distance": round12(self.metric_distance),
            "sensitivity_score": round12(self.sensitivity_score),
        }

    def stable_hash(self) -> str:
        return self._stable_hash

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class RetroTracePolicySensitivitySummary:
    policy_count: int
    comparison_count: int
    sensitivity_score: float
    classification: str
    _stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy_count, int) or self.policy_count < _MIN_POLICY_COUNT:
            raise ValueError("policy_count must be >= 2")
        if not isinstance(self.comparison_count, int) or self.comparison_count != self.policy_count * (self.policy_count - 1) // 2:
            raise ValueError("comparison_count mismatch")
        object.__setattr__(self, "sensitivity_score", validate_unit_interval(self.sensitivity_score, "sensitivity_score"))
        if self.classification not in (_CLASS_LOW, _CLASS_MODERATE, _CLASS_HIGH):
            raise ValueError("classification must be LOW|MODERATE|HIGH")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "policy_count": self.policy_count,
            "comparison_count": self.comparison_count,
            "sensitivity_score": round12(self.sensitivity_score),
            "classification": self.classification,
        }

    def stable_hash(self) -> str:
        return self._stable_hash

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class RetroTracePolicySensitivityReceipt:
    retro_trace_hash: str
    policy_runs: tuple[RetroTracePolicyRun, ...]
    policy_comparisons: tuple[RetroTracePolicyComparison, ...]
    summary: RetroTracePolicySensitivitySummary
    _stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "retro_trace_hash", validate_sha256_hex(self.retro_trace_hash, "retro_trace_hash"))
        if not isinstance(self.policy_runs, tuple) or len(self.policy_runs) < _MIN_POLICY_COUNT:
            raise ValueError("policy_runs must contain at least two runs")
        if any(not isinstance(run, RetroTracePolicyRun) for run in self.policy_runs):
            raise ValueError("policy_runs contains invalid item")
        if tuple(sorted(self.policy_runs, key=lambda run: (run.policy_hash, run.stable_hash()))) != self.policy_runs:
            raise ValueError("policy_runs must be canonically ordered")
        if any(not isinstance(cmp, RetroTracePolicyComparison) for cmp in self.policy_comparisons):
            raise ValueError("policy_comparisons contains invalid item")
        if tuple(
            sorted(
                self.policy_comparisons,
                key=lambda cmp: (cmp.left_policy_hash, cmp.right_policy_hash, cmp.stable_hash()),
            )
        ) != self.policy_comparisons:
            raise ValueError("policy_comparisons must be canonically ordered")
        expected = len(self.policy_runs) * (len(self.policy_runs) - 1) // 2
        if len(self.policy_comparisons) != expected:
            raise ValueError("policy_comparisons count mismatch")
        policy_hashes = tuple(run.policy_hash for run in self.policy_runs)
        expected_pairs = Counter(tuple(sorted((left, right))) for left, right in combinations(policy_hashes, 2))
        observed_pairs = Counter(tuple(sorted((cmp.left_policy_hash, cmp.right_policy_hash))) for cmp in self.policy_comparisons)
        if observed_pairs != expected_pairs:
            raise ValueError("policy_comparisons must cover each unordered policy pair exactly once")
        if not isinstance(self.summary, RetroTracePolicySensitivitySummary):
            raise ValueError("summary must be RetroTracePolicySensitivitySummary")
        if self.summary.policy_count != len(self.policy_runs):
            raise ValueError("summary policy_count mismatch")
        if self.summary.comparison_count != len(self.policy_comparisons):
            raise ValueError("summary comparison_count mismatch")
        canonical_comparisons = tuple(
            sorted(
                self.policy_comparisons,
                key=lambda cmp: (cmp.left_policy_hash, cmp.right_policy_hash, cmp.stable_hash()),
            )
        )
        computed_sensitivity = _clamp01(
            sum(comparison.sensitivity_score for comparison in canonical_comparisons) / float(len(canonical_comparisons))
        )
        if self.summary.sensitivity_score != computed_sensitivity:
            raise ValueError("summary sensitivity_score mismatch")
        if self.summary.classification != _classify(computed_sensitivity):
            raise ValueError("summary classification mismatch")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "retro_trace_hash": self.retro_trace_hash,
            "policy_runs": tuple(item.to_dict() for item in self.policy_runs),
            "policy_comparisons": tuple(item.to_dict() for item in self.policy_comparisons),
            "summary": self.summary.to_dict(),
        }

    def stable_hash(self) -> str:
        return self._stable_hash

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _run_for_policy(policy: GovernancePolicy, features: dict[str, float]) -> RetroTracePolicyRun:
    strictness = _clamp01(
        (
            policy.min_required_score
            + policy.min_required_confidence
            + policy.min_required_margin
            + policy.min_required_convergence
            + (0.0 if policy.allow_tie_break else 1.0)
            + (0.0 if policy.allow_no_improvement else 1.0)
            + (1.0 if policy.require_stable_transition else 0.0)
        )
        / 7.0
    )

    stability_observed = 1.0 if features["ordering_integrity"] >= 0.5 else 0.0
    compatibility = _clamp01(
        (
            _alignment(features["observability"], policy.min_required_score)
            + _alignment(1.0 - features["sparsity"], policy.min_required_confidence)
            + _alignment(features["ordering_integrity"], policy.min_required_margin)
            + _alignment((features["timing_density"] + features["trace_length"]) / 2.0, policy.min_required_convergence)
            + (1.0 if policy.allow_tie_break else 0.75)
            + (1.0 if policy.allow_no_improvement else 0.75)
            + (1.0 if (not policy.require_stable_transition or stability_observed == 1.0) else 0.0)
        )
        / 7.0
    )

    metrics = (
        ("ordering_integrity", features["ordering_integrity"]),
        ("observability", features["observability"]),
        ("sparsity", features["sparsity"]),
        ("timing_density", features["timing_density"]),
        ("trace_length", features["trace_length"]),
    )
    payload = {
        "policy_hash": policy.stable_hash,
        "strictness": round12(strictness),
        "compatibility": round12(compatibility),
        "metrics": tuple((k, round12(v)) for k, v in metrics),
    }
    return RetroTracePolicyRun(
        policy_hash=policy.stable_hash,
        strictness=strictness,
        compatibility=compatibility,
        metrics=metrics,
        _stable_hash=sha256_hex(payload),
    )


def _comparison(left: RetroTracePolicyRun, right: RetroTracePolicyRun) -> RetroTracePolicyComparison:
    ordered_left, ordered_right = (left, right) if left.policy_hash <= right.policy_hash else (right, left)
    metric_map_left = dict(ordered_left.metrics)
    metric_map_right = dict(ordered_right.metrics)
    metric_distance = _clamp01(
        sum(abs(metric_map_right[key] - metric_map_left[key]) for key in sorted(metric_map_left)) / float(len(metric_map_left))
    )
    strictness_delta = round12(ordered_right.strictness - ordered_left.strictness)
    compatibility_delta = round12(ordered_right.compatibility - ordered_left.compatibility)
    sensitivity = _clamp01(max(abs(strictness_delta), abs(compatibility_delta), metric_distance))
    payload = {
        "left_policy_hash": ordered_left.policy_hash,
        "right_policy_hash": ordered_right.policy_hash,
        "strictness_delta": strictness_delta,
        "compatibility_delta": compatibility_delta,
        "metric_distance": round12(metric_distance),
        "sensitivity_score": round12(sensitivity),
    }
    return RetroTracePolicyComparison(
        left_policy_hash=ordered_left.policy_hash,
        right_policy_hash=ordered_right.policy_hash,
        strictness_delta=strictness_delta,
        compatibility_delta=compatibility_delta,
        metric_distance=metric_distance,
        sensitivity_score=sensitivity,
        _stable_hash=sha256_hex(payload),
    )


def _classify(score: float) -> str:
    if score >= 0.66:
        return _CLASS_HIGH
    if score >= 0.33:
        return _CLASS_MODERATE
    return _CLASS_LOW


def analyze_retro_trace_policy_sensitivity(
    retro_trace: RetroTraceReceipt,
    policies: tuple[GovernancePolicy, ...],
) -> RetroTracePolicySensitivityReceipt:
    if not isinstance(retro_trace, RetroTraceReceipt):
        raise ValueError("retro_trace must be RetroTraceReceipt")
    payload_without_hash = retro_trace.to_dict()
    observed_trace_hash = str(payload_without_hash.pop("stable_hash"))
    if observed_trace_hash != sha256_hex(payload_without_hash):
        raise ValueError("retro_trace stable_hash mismatch")
    if not isinstance(policies, tuple):
        raise ValueError("policies must be tuple[GovernancePolicy, ...]")
    if len(policies) < _MIN_POLICY_COUNT:
        raise ValueError("policies must contain at least 2 policies")
    if any(not isinstance(policy, GovernancePolicy) for policy in policies):
        raise ValueError("policies must be tuple[GovernancePolicy, ...]")
    for idx, policy in enumerate(policies):
        if policy.stable_hash != policy.computed_stable_hash():
            raise ValueError(f"policies[{idx}] stable_hash mismatch")

    features = _trace_features(retro_trace)
    runs = tuple(sorted((_run_for_policy(policy, features) for policy in policies), key=lambda run: (run.policy_hash, run.stable_hash())))

    comparisons = tuple(
        sorted(
            (_comparison(left, right) for left, right in combinations(runs, 2)),
            key=lambda item: (item.left_policy_hash, item.right_policy_hash, item.stable_hash()),
        )
    )
    if comparisons:
        sensitivity = _clamp01(sum(item.sensitivity_score for item in comparisons) / float(len(comparisons)))
    else:
        sensitivity = 0.0
    summary_payload = {
        "policy_count": len(runs),
        "comparison_count": len(comparisons),
        "sensitivity_score": round12(sensitivity),
        "classification": _classify(sensitivity),
    }
    summary = RetroTracePolicySensitivitySummary(
        policy_count=len(runs),
        comparison_count=len(comparisons),
        sensitivity_score=sensitivity,
        classification=_classify(sensitivity),
        _stable_hash=sha256_hex(summary_payload),
    )
    payload = {
        "retro_trace_hash": retro_trace.stable_hash,
        "policy_runs": tuple(item.to_dict() for item in runs),
        "policy_comparisons": tuple(item.to_dict() for item in comparisons),
        "summary": summary.to_dict(),
    }
    return RetroTracePolicySensitivityReceipt(
        retro_trace_hash=retro_trace.stable_hash,
        policy_runs=runs,
        policy_comparisons=comparisons,
        summary=summary,
        _stable_hash=sha256_hex(payload),
    )


__all__ = [
    "RetroTracePolicyRun",
    "RetroTracePolicyComparison",
    "RetroTracePolicySensitivitySummary",
    "RetroTracePolicySensitivityReceipt",
    "analyze_retro_trace_policy_sensitivity",
]
