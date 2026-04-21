"""v138.9.2 — Multi-Code Orchestration Benchmark.

Deterministic analysis-layer benchmark harness for comparing precomputed
single-code and multi-code orchestration candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

SCHEMA_VERSION = 1

ALLOWED_CODE_FAMILIES: tuple[str, ...] = (
    "qldpc",
    "repetition",
    "surface",
    "toric",
)

REQUIRED_BENCHMARK_WEIGHT_KEYS: tuple[str, ...] = (
    "stability_gain",
    "loss_reduction",
    "hardware_gain",
    "efficiency_gain",
    "overhead_penalty",
    "confidence_gain",
)

BOUNDED_CANDIDATE_NUMERIC_FIELDS: tuple[str, ...] = (
    "selection_confidence",
    "migration_confidence",
    "logical_stability",
    "projected_loss",
    "hardware_alignment",
    "execution_efficiency",
    "migration_overhead",
)

BOUNDED_SCORE_FIELDS: tuple[str, ...] = (
    "stability_gain",
    "loss_reduction",
    "hardware_gain",
    "efficiency_gain",
    "overhead_penalty",
    "confidence_gain",
    "orchestration_utility",
)


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("payload keys must be strings")
        return {k: _canonicalize_json(value[k]) for k in sorted(keys)}
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_bounded_number(field_name: str, value: object) -> float:
    if not _is_number(value):
        raise ValueError(f"{field_name} must be a finite numeric value in [0,1]")
    out = float(value)
    if not math.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{field_name} must be a finite numeric value in [0,1]")
    return out


def _validate_non_negative_weight(field_name: str, value: object) -> float:
    if not _is_number(value):
        raise ValueError(f"weight '{field_name}' must be a finite numeric value >= 0")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"weight '{field_name}' must be a finite numeric value >= 0")
    return out


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("bounded metric became non-finite")
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _stable_ranking_key(
    candidate: OrchestrationCandidate,
    utility: float,
) -> tuple[float, float, float, float, str]:
    return (
        -utility,
        -candidate.logical_stability,
        candidate.projected_loss,
        -candidate.hardware_alignment,
        candidate.candidate_id,
    )


@dataclass(frozen=True)
class OrchestrationCandidate:
    candidate_id: str
    source_family: str
    target_family: str
    selection_confidence: float
    migration_confidence: float
    logical_stability: float
    projected_loss: float
    hardware_alignment: float
    execution_efficiency: float
    migration_overhead: float
    orchestration_depth: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if self.source_family not in ALLOWED_CODE_FAMILIES:
            raise ValueError("source_family is not an allowed code family")
        if self.target_family not in ALLOWED_CODE_FAMILIES:
            raise ValueError("target_family is not an allowed code family")
        for field_name in BOUNDED_CANDIDATE_NUMERIC_FIELDS:
            _validate_bounded_number(field_name, getattr(self, field_name))
        if isinstance(self.orchestration_depth, bool) or not isinstance(self.orchestration_depth, int):
            raise ValueError("orchestration_depth must be an int >= 0")
        if self.orchestration_depth < 0:
            raise ValueError("orchestration_depth must be an int >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_family": self.source_family,
            "target_family": self.target_family,
            "selection_confidence": self.selection_confidence,
            "migration_confidence": self.migration_confidence,
            "logical_stability": self.logical_stability,
            "projected_loss": self.projected_loss,
            "hardware_alignment": self.hardware_alignment,
            "execution_efficiency": self.execution_efficiency,
            "migration_overhead": self.migration_overhead,
            "orchestration_depth": self.orchestration_depth,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BenchmarkPolicy:
    minimum_selection_confidence: float
    minimum_migration_confidence: float
    maximum_projected_loss: float
    maximum_migration_overhead: float
    require_cross_family_benefit: bool
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        _validate_bounded_number("minimum_selection_confidence", self.minimum_selection_confidence)
        _validate_bounded_number("minimum_migration_confidence", self.minimum_migration_confidence)
        _validate_bounded_number("maximum_projected_loss", self.maximum_projected_loss)
        _validate_bounded_number("maximum_migration_overhead", self.maximum_migration_overhead)

        if not isinstance(self.require_cross_family_benefit, bool):
            raise ValueError("require_cross_family_benefit must be bool")

        if not isinstance(self.weights, Mapping):
            raise ValueError("weights must be a mapping")

        weight_keys = tuple(self.weights.keys())
        if any(not isinstance(k, str) for k in weight_keys):
            raise ValueError("weight keys must be strings")

        missing = tuple(k for k in REQUIRED_BENCHMARK_WEIGHT_KEYS if k not in self.weights)
        if missing:
            raise ValueError(f"missing required benchmark weight keys: {missing}")

        total_weight = 0.0
        for key in REQUIRED_BENCHMARK_WEIGHT_KEYS:
            total_weight += _validate_non_negative_weight(key, self.weights[key])
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("aggregate required weight must be finite and > 0")

    def to_dict(self) -> dict[str, Any]:
        canonical_weights = {k: float(self.weights[k]) for k in sorted(self.weights.keys())}
        return {
            "minimum_selection_confidence": self.minimum_selection_confidence,
            "minimum_migration_confidence": self.minimum_migration_confidence,
            "maximum_projected_loss": self.maximum_projected_loss,
            "maximum_migration_overhead": self.maximum_migration_overhead,
            "require_cross_family_benefit": self.require_cross_family_benefit,
            "weights": canonical_weights,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BenchmarkScore:
    candidate_id: str
    admissible: bool
    stability_gain: float
    loss_reduction: float
    hardware_gain: float
    efficiency_gain: float
    overhead_penalty: float
    confidence_gain: float
    orchestration_utility: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.admissible, bool):
            raise ValueError("admissible must be bool")
        if not isinstance(self.reasons, tuple) or any(not isinstance(r, str) for r in self.reasons):
            raise ValueError("reasons must be tuple[str, ...]")
        for field_name in BOUNDED_SCORE_FIELDS:
            _validate_bounded_number(field_name, getattr(self, field_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "admissible": self.admissible,
            "stability_gain": self.stability_gain,
            "loss_reduction": self.loss_reduction,
            "hardware_gain": self.hardware_gain,
            "efficiency_gain": self.efficiency_gain,
            "overhead_penalty": self.overhead_penalty,
            "confidence_gain": self.confidence_gain,
            "orchestration_utility": self.orchestration_utility,
            "reasons": self.reasons,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BenchmarkComparison:
    best_candidate_id: str
    best_utility: float
    baseline_utility: float
    improvement_margin: float
    cross_family_winner: bool
    deterministic_ranking: tuple[str, ...]
    best_overall_candidate_id: str
    best_overall_utility: float

    def __post_init__(self) -> None:
        if not isinstance(self.best_candidate_id, str) or not self.best_candidate_id:
            raise ValueError("best_candidate_id must be non-empty string")
        if not isinstance(self.best_overall_candidate_id, str) or not self.best_overall_candidate_id:
            raise ValueError("best_overall_candidate_id must be non-empty string")
        if not isinstance(self.cross_family_winner, bool):
            raise ValueError("cross_family_winner must be bool")
        if not isinstance(self.deterministic_ranking, tuple) or any(
            not isinstance(c, str) for c in self.deterministic_ranking
        ):
            raise ValueError("deterministic_ranking must be tuple[str, ...]")
        _validate_bounded_number("best_utility", self.best_utility)
        _validate_bounded_number("baseline_utility", self.baseline_utility)
        _validate_bounded_number("improvement_margin", self.improvement_margin)
        _validate_bounded_number("best_overall_utility", self.best_overall_utility)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_candidate_id": self.best_candidate_id,
            "best_utility": self.best_utility,
            "baseline_utility": self.baseline_utility,
            "improvement_margin": self.improvement_margin,
            "cross_family_winner": self.cross_family_winner,
            "deterministic_ranking": self.deterministic_ranking,
            "best_overall_candidate_id": self.best_overall_candidate_id,
            "best_overall_utility": self.best_overall_utility,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class OrchestrationBenchmarkReceipt:
    benchmark_scores: tuple[BenchmarkScore, ...]
    comparison: BenchmarkComparison
    policy_snapshot: BenchmarkPolicy
    replay_identity: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_scores, tuple) or any(
            not isinstance(s, BenchmarkScore) for s in self.benchmark_scores
        ):
            raise ValueError("benchmark_scores must be tuple[BenchmarkScore, ...]")
        if not isinstance(self.comparison, BenchmarkComparison):
            raise ValueError("comparison must be BenchmarkComparison")
        if not isinstance(self.policy_snapshot, BenchmarkPolicy):
            raise ValueError("policy_snapshot must be BenchmarkPolicy")
        if not isinstance(self.replay_identity, str) or not self.replay_identity:
            raise ValueError("replay_identity must be non-empty string")
        if not isinstance(self.stable_hash, str) or not self.stable_hash:
            raise ValueError("stable_hash must be non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_scores": tuple(score.to_dict() for score in self.benchmark_scores),
            "comparison": self.comparison.to_dict(),
            "policy_snapshot": self.policy_snapshot.to_dict(),
            "replay_identity": self.replay_identity,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_dict())


def _evaluate_admissibility(candidate: OrchestrationCandidate, policy: BenchmarkPolicy) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if candidate.selection_confidence < policy.minimum_selection_confidence:
        reasons.append("selection_confidence_below_minimum")
    if candidate.migration_confidence < policy.minimum_migration_confidence:
        reasons.append("migration_confidence_below_minimum")
    if candidate.projected_loss > policy.maximum_projected_loss:
        reasons.append("projected_loss_above_maximum")
    if candidate.migration_overhead > policy.maximum_migration_overhead:
        reasons.append("migration_overhead_above_maximum")
    return (len(reasons) == 0, tuple(reasons))


def _score_candidate(candidate: OrchestrationCandidate, policy: BenchmarkPolicy) -> BenchmarkScore:
    admissible, reasons = _evaluate_admissibility(candidate, policy)
    confidence_gain = _clamp01((candidate.selection_confidence + candidate.migration_confidence) / 2.0)
    signals = {
        "stability_gain": _clamp01(candidate.logical_stability),
        "loss_reduction": _clamp01(1.0 - candidate.projected_loss),
        "hardware_gain": _clamp01(candidate.hardware_alignment),
        "efficiency_gain": _clamp01(candidate.execution_efficiency),
        "overhead_penalty": _clamp01(1.0 - candidate.migration_overhead),
        "confidence_gain": confidence_gain,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for key in REQUIRED_BENCHMARK_WEIGHT_KEYS:
        weight = float(policy.weights[key])
        weighted_sum += signals[key] * weight
        total_weight += weight
    orchestration_utility = _clamp01(weighted_sum / total_weight)

    return BenchmarkScore(
        candidate_id=candidate.candidate_id,
        admissible=admissible,
        stability_gain=signals["stability_gain"],
        loss_reduction=signals["loss_reduction"],
        hardware_gain=signals["hardware_gain"],
        efficiency_gain=signals["efficiency_gain"],
        overhead_penalty=signals["overhead_penalty"],
        confidence_gain=signals["confidence_gain"],
        orchestration_utility=orchestration_utility,
        reasons=reasons,
    )


def _rank_candidates(
    candidates: tuple[OrchestrationCandidate, ...],
    scores_by_id: Mapping[str, BenchmarkScore],
) -> tuple[str, ...]:
    ranked = sorted(
        candidates,
        key=lambda c: _stable_ranking_key(c, scores_by_id[c.candidate_id].orchestration_utility),
    )
    return tuple(c.candidate_id for c in ranked)


def _select_baseline(
    candidates_by_id: Mapping[str, OrchestrationCandidate],
    scores_by_id: Mapping[str, BenchmarkScore],
) -> str:
    baseline_candidates = tuple(
        c
        for c in candidates_by_id.values()
        if c.source_family == c.target_family and c.orchestration_depth == 0
    )
    if not baseline_candidates:
        raise ValueError("at least one baseline candidate is required")

    baseline_ranked = sorted(
        baseline_candidates,
        key=lambda c: _stable_ranking_key(c, scores_by_id[c.candidate_id].orchestration_utility),
    )
    return baseline_ranked[0].candidate_id


def _build_comparison(
    candidates_by_id: Mapping[str, OrchestrationCandidate],
    scores_by_id: Mapping[str, BenchmarkScore],
    policy: BenchmarkPolicy,
    deterministic_ranking: tuple[str, ...],
) -> BenchmarkComparison:
    baseline_id = _select_baseline(candidates_by_id, scores_by_id)
    baseline_utility = scores_by_id[baseline_id].orchestration_utility

    admissible_ids = tuple(cid for cid in deterministic_ranking if scores_by_id[cid].admissible)
    if not admissible_ids:
        raise ValueError("at least one admissible candidate is required")

    best_overall_id = deterministic_ranking[0]
    best_admissible_id = admissible_ids[0]
    best_admissible_utility = scores_by_id[best_admissible_id].orchestration_utility

    best_admissible_candidate = candidates_by_id[best_admissible_id]
    is_cross_family_best = best_admissible_candidate.source_family != best_admissible_candidate.target_family
    cross_family_winner = is_cross_family_best or not policy.require_cross_family_benefit

    improvement_margin = _clamp01(max(0.0, best_admissible_utility - baseline_utility))

    return BenchmarkComparison(
        best_candidate_id=best_admissible_id,
        best_utility=best_admissible_utility,
        baseline_utility=baseline_utility,
        improvement_margin=improvement_margin,
        cross_family_winner=cross_family_winner,
        deterministic_ranking=deterministic_ranking,
        best_overall_candidate_id=best_overall_id,
        best_overall_utility=scores_by_id[best_overall_id].orchestration_utility,
    )


def _validate_candidate_list(candidates: tuple[OrchestrationCandidate, ...]) -> None:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, OrchestrationCandidate):
            raise ValueError("all candidates must be OrchestrationCandidate instances")
        if candidate.candidate_id in seen:
            raise ValueError("duplicate candidate_id is not allowed")
        seen.add(candidate.candidate_id)


def _build_receipt(
    scores: tuple[BenchmarkScore, ...],
    comparison: BenchmarkComparison,
    policy: BenchmarkPolicy,
) -> OrchestrationBenchmarkReceipt:
    payload_without_hash = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_scores": tuple(score.to_dict() for score in scores),
        "comparison": comparison.to_dict(),
        "policy_snapshot": policy.to_dict(),
    }
    replay_identity = _sha256_hex(payload_without_hash)
    payload_with_replay = {
        **payload_without_hash,
        "replay_identity": replay_identity,
    }
    stable_hash = _sha256_hex(payload_with_replay)

    return OrchestrationBenchmarkReceipt(
        benchmark_scores=scores,
        comparison=comparison,
        policy_snapshot=policy,
        replay_identity=replay_identity,
        stable_hash=stable_hash,
    )


def benchmark_multi_code_orchestration(
    candidates: tuple[OrchestrationCandidate, ...] | list[OrchestrationCandidate],
    policy: BenchmarkPolicy,
) -> OrchestrationBenchmarkReceipt:
    if not isinstance(policy, BenchmarkPolicy):
        raise ValueError("policy must be BenchmarkPolicy")
    validated_candidates = tuple(candidates)
    _validate_candidate_list(validated_candidates)

    scores_by_id: dict[str, BenchmarkScore] = {}
    for candidate in validated_candidates:
        scores_by_id[candidate.candidate_id] = _score_candidate(candidate, policy)

    deterministic_ranking = _rank_candidates(validated_candidates, scores_by_id)
    comparison = _build_comparison(
        {c.candidate_id: c for c in validated_candidates},
        scores_by_id,
        policy,
        deterministic_ranking,
    )
    ordered_scores = tuple(scores_by_id[cid] for cid in deterministic_ranking)
    return _build_receipt(ordered_scores, comparison, policy)
