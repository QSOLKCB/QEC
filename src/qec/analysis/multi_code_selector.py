# SPDX-License-Identifier: MIT
"""v138.9.0 — deterministic additive analysis-layer multi-code selector."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .canonical_hashing import canonical_json, sha256_hex

_ALLOWED_CODE_FAMILIES: frozenset[str] = frozenset(("surface", "qldpc"))


@dataclass(frozen=True)
class CodeCandidateProfile:
    """Bounded candidate metrics for deterministic advisory selection."""

    code_id: str
    code_family: str
    logical_stability: float
    latency_efficiency: float
    overhead_efficiency: float
    hardware_alignment: float
    noise_fit: float
    convergence_confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.code_id, str) or not self.code_id:
            raise ValueError("code_id must be a non-empty str")
        if self.code_family not in _ALLOWED_CODE_FAMILIES:
            allowed_families = ", ".join(sorted(_ALLOWED_CODE_FAMILIES))
            raise ValueError(f"code_family must be one of: {allowed_families}")

        for metric_name in (
            "logical_stability",
            "latency_efficiency",
            "overhead_efficiency",
            "hardware_alignment",
            "noise_fit",
            "convergence_confidence",
        ):
            _validate_unit_interval_float(metric_name, getattr(self, metric_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_id": self.code_id,
            "code_family": self.code_family,
            "logical_stability": self.logical_stability,
            "latency_efficiency": self.latency_efficiency,
            "overhead_efficiency": self.overhead_efficiency,
            "hardware_alignment": self.hardware_alignment,
            "noise_fit": self.noise_fit,
            "convergence_confidence": self.convergence_confidence,
        }


@dataclass(frozen=True)
class CodeSelectionPolicy:
    """Deterministic explicit policy weights and admissibility thresholds."""

    weight_logical_stability: float
    weight_latency_efficiency: float
    weight_overhead_efficiency: float
    weight_hardware_alignment: float
    weight_noise_fit: float
    weight_convergence_confidence: float
    min_noise_fit: float
    min_hardware_alignment: float
    min_convergence_confidence: float

    def __post_init__(self) -> None:
        weight_values = (
            self.weight_logical_stability,
            self.weight_latency_efficiency,
            self.weight_overhead_efficiency,
            self.weight_hardware_alignment,
            self.weight_noise_fit,
            self.weight_convergence_confidence,
        )
        for weight_name, weight_value in zip(self.weight_names(), weight_values):
            if not isinstance(weight_value, (int, float)) or isinstance(weight_value, bool):
                raise ValueError(f"{weight_name} must be a finite float")
            numeric_weight = float(weight_value)
            if not math.isfinite(numeric_weight):
                raise ValueError(f"{weight_name} must be a finite float")
            if numeric_weight <= 0.0:
                raise ValueError(f"{weight_name} must be > 0")

        if sum(float(value) for value in weight_values) <= 0.0:
            raise ValueError("sum of policy weights must be > 0")

        _validate_unit_interval_float("min_noise_fit", self.min_noise_fit)
        _validate_unit_interval_float(
            "min_hardware_alignment",
            self.min_hardware_alignment,
        )
        _validate_unit_interval_float(
            "min_convergence_confidence",
            self.min_convergence_confidence,
        )

    @staticmethod
    def weight_names() -> tuple[str, ...]:
        return (
            "weight_logical_stability",
            "weight_latency_efficiency",
            "weight_overhead_efficiency",
            "weight_hardware_alignment",
            "weight_noise_fit",
            "weight_convergence_confidence",
        )

    def to_weight_map(self) -> dict[str, float]:
        return {
            "logical_stability": float(self.weight_logical_stability),
            "latency_efficiency": float(self.weight_latency_efficiency),
            "overhead_efficiency": float(self.weight_overhead_efficiency),
            "hardware_alignment": float(self.weight_hardware_alignment),
            "noise_fit": float(self.weight_noise_fit),
            "convergence_confidence": float(self.weight_convergence_confidence),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.to_weight_map(),
            "thresholds": {
                "min_noise_fit": float(self.min_noise_fit),
                "min_hardware_alignment": float(self.min_hardware_alignment),
                "min_convergence_confidence": float(self.min_convergence_confidence),
            },
        }


@dataclass(frozen=True)
class CodeSelectionScore:
    """Per-candidate admissibility verdict and deterministic weighted score."""

    code_id: str
    code_family: str
    admissible: bool
    weighted_score: float | None
    logical_stability: float
    hardware_alignment: float
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_id": self.code_id,
            "code_family": self.code_family,
            "admissible": self.admissible,
            "weighted_score": self.weighted_score,
            "logical_stability": self.logical_stability,
            "hardware_alignment": self.hardware_alignment,
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass(frozen=True)
class CodeSelectionReceipt:
    """Deterministic selection receipt with canonical replay identity."""

    selected_code_id: str
    selected_code_family: str
    candidate_scores: tuple[CodeSelectionScore, ...]
    ranking_order: tuple[str, ...]
    policy_snapshot: CodeSelectionPolicy
    replay_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": {
                "code_id": self.selected_code_id,
                "code_family": self.selected_code_family,
            },
            "candidate_scores": tuple(score.to_dict() for score in self.candidate_scores),
            "ranking_order": self.ranking_order,
            "policy_snapshot": self.policy_snapshot.to_dict(),
            "replay_identity": self.replay_identity,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


def select_runtime_code(
    candidates: tuple[CodeCandidateProfile, ...] | list[CodeCandidateProfile],
    policy: CodeSelectionPolicy,
) -> CodeSelectionReceipt:
    """Select the deterministic best admissible code candidate under the policy."""

    if not isinstance(policy, CodeSelectionPolicy):
        raise ValueError("policy must be a CodeSelectionPolicy")

    if not isinstance(candidates, (tuple, list)):
        raise ValueError("candidates must be a non-empty sequence of CodeCandidateProfile")
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidates must be non-empty")

    for candidate in candidate_list:
        if not isinstance(candidate, CodeCandidateProfile):
            raise ValueError("all candidates must be CodeCandidateProfile")

    seen_code_ids: set[str] = set()
    seen_candidate_keys: set[tuple[str, str]] = set()
    for candidate in candidate_list:
        candidate_key = (candidate.code_family, candidate.code_id)
        if candidate.code_id in seen_code_ids:
            raise ValueError(f"duplicate candidate code_id: {candidate.code_id}")
        if candidate_key in seen_candidate_keys:
            raise ValueError(
                "duplicate candidate (code_family, code_id): "
                f"{candidate.code_family}, {candidate.code_id}"
            )
        seen_code_ids.add(candidate.code_id)
        seen_candidate_keys.add(candidate_key)
    weight_map = policy.to_weight_map()
    weight_sum = sum(weight_map.values())

    admissible_scores: list[CodeSelectionScore] = []
    inadmissible_scores: list[CodeSelectionScore] = []

    for candidate in sorted(candidate_list, key=lambda item: (item.code_family, item.code_id)):
        rejection_reasons: list[str] = []
        if candidate.noise_fit < policy.min_noise_fit:
            rejection_reasons.append("noise_fit_below_threshold")
        if candidate.hardware_alignment < policy.min_hardware_alignment:
            rejection_reasons.append("hardware_alignment_below_threshold")
        if candidate.convergence_confidence < policy.min_convergence_confidence:
            rejection_reasons.append("convergence_confidence_below_threshold")

        if rejection_reasons:
            score = CodeSelectionScore(
                code_id=candidate.code_id,
                code_family=candidate.code_family,
                admissible=False,
                weighted_score=None,
                logical_stability=candidate.logical_stability,
                hardware_alignment=candidate.hardware_alignment,
                rejection_reasons=tuple(rejection_reasons),
            )
            inadmissible_scores.append(score)
        else:
            weighted_score = (
                weight_map["logical_stability"] * candidate.logical_stability
                + weight_map["latency_efficiency"] * candidate.latency_efficiency
                + weight_map["overhead_efficiency"] * candidate.overhead_efficiency
                + weight_map["hardware_alignment"] * candidate.hardware_alignment
                + weight_map["noise_fit"] * candidate.noise_fit
                + weight_map["convergence_confidence"] * candidate.convergence_confidence
            ) / weight_sum

            score = CodeSelectionScore(
                code_id=candidate.code_id,
                code_family=candidate.code_family,
                admissible=True,
                weighted_score=float(weighted_score),
                logical_stability=candidate.logical_stability,
                hardware_alignment=candidate.hardware_alignment,
                rejection_reasons=(),
            )
            admissible_scores.append(score)

    if not admissible_scores:
        raise ValueError("no admissible candidate remains after threshold filtering")

    # All admissible scores must have a concrete weighted_score; enforce this invariant.
    assert all(score.weighted_score is not None for score in admissible_scores)

    ranked_admissible = sorted(
        admissible_scores,
        key=lambda score: (
            -float(score.weighted_score),
            -score.logical_stability,
            -score.hardware_alignment,
            score.code_family,
            score.code_id,
        ),
    )
    ranked_inadmissible = sorted(
        inadmissible_scores,
        key=lambda score: (score.code_family, score.code_id),
    )
    ranked_all = tuple(ranked_admissible + ranked_inadmissible)

    selected = ranked_admissible[0]
    ranking_order = tuple(score.code_id for score in ranked_all)

    identity_payload = {
        "selected_code_id": selected.code_id,
        "selected_code_family": selected.code_family,
        "ranking_order": ranking_order,
        "candidate_scores": tuple(score.to_dict() for score in ranked_all),
        "policy_snapshot": policy.to_dict(),
    }
    replay_identity = sha256_hex(identity_payload)

    return CodeSelectionReceipt(
        selected_code_id=selected.code_id,
        selected_code_family=selected.code_family,
        candidate_scores=ranked_all,
        ranking_order=ranking_order,
        policy_snapshot=policy,
        replay_identity=replay_identity,
    )


def _validate_unit_interval_float(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


__all__ = [
    "CodeCandidateProfile",
    "CodeSelectionPolicy",
    "CodeSelectionScore",
    "CodeSelectionReceipt",
    "select_runtime_code",
]
