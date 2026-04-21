# SPDX-License-Identifier: MIT
"""v138.7.1 — early termination via dark-state proofs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import types
from typing import Any

from qec.analysis.canonical_hashing import (
    CanonicalHashingError,
    canonical_bytes,
    canonical_json,
    canonicalize_json,
    sha256_hex,
)

RELEASE_VERSION = "v138.7.1"
RUNTIME_KIND = "early_termination_via_dark_state_proofs"

DECISION_LABELS = (
    "terminate_early",
    "continue_iteration",
    "insufficient_proof",
    "ambiguous_state",
)

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | Mapping[str, "_JSONValue"]


class EarlyTerminationDarkStateProofError(ValueError):
    """Raised when early-termination proof inputs or invariants are invalid."""


def _canonicalize_json(value: Any) -> _JSONValue:
    try:
        return canonicalize_json(value)
    except CanonicalHashingError as exc:
        raise EarlyTerminationDarkStateProofError(str(exc)) from exc


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json(value)
    except CanonicalHashingError as exc:
        raise EarlyTerminationDarkStateProofError(str(exc)) from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return canonical_bytes(value)
    except CanonicalHashingError as exc:
        raise EarlyTerminationDarkStateProofError(str(exc)) from exc


def _sha256_hex(value: Any) -> str:
    try:
        return sha256_hex(value)
    except CanonicalHashingError as exc:
        raise EarlyTerminationDarkStateProofError(str(exc)) from exc


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _round_bounded(value: float, digits: int) -> float:
    return round(_clamp01(value), digits)


def _validate_finite_number(
    value: Any,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise EarlyTerminationDarkStateProofError(f"{field_name} must not be a bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EarlyTerminationDarkStateProofError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise EarlyTerminationDarkStateProofError(f"{field_name} must be finite")
    if minimum is not None and number < minimum:
        raise EarlyTerminationDarkStateProofError(f"{field_name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise EarlyTerminationDarkStateProofError(f"{field_name} must be <= {maximum}")
    return number


def _validate_int(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise EarlyTerminationDarkStateProofError(f"{field_name} must not be a bool")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise EarlyTerminationDarkStateProofError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EarlyTerminationDarkStateProofError(f"{field_name} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise EarlyTerminationDarkStateProofError(f"{field_name} must be >= {minimum}")
    return parsed


def _validate_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise EarlyTerminationDarkStateProofError(f"{field_name} must be a bool")
    return value


def _immutable_mapping(mapping: Mapping[str, Any]) -> Mapping[str, _JSONValue]:
    canonical = _canonicalize_json(mapping)
    if not isinstance(canonical, dict):
        raise EarlyTerminationDarkStateProofError("mapping must serialize as an object")
    return types.MappingProxyType(canonical)


def _hex_or_none(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    token = str(value)
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise EarlyTerminationDarkStateProofError(f"{field_name} must be a lowercase 64-char sha256 hex")
    return token


@dataclass(frozen=True)
class EarlyTerminationDarkStateConfig:
    minimum_top_proposal_confidence: float
    minimum_top_proposal_score: float
    maximum_convergence_delta: float
    minimum_dark_state_score: float
    minimum_dark_state_coverage: float
    minimum_proof_consistency_score: float
    require_convergence: bool
    require_nonempty_proposals: bool
    allow_termination_without_dark_state_proof: bool
    decision_round_digits: int
    normalization_policy: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "minimum_top_proposal_confidence": self.minimum_top_proposal_confidence,
            "minimum_top_proposal_score": self.minimum_top_proposal_score,
            "maximum_convergence_delta": self.maximum_convergence_delta,
            "minimum_dark_state_score": self.minimum_dark_state_score,
            "minimum_dark_state_coverage": self.minimum_dark_state_coverage,
            "minimum_proof_consistency_score": self.minimum_proof_consistency_score,
            "require_convergence": self.require_convergence,
            "require_nonempty_proposals": self.require_nonempty_proposals,
            "allow_termination_without_dark_state_proof": self.allow_termination_without_dark_state_proof,
            "decision_round_digits": self.decision_round_digits,
            "normalization_policy": self.normalization_policy,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class DarkStateProofSignal:
    dark_state_score: float
    dark_state_coverage: float
    proposal_dominance_score: float
    convergence_confidence: float
    proof_consistency_score: float
    termination_confidence: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "dark_state_score": self.dark_state_score,
            "dark_state_coverage": self.dark_state_coverage,
            "proposal_dominance_score": self.proposal_dominance_score,
            "convergence_confidence": self.convergence_confidence,
            "proof_consistency_score": self.proof_consistency_score,
            "termination_confidence": self.termination_confidence,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class EarlyTerminationCondition:
    condition_name: str
    satisfied: bool
    observed_value: float
    threshold_value: float
    comparator: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "condition_name": self.condition_name,
            "satisfied": self.satisfied,
            "observed_value": self.observed_value,
            "threshold_value": self.threshold_value,
            "comparator": self.comparator,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class EarlyTerminationDecision:
    decision_label: str
    terminate_early: bool
    rationale: str
    conditions: tuple[EarlyTerminationCondition, ...]
    proof_signals: DarkStateProofSignal
    decision_hash: str
    replay_identity: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "decision_label": self.decision_label,
            "terminate_early": self.terminate_early,
            "rationale": self.rationale,
            "conditions": tuple(condition.to_dict() for condition in self.conditions),
            "proof_signals": self.proof_signals.to_dict(),
            "decision_hash": self.decision_hash,
            "replay_identity": self.replay_identity,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("decision_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())

    def __post_init__(self) -> None:
        if self.decision_hash != self.stable_hash():
            raise EarlyTerminationDarkStateProofError("decision_hash must match stable_hash payload")


@dataclass(frozen=True)
class EarlyTerminationReceipt:
    release_version: str
    runtime_kind: str
    config_hash: str
    kernel_result_hash: str
    kernel_replay_identity: str
    proof_input_hash: str | None
    termination_decision_hash: str
    termination_replay_identity: str
    top_proposal_hash: str | None
    decision_label: str
    terminate_early: bool
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "config_hash": self.config_hash,
            "kernel_result_hash": self.kernel_result_hash,
            "kernel_replay_identity": self.kernel_replay_identity,
            "proof_input_hash": self.proof_input_hash,
            "termination_decision_hash": self.termination_decision_hash,
            "termination_replay_identity": self.termination_replay_identity,
            "top_proposal_hash": self.top_proposal_hash,
            "decision_label": self.decision_label,
            "terminate_early": self.terminate_early,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())

    def __post_init__(self) -> None:
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise EarlyTerminationDarkStateProofError("receipt_hash must match stable_hash payload")


@dataclass(frozen=True)
class EarlyTerminationAnalysisResult:
    release_version: str
    runtime_kind: str
    config: EarlyTerminationDarkStateConfig
    kernel_result: Mapping[str, _JSONValue]
    dark_state_inputs: Mapping[str, _JSONValue] | None
    top_proposal: Mapping[str, _JSONValue] | None
    proof_signals: DarkStateProofSignal
    decision: EarlyTerminationDecision
    receipt: EarlyTerminationReceipt
    advisory_only: bool
    decoder_core_modified: bool
    result_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "config": self.config.to_dict(),
            "kernel_result": _canonicalize_json(self.kernel_result),
            "dark_state_inputs": _canonicalize_json(self.dark_state_inputs),
            "top_proposal": _canonicalize_json(self.top_proposal),
            "proof_signals": self.proof_signals.to_dict(),
            "decision": self.decision.to_dict(),
            "receipt": self.receipt.to_dict(),
            "advisory_only": self.advisory_only,
            "decoder_core_modified": self.decoder_core_modified,
            "result_hash": self.result_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("result_hash")
        payload["decision"] = self.decision.to_hash_payload_dict()
        payload["receipt"] = self.receipt.to_hash_payload_dict()
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def normalize_early_termination_config(
    config: EarlyTerminationDarkStateConfig | Mapping[str, Any],
) -> EarlyTerminationDarkStateConfig:
    raw = config.to_dict() if isinstance(config, EarlyTerminationDarkStateConfig) else config
    if not isinstance(raw, Mapping):
        raise EarlyTerminationDarkStateProofError("config must be EarlyTerminationDarkStateConfig or mapping")

    normalized = EarlyTerminationDarkStateConfig(
        minimum_top_proposal_confidence=_validate_finite_number(
            raw.get("minimum_top_proposal_confidence", 0.8),
            field_name="minimum_top_proposal_confidence",
            minimum=0.0,
            maximum=1.0,
        ),
        minimum_top_proposal_score=_validate_finite_number(
            raw.get("minimum_top_proposal_score", 0.8),
            field_name="minimum_top_proposal_score",
            minimum=0.0,
            maximum=1.0,
        ),
        maximum_convergence_delta=_validate_finite_number(
            raw.get("maximum_convergence_delta", 0.1),
            field_name="maximum_convergence_delta",
            minimum=0.0,
            maximum=1.0,
        ),
        minimum_dark_state_score=_validate_finite_number(
            raw.get("minimum_dark_state_score", 0.8),
            field_name="minimum_dark_state_score",
            minimum=0.0,
            maximum=1.0,
        ),
        minimum_dark_state_coverage=_validate_finite_number(
            raw.get("minimum_dark_state_coverage", 0.8),
            field_name="minimum_dark_state_coverage",
            minimum=0.0,
            maximum=1.0,
        ),
        minimum_proof_consistency_score=_validate_finite_number(
            raw.get("minimum_proof_consistency_score", 0.8),
            field_name="minimum_proof_consistency_score",
            minimum=0.0,
            maximum=1.0,
        ),
        require_convergence=_validate_bool(raw.get("require_convergence", True), field_name="require_convergence"),
        require_nonempty_proposals=_validate_bool(
            raw.get("require_nonempty_proposals", True), field_name="require_nonempty_proposals"
        ),
        allow_termination_without_dark_state_proof=_validate_bool(
            raw.get("allow_termination_without_dark_state_proof", False),
            field_name="allow_termination_without_dark_state_proof",
        ),
        decision_round_digits=_validate_int(raw.get("decision_round_digits", 8), field_name="decision_round_digits", minimum=0),
        normalization_policy=str(raw.get("normalization_policy", "clamp_0_1")),
    )
    if normalized.normalization_policy != "clamp_0_1":
        raise EarlyTerminationDarkStateProofError("normalization_policy must be 'clamp_0_1'")
    return normalized


def normalize_kernel_result(kernel_result: Any) -> Mapping[str, _JSONValue]:
    raw = kernel_result.to_dict() if hasattr(kernel_result, "to_dict") and callable(kernel_result.to_dict) else kernel_result
    if not isinstance(raw, Mapping):
        raise EarlyTerminationDarkStateProofError("kernel_result must be mapping or result-like object")

    required = {
        "release_version",
        "runtime_kind",
        "proposals",
        "converged",
        "convergence_delta",
        "result_hash",
        "replay_identity",
    }
    missing = sorted(required.difference(raw.keys()))
    if missing:
        raise EarlyTerminationDarkStateProofError(f"malformed kernel_result: missing keys {missing}")

    proposals_raw = raw.get("proposals")
    if not isinstance(proposals_raw, (list, tuple)):
        raise EarlyTerminationDarkStateProofError("kernel_result.proposals must be a sequence")

    normalized_proposals: list[Mapping[str, _JSONValue]] = []
    prev_key: tuple[float, float, str, str] | None = None
    for idx, proposal_raw in enumerate(proposals_raw):
        if hasattr(proposal_raw, "to_dict") and callable(proposal_raw.to_dict):
            proposal_raw = proposal_raw.to_dict()
        if not isinstance(proposal_raw, Mapping):
            raise EarlyTerminationDarkStateProofError("kernel_result.proposals entries must be mapping-like")
        proposal_id = str(proposal_raw.get("proposal_id", ""))
        if not proposal_id:
            raise EarlyTerminationDarkStateProofError("proposal_id must be non-empty")
        score = _validate_finite_number(
            proposal_raw.get("proposal_score"), field_name=f"proposal[{idx}].proposal_score", minimum=0.0, maximum=1.0
        )
        confidence = _validate_finite_number(
            proposal_raw.get("confidence"), field_name=f"proposal[{idx}].confidence", minimum=0.0, maximum=1.0
        )
        target_nodes = proposal_raw.get("target_nodes", ())
        if not isinstance(target_nodes, (list, tuple)):
            raise EarlyTerminationDarkStateProofError("proposal.target_nodes must be a sequence")
        target_edges = proposal_raw.get("target_edges", ())
        if not isinstance(target_edges, (list, tuple)):
            raise EarlyTerminationDarkStateProofError("proposal.target_edges must be a sequence")
        canonical_proposal = _canonicalize_json(
            {
                "proposal_id": proposal_id,
                "target_nodes": tuple(str(node) for node in target_nodes),
                "target_edges": tuple(str(edge) for edge in target_edges),
                "action_class": str(proposal_raw.get("action_class", "")),
                "proposal_score": score,
                "confidence": confidence,
                "rationale": proposal_raw.get("rationale", {}),
            }
        )
        if not isinstance(canonical_proposal, dict):
            raise EarlyTerminationDarkStateProofError("proposal payload must serialize as object")
        sort_key = (-score, -confidence, tuple(canonical_proposal["target_nodes"]), proposal_id)
        if prev_key is not None and sort_key < prev_key:
            raise EarlyTerminationDarkStateProofError("kernel_result.proposals must preserve canonical ranking order")
        prev_key = sort_key
        normalized_proposals.append(types.MappingProxyType(canonical_proposal))

    converged_raw = raw.get("converged")
    if not isinstance(converged_raw, bool):
        raise EarlyTerminationDarkStateProofError("kernel_result.converged must be a bool")

    convergence_delta = _validate_finite_number(
        raw.get("convergence_delta"), field_name="kernel_result.convergence_delta", minimum=0.0, maximum=1.0
    )
    result_hash = _hex_or_none(raw.get("result_hash"), field_name="kernel_result.result_hash")
    replay_identity = _hex_or_none(raw.get("replay_identity"), field_name="kernel_result.replay_identity")
    if result_hash is None or replay_identity is None:
        raise EarlyTerminationDarkStateProofError("kernel_result.result_hash and replay_identity are required")

    normalized = {
        "release_version": str(raw.get("release_version")),
        "runtime_kind": str(raw.get("runtime_kind")),
        "proposals": tuple(normalized_proposals),
        "converged": converged_raw,
        "convergence_delta": convergence_delta,
        "result_hash": result_hash,
        "replay_identity": replay_identity,
        "config_hash": _hex_or_none(raw.get("config_hash"), field_name="kernel_result.config_hash"),
        "input_hash": _hex_or_none(raw.get("input_hash"), field_name="kernel_result.input_hash"),
    }
    computed_result_hash = _sha256_hex(
        {
            "release_version": normalized["release_version"],
            "runtime_kind": normalized["runtime_kind"],
            "proposals": normalized["proposals"],
            "converged": normalized["converged"],
            "convergence_delta": normalized["convergence_delta"],
            "config_hash": normalized["config_hash"],
            "input_hash": normalized["input_hash"],
        }
    )
    if normalized["result_hash"] != computed_result_hash:
        raise EarlyTerminationDarkStateProofError(
            "kernel_result.result_hash must match normalized kernel payload used for early-termination analysis"
        )
    computed_replay_identity = _sha256_hex(
        {"result_hash": computed_result_hash, "input_hash": normalized["input_hash"]}
    )
    if normalized["replay_identity"] != computed_replay_identity:
        raise EarlyTerminationDarkStateProofError(
            "kernel_result.replay_identity must match normalized kernel payload lineage"
        )
    return types.MappingProxyType(normalized)


def normalize_dark_state_inputs(dark_state_inputs: Any) -> Mapping[str, _JSONValue] | None:
    if dark_state_inputs is None:
        return None
    raw = dark_state_inputs.to_dict() if hasattr(dark_state_inputs, "to_dict") and callable(dark_state_inputs.to_dict) else dark_state_inputs
    if not isinstance(raw, Mapping):
        raise EarlyTerminationDarkStateProofError("dark_state_inputs must be mapping-like")

    for field in (
        "dark_state_score",
        "dark_state_coverage",
        "proof_consistency_score",
        "skip_safety_score",
        "stability_score",
        "coverage_score",
    ):
        if field in raw:
            _validate_finite_number(raw[field], field_name=field, minimum=0.0, maximum=1.0)

    normalized = _canonicalize_json(dict(raw))
    if not isinstance(normalized, dict):
        raise EarlyTerminationDarkStateProofError("dark_state_inputs must serialize as object")
    if "stable_hash" in normalized:
        _hex_or_none(normalized.get("stable_hash"), field_name="dark_state_inputs.stable_hash")
    if "replay_identity" in normalized:
        _hex_or_none(normalized.get("replay_identity"), field_name="dark_state_inputs.replay_identity")
    return types.MappingProxyType(normalized)


def compute_dark_state_proof_signals(
    *,
    config: EarlyTerminationDarkStateConfig,
    kernel_result: Mapping[str, _JSONValue],
    dark_state_inputs: Mapping[str, _JSONValue] | None,
) -> DarkStateProofSignal:
    proposals = kernel_result["proposals"]
    if not isinstance(proposals, tuple):
        raise EarlyTerminationDarkStateProofError("normalized kernel proposals must be a tuple")

    top_score = 0.0
    top_confidence = 0.0
    second_score = 0.0
    if proposals:
        top = proposals[0]
        if not isinstance(top, Mapping):
            raise EarlyTerminationDarkStateProofError("top proposal must be mapping")
        top_score = float(top["proposal_score"])
        top_confidence = float(top["confidence"])
        if len(proposals) > 1 and isinstance(proposals[1], Mapping):
            second_score = float(proposals[1]["proposal_score"])

    convergence_delta = float(kernel_result["convergence_delta"])
    converged = bool(kernel_result["converged"])

    dark_state_score = 0.0
    dark_state_coverage = 0.0
    proof_consistency = 0.0
    if dark_state_inputs is not None:
        dark_state_score = float(dark_state_inputs.get("dark_state_score", dark_state_inputs.get("skip_safety_score", 0.0)))
        dark_state_coverage = float(dark_state_inputs.get("dark_state_coverage", dark_state_inputs.get("coverage_score", 0.0)))
        proof_consistency = float(dark_state_inputs.get("proof_consistency_score", dark_state_inputs.get("stability_score", 0.0)))

    proposal_dominance = _clamp01(max(0.0, top_score - second_score) + (0.5 * top_confidence))
    convergence_confidence = _clamp01(0.5 * (1.0 if converged else 0.0) + 0.5 * (1.0 - convergence_delta))
    termination_confidence = _clamp01(
        0.30 * top_confidence
        + 0.20 * top_score
        + 0.15 * proposal_dominance
        + 0.15 * convergence_confidence
        + 0.10 * dark_state_score
        + 0.10 * proof_consistency
    )

    digits = config.decision_round_digits
    return DarkStateProofSignal(
        dark_state_score=_round_bounded(dark_state_score, digits),
        dark_state_coverage=_round_bounded(dark_state_coverage, digits),
        proposal_dominance_score=_round_bounded(proposal_dominance, digits),
        convergence_confidence=_round_bounded(convergence_confidence, digits),
        proof_consistency_score=_round_bounded(proof_consistency, digits),
        termination_confidence=_round_bounded(termination_confidence, digits),
    )


def classify_early_termination_decision(
    *,
    config: EarlyTerminationDarkStateConfig,
    kernel_result: Mapping[str, _JSONValue],
    dark_state_inputs: Mapping[str, _JSONValue] | None,
    proof_signals: DarkStateProofSignal,
) -> EarlyTerminationDecision:
    proposals = kernel_result["proposals"]
    if not isinstance(proposals, tuple):
        raise EarlyTerminationDarkStateProofError("normalized kernel proposals must be tuple")

    has_proposals = len(proposals) > 0
    top_score = float(proposals[0]["proposal_score"]) if has_proposals and isinstance(proposals[0], Mapping) else 0.0
    top_confidence = float(proposals[0]["confidence"]) if has_proposals and isinstance(proposals[0], Mapping) else 0.0
    converged = bool(kernel_result["converged"])
    convergence_delta = float(kernel_result["convergence_delta"])

    conditions = (
        EarlyTerminationCondition(
            condition_name="nonempty_proposals",
            satisfied=has_proposals,
            observed_value=1.0 if has_proposals else 0.0,
            threshold_value=1.0,
            comparator=">=",
        ),
        EarlyTerminationCondition(
            condition_name="top_proposal_confidence",
            satisfied=top_confidence >= config.minimum_top_proposal_confidence,
            observed_value=round(top_confidence, config.decision_round_digits),
            threshold_value=config.minimum_top_proposal_confidence,
            comparator=">=",
        ),
        EarlyTerminationCondition(
            condition_name="top_proposal_score",
            satisfied=top_score >= config.minimum_top_proposal_score,
            observed_value=round(top_score, config.decision_round_digits),
            threshold_value=config.minimum_top_proposal_score,
            comparator=">=",
        ),
        EarlyTerminationCondition(
            condition_name="convergence_delta",
            satisfied=convergence_delta <= config.maximum_convergence_delta,
            observed_value=round(convergence_delta, config.decision_round_digits),
            threshold_value=config.maximum_convergence_delta,
            comparator="<=",
        ),
        EarlyTerminationCondition(
            condition_name="dark_state_score",
            satisfied=proof_signals.dark_state_score >= config.minimum_dark_state_score,
            observed_value=proof_signals.dark_state_score,
            threshold_value=config.minimum_dark_state_score,
            comparator=">=",
        ),
        EarlyTerminationCondition(
            condition_name="dark_state_coverage",
            satisfied=proof_signals.dark_state_coverage >= config.minimum_dark_state_coverage,
            observed_value=proof_signals.dark_state_coverage,
            threshold_value=config.minimum_dark_state_coverage,
            comparator=">=",
        ),
        EarlyTerminationCondition(
            condition_name="proof_consistency_score",
            satisfied=proof_signals.proof_consistency_score >= config.minimum_proof_consistency_score,
            observed_value=proof_signals.proof_consistency_score,
            threshold_value=config.minimum_proof_consistency_score,
            comparator=">=",
        ),
    )

    missing_proof = dark_state_inputs is None
    proposal_weak = (not has_proposals and config.require_nonempty_proposals) or (
        has_proposals
        and (top_confidence < config.minimum_top_proposal_confidence or top_score < config.minimum_top_proposal_score)
    )
    convergence_missing_or_weak = config.require_convergence and (not converged or convergence_delta > config.maximum_convergence_delta)

    if missing_proof and not config.allow_termination_without_dark_state_proof:
        label = "insufficient_proof"
        terminate_early = False
        rationale = "dark-state proof is required by policy but not supplied"
    else:
        proof_supplied = dark_state_inputs is not None
        proof_weak = proof_supplied and (
            proof_signals.dark_state_score < config.minimum_dark_state_score
            or proof_signals.dark_state_coverage < config.minimum_dark_state_coverage
            or proof_signals.proof_consistency_score < config.minimum_proof_consistency_score
        )
        if (
            has_proposals
            and top_confidence >= config.minimum_top_proposal_confidence
            and top_score >= config.minimum_top_proposal_score
            and proof_signals.termination_confidence >= 0.5
            and proof_weak
        ):
            label = "ambiguous_state"
            terminate_early = False
            rationale = "proposal signals are strong but dark-state proof evidence conflicts"
        elif proposal_weak or convergence_missing_or_weak or proof_weak:
            label = "continue_iteration"
            terminate_early = False
            rationale = "threshold conditions for early termination are not yet satisfied"
        else:
            label = "terminate_early"
            terminate_early = True
            rationale = "all required proposal, convergence, and dark-state proof thresholds were met"

    replay_identity = _sha256_hex(
        {
            "kernel_replay_identity": kernel_result["replay_identity"],
            "kernel_result_hash": kernel_result["result_hash"],
            "proof_input_hash": _sha256_hex(dark_state_inputs) if dark_state_inputs is not None else None,
            "decision_label": label,
        }
    )
    decision_hash = _sha256_hex(
        {
            "decision_label": label,
            "terminate_early": terminate_early,
            "rationale": rationale,
            "conditions": tuple(condition.to_dict() for condition in conditions),
            "proof_signals": proof_signals.to_dict(),
            "replay_identity": replay_identity,
        }
    )
    return EarlyTerminationDecision(
        decision_label=label,
        terminate_early=terminate_early,
        rationale=rationale,
        conditions=conditions,
        proof_signals=proof_signals,
        decision_hash=decision_hash,
        replay_identity=replay_identity,
    )


def build_early_termination_analysis_result(
    *,
    config: EarlyTerminationDarkStateConfig | Mapping[str, Any],
    kernel_result: Any,
    dark_state_inputs: Any = None,
) -> EarlyTerminationAnalysisResult:
    cfg = normalize_early_termination_config(config)
    normalized_kernel = normalize_kernel_result(kernel_result)
    normalized_proof = normalize_dark_state_inputs(dark_state_inputs)

    proof_signals = compute_dark_state_proof_signals(
        config=cfg,
        kernel_result=normalized_kernel,
        dark_state_inputs=normalized_proof,
    )
    decision = classify_early_termination_decision(
        config=cfg,
        kernel_result=normalized_kernel,
        dark_state_inputs=normalized_proof,
        proof_signals=proof_signals,
    )

    proposals = normalized_kernel["proposals"]
    top_proposal = proposals[0] if isinstance(proposals, tuple) and proposals else None
    top_proposal_hash = _sha256_hex(top_proposal) if top_proposal is not None else None

    proof_input_hash = _sha256_hex(normalized_proof) if normalized_proof is not None else None
    receipt_seed = EarlyTerminationReceipt(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        config_hash=cfg.stable_hash(),
        kernel_result_hash=str(normalized_kernel["result_hash"]),
        kernel_replay_identity=str(normalized_kernel["replay_identity"]),
        proof_input_hash=proof_input_hash,
        termination_decision_hash=decision.decision_hash,
        termination_replay_identity=decision.replay_identity,
        top_proposal_hash=top_proposal_hash,
        decision_label=decision.decision_label,
        terminate_early=decision.terminate_early,
        receipt_hash="",
    )
    receipt = EarlyTerminationReceipt(**{**receipt_seed.to_dict(), "receipt_hash": receipt_seed.stable_hash()})

    result_seed = {
        "release_version": RELEASE_VERSION,
        "runtime_kind": RUNTIME_KIND,
        "config": cfg.to_dict(),
        "kernel_result": normalized_kernel,
        "dark_state_inputs": normalized_proof,
        "top_proposal": top_proposal,
        "proof_signals": proof_signals.to_dict(),
        "decision": decision.to_hash_payload_dict(),
        "receipt": receipt.to_hash_payload_dict(),
        "advisory_only": True,
        "decoder_core_modified": False,
    }
    result_hash = _sha256_hex(result_seed)

    result = EarlyTerminationAnalysisResult(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        config=cfg,
        kernel_result=normalized_kernel,
        dark_state_inputs=normalized_proof,
        top_proposal=top_proposal,
        proof_signals=proof_signals,
        decision=decision,
        receipt=receipt,
        advisory_only=True,
        decoder_core_modified=False,
        result_hash=result_hash,
    )

    if result.decision.decision_hash != result.decision.stable_hash():
        raise EarlyTerminationDarkStateProofError("decision hash mismatch")
    if result.receipt.receipt_hash != result.receipt.stable_hash():
        raise EarlyTerminationDarkStateProofError("receipt hash mismatch")
    if result.receipt.termination_decision_hash != result.decision.decision_hash:
        raise EarlyTerminationDarkStateProofError("receipt termination_decision_hash mismatch")
    if result.receipt.kernel_result_hash != result.kernel_result["result_hash"]:
        raise EarlyTerminationDarkStateProofError("receipt kernel_result_hash mismatch")
    if result.result_hash != result.stable_hash():
        raise EarlyTerminationDarkStateProofError("result hash mismatch")
    return result


__all__ = [
    "RELEASE_VERSION",
    "RUNTIME_KIND",
    "EarlyTerminationDarkStateProofError",
    "EarlyTerminationDarkStateConfig",
    "DarkStateProofSignal",
    "EarlyTerminationCondition",
    "EarlyTerminationDecision",
    "EarlyTerminationReceipt",
    "EarlyTerminationAnalysisResult",
    "normalize_early_termination_config",
    "normalize_kernel_result",
    "normalize_dark_state_inputs",
    "compute_dark_state_proof_signals",
    "classify_early_termination_decision",
    "build_early_termination_analysis_result",
]
