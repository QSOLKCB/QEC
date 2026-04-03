"""
QEC Decoder Portfolio Orchestrator — Deterministic Selection & Routing (v136.8.4).

Deterministic selection and routing layer for choosing the best decoder
pathway based on:
- code zoo registry
- controller snapshots
- audio cognition matches
- failure recall
- code family identity
- evidence score

Selection law (stable priority):
1. exact code family match
2. highest cognition confidence
3. highest expected_recovery_score
4. lowest route_priority integer

Tie-breaking is stable via sorted candidate ordering.

Design invariants
-----------------
* frozen dataclasses only
* deterministic — same inputs always produce identical decision
* no hidden randomness
* no decoder imports
* no decoder mutations
* stdlib only (+ existing repo deps)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from qec.ai.controller_snapshot_schema import ControllerSnapshot
from qec.audio.audio_cognition_engine import CognitionCycleResult
from qec.codes.code_zoo import CodeZooRegistry, build_default_code_zoo


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

ORCHESTRATOR_VERSION: str = "v136.8.4"


# ---------------------------------------------------------------------------
# Valid policy actions
# ---------------------------------------------------------------------------

VALID_ACTIONS: Tuple[str, ...] = (
    "DECODE_PORTFOLIO_A",
    "DECODE_PORTFOLIO_B",
    "DECODE_PORTFOLIO_C",
    "QLDPC_PORTFOLIO_B",
    "REINIT_CODE_LATTICE",
    "SURFACE_FAST_PATH",
    "TORIC_STABILITY_PATH",
)

# Deterministic mapping from code family to default action
_FAMILY_ACTION_MAP: Dict[str, str] = {
    "qldpc": "QLDPC_PORTFOLIO_B",
    "repetition": "DECODE_PORTFOLIO_A",
    "surface": "SURFACE_FAST_PATH",
    "toric": "TORIC_STABILITY_PATH",
}

_DEFAULT_ACTION: str = "DECODE_PORTFOLIO_B"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioCandidate:
    """Immutable candidate for decoder portfolio selection."""

    decoder_id: str
    code_family: str
    confidence: float
    expected_recovery_score: float
    route_priority: int


@dataclass(frozen=True)
class OrchestratorDecision:
    """Immutable result of orchestrator selection."""

    selected_decoder: str
    confidence: float
    rationale: str
    source_match: str
    policy_action: str


@dataclass(frozen=True)
class PortfolioRegistry:
    """Immutable registry of portfolio candidates."""

    candidates: Tuple[PortfolioCandidate, ...]
    registry_hash: str


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _candidate_to_canonical_dict(c: PortfolioCandidate) -> Dict[str, Any]:
    """Convert a PortfolioCandidate to a canonical dict."""
    return {
        "code_family": c.code_family,
        "confidence": c.confidence,
        "decoder_id": c.decoder_id,
        "expected_recovery_score": c.expected_recovery_score,
        "route_priority": c.route_priority,
    }


# ---------------------------------------------------------------------------
# Portfolio operations
# ---------------------------------------------------------------------------


def compute_portfolio_hash(registry: PortfolioRegistry) -> str:
    """Compute deterministic SHA-256 hash of a PortfolioRegistry.

    Same registry always produces identical hash.
    """
    canonical = {
        "candidates": [_candidate_to_canonical_dict(c) for c in registry.candidates],
        "version": ORCHESTRATOR_VERSION,
    }
    payload = _canonical_json(canonical)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sort_key(c: PortfolioCandidate) -> Tuple[str, str]:
    """Stable sort key for candidates: (code_family, decoder_id)."""
    return (c.code_family, c.decoder_id)


def _validate_candidate(candidate: PortfolioCandidate) -> None:
    """Validate a single PortfolioCandidate.

    Raises ValueError if any invariant is violated.
    """
    if not isinstance(candidate, PortfolioCandidate):
        raise ValueError(f"Expected PortfolioCandidate, got {type(candidate).__name__}")
    if not candidate.decoder_id:
        raise ValueError("decoder_id must be non-empty")
    if not candidate.code_family:
        raise ValueError("code_family must be non-empty")
    if not (0.0 <= candidate.confidence <= 1.0):
        raise ValueError(
            f"confidence must be in [0.0, 1.0], got {candidate.confidence}"
        )
    if not (0.0 <= candidate.expected_recovery_score <= 1.0):
        raise ValueError(
            f"expected_recovery_score must be in [0.0, 1.0], got {candidate.expected_recovery_score}"
        )
    if candidate.route_priority < 0:
        raise ValueError(
            f"route_priority must be >= 0, got {candidate.route_priority}"
        )


def register_portfolio_candidate(
    candidate: PortfolioCandidate,
    registry: Optional[PortfolioRegistry] = None,
) -> PortfolioRegistry:
    """Register a candidate into a portfolio registry.

    Validates candidate invariants at registration time.
    Maintains sorted order by (code_family, decoder_id).
    Recomputes registry hash.
    Raises ValueError on duplicate decoder_id or invalid candidate.
    """
    _validate_candidate(candidate)

    if registry is None:
        candidates = (candidate,)
    else:
        for existing in registry.candidates:
            if existing.decoder_id == candidate.decoder_id:
                raise ValueError(f"Duplicate decoder_id: {candidate.decoder_id!r}")
        candidates = tuple(
            sorted(registry.candidates + (candidate,), key=_sort_key)
        )

    tmp = PortfolioRegistry(candidates=candidates, registry_hash="")
    h = compute_portfolio_hash(tmp)
    return PortfolioRegistry(candidates=candidates, registry_hash=h)


def validate_portfolio_registry(registry: PortfolioRegistry) -> bool:
    """Validate a PortfolioRegistry against invariant rules.

    Returns True if valid. Raises ValueError if invalid.
    """
    if not isinstance(registry.candidates, tuple):
        raise ValueError(
            f"candidates must be a tuple, got {type(registry.candidates).__name__}"
        )

    if len(registry.candidates) == 0:
        raise ValueError("Portfolio registry must contain at least one candidate")

    seen_ids: set = set()
    for c in registry.candidates:
        _validate_candidate(c)
        if c.decoder_id in seen_ids:
            raise ValueError(f"Duplicate decoder_id: {c.decoder_id!r}")
        seen_ids.add(c.decoder_id)

    # Verify sorted order
    sort_keys = [_sort_key(c) for c in registry.candidates]
    if sort_keys != sorted(sort_keys):
        raise ValueError("Candidates must be sorted by (code_family, decoder_id)")

    # Verify hash
    expected = compute_portfolio_hash(registry)
    if registry.registry_hash != expected:
        raise ValueError(
            f"registry_hash mismatch: expected {expected}, got {registry.registry_hash}"
        )

    return True


def build_default_decoder_portfolio() -> PortfolioRegistry:
    """Build the default decoder portfolio from the code zoo.

    Deterministic: same call always produces identical registry.
    """
    zoo = build_default_code_zoo()
    families = sorted(set(spec.family for spec in zoo.codes))

    candidates = []
    for idx, family in enumerate(families):
        candidates.append(
            PortfolioCandidate(
                decoder_id=f"decoder_{family}_default",
                code_family=family,
                confidence=0.9,
                expected_recovery_score=0.85,
                route_priority=idx,
            )
        )

    candidates_sorted = tuple(sorted(candidates, key=_sort_key))
    tmp = PortfolioRegistry(candidates=candidates_sorted, registry_hash="")
    h = compute_portfolio_hash(tmp)
    return PortfolioRegistry(candidates=candidates_sorted, registry_hash=h)


# ---------------------------------------------------------------------------
# Selection engine
# ---------------------------------------------------------------------------


def _resolve_action(code_family: str) -> str:
    """Resolve deterministic policy action from code family."""
    return _FAMILY_ACTION_MAP.get(code_family, _DEFAULT_ACTION)


def select_decoder_path(
    code_family: str,
    cognition_match: float,
    snapshot: ControllerSnapshot,
    registry: Optional[PortfolioRegistry] = None,
) -> OrchestratorDecision:
    """Select the optimal decoder path deterministically.

    Selection priority (stable):
    1. exact code_family match filters candidates
    2. highest candidate confidence
    3. highest expected_recovery_score
    4. lowest route_priority integer
    5. decoder_id alphabetical (final tie-break)

    cognition_match and snapshot.evidence_score are combined with
    the winning candidate's confidence to produce the output
    confidence score but do not affect candidate ranking.

    Parameters
    ----------
    code_family
        Target code family identifier.
    cognition_match
        Confidence from audio cognition cycle [0.0, 1.0].
    snapshot
        Controller snapshot for state context. ``snapshot.evidence_score``
        is the authoritative evidence input.
    registry
        Portfolio registry. Uses default if None.
    """
    if registry is None:
        registry = build_default_decoder_portfolio()

    if not registry.candidates:
        raise ValueError("Cannot select from empty portfolio registry")

    evidence_score = snapshot.evidence_score

    # Filter candidates matching code_family first
    family_matches = tuple(
        c for c in registry.candidates if c.code_family == code_family
    )

    if family_matches:
        # Sort by selection law: highest confidence, highest recovery, lowest priority
        ranked = sorted(
            family_matches,
            key=lambda c: (-c.confidence, -c.expected_recovery_score, c.route_priority, c.decoder_id),
        )
        best = ranked[0]
        rationale = f"exact_family_match:{code_family}"
        source = "code_family"
    else:
        # No exact match: fall back to best overall candidate
        ranked = sorted(
            registry.candidates,
            key=lambda c: (-c.confidence, -c.expected_recovery_score, c.route_priority, c.decoder_id),
        )
        best = ranked[0]
        rationale = f"fallback_best_candidate:{best.code_family}"
        source = "fallback"

    # Combine confidence: candidate confidence weighted with cognition and evidence
    combined_confidence = round(
        best.confidence * 0.5 + cognition_match * 0.3 + evidence_score * 0.2, 15
    )

    # Resolve action from the requested code_family so that unknown
    # families deterministically map to _DEFAULT_ACTION rather than
    # inheriting the fallback candidate's family action.
    action = _resolve_action(code_family)

    # If snapshot invariant failed, override to REINIT
    if not snapshot.invariant_passed:
        action = "REINIT_CODE_LATTICE"
        rationale = f"invariant_failed:{rationale}"

    return OrchestratorDecision(
        selected_decoder=best.decoder_id,
        confidence=combined_confidence,
        rationale=rationale,
        source_match=source,
        policy_action=action,
    )


# ---------------------------------------------------------------------------
# Orchestration cycle
# ---------------------------------------------------------------------------


def run_orchestration_cycle(
    code_family: str,
    cognition_cycle_result: CognitionCycleResult,
    snapshot: ControllerSnapshot,
    registry: PortfolioRegistry,
) -> OrchestratorDecision:
    """Run a full orchestration cycle integrating all inputs.

    Combines:
    - code family identity
    - audio cognition match confidence
    - controller snapshot evidence
    - portfolio registry candidates

    Deterministic: same inputs always produce identical decision.
    """
    cognition_confidence = cognition_cycle_result.match.confidence

    return select_decoder_path(
        code_family=code_family,
        cognition_match=cognition_confidence,
        snapshot=snapshot,
        registry=registry,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_orchestration_bundle(result: OrchestratorDecision) -> Dict[str, Any]:
    """Export an OrchestratorDecision as a canonical JSON-serializable dict.

    Deterministic: same result always produces identical export.
    """
    return {
        "confidence": result.confidence,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "policy_action": result.policy_action,
        "rationale": result.rationale,
        "selected_decoder": result.selected_decoder,
        "source_match": result.source_match,
    }


def export_orchestration_bundle_json(result: OrchestratorDecision) -> str:
    """Export an OrchestratorDecision as canonical JSON string.

    Deterministic: same result always produces byte-identical JSON.
    """
    bundle = export_orchestration_bundle(result)
    return _canonical_json(bundle)
