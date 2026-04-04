"""
Governed Steering Integration — v136.9.4

Composes v136.9.2 adaptive steering with v136.9.3 policy memory
as an optional supervisory wrapper.

Closes the full deterministic control loop:

    forecast
    -> adaptive steering (v136.9.2)
    -> policy memory (v136.9.3)
    -> governed output (v136.9.4)

This is integration + composition.
NOT a new control law.

Layer: analysis (Layer 4) — additive supervisory control.
Never imports or mutates decoder internals.

All outputs are deterministic, frozen, and byte-identical on replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from qec.analysis.forecast_guided_steering import (
    AdaptiveSteeringDecision,
    route_with_forecast_guidance,
)
from qec.analysis.adaptive_steering_policy_memory import (
    PolicyMemoryState,
    update_policy_memory,
)
from qec.analysis.phase_space_decoder_steering import (
    PhaseSteeringDecision,
)
from qec.analysis.spectral_attractor_forecasting import (
    SpectralForecastDecision,
)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

GOVERNED_STEERING_VERSION: str = "v136.9.4"

# Float precision for deterministic hashing
FLOAT_PRECISION: int = 12


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GovernedSteeringBundle:
    """Immutable governed steering bundle.

    Contains both the raw adaptive steering decision and the
    policy-memory-governed route, with explicit transparency
    about whether governance mutated the route.
    """
    raw_decision: AdaptiveSteeringDecision
    governed_route: str
    policy_state: Optional[PolicyMemoryState]
    governance_applied: bool
    route_mutated: bool
    cycle_index: int
    stable_hash: str


@dataclass(frozen=True)
class GovernedSteeringLedger:
    """Immutable ordered record of governed steering bundles."""
    bundles: Tuple[GovernedSteeringBundle, ...]
    bundle_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_governed_steering(
    steering_decision: PhaseSteeringDecision,
    forecast_decision: SpectralForecastDecision,
    prior_policy_state: Optional[PolicyMemoryState] = None,
    enable_policy_memory: bool = True,
) -> GovernedSteeringBundle:
    """Compute a governed steering bundle from steering + forecast inputs.

    Flow:
      1. Compute raw adaptive steering decision (v136.9.2)
      2. Optionally apply policy memory wrapper (v136.9.3)
      3. Emit both raw + governed outputs in a single bundle

    Parameters
    ----------
    steering_decision : PhaseSteeringDecision
        The v136.9.0 phase-space steering decision.
    forecast_decision : SpectralForecastDecision
        The v136.9.1 spectral forecast decision.
    prior_policy_state : PolicyMemoryState or None
        Prior policy memory state. None initializes fresh.
    enable_policy_memory : bool
        If True, apply policy memory governance.
        If False, governed route == raw route, no supervisory mutation.

    Returns
    -------
    GovernedSteeringBundle
        Frozen, deterministic governed steering bundle.
    """
    # Step 1: compute raw adaptive steering decision
    raw_decision = route_with_forecast_guidance(
        steering_decision, forecast_decision,
    )

    # Step 2: optionally apply policy memory
    if enable_policy_memory:
        policy_state, governed_route = update_policy_memory(
            raw_decision, prior_policy_state,
        )
        governance_applied = True
        cycle_index = policy_state.policy_cycle_index
    else:
        policy_state = None
        governed_route = raw_decision.adaptive_recovery_route
        governance_applied = False
        cycle_index = (
            prior_policy_state.policy_cycle_index + 1
            if prior_policy_state is not None
            else 0
        )

    # Step 3: determine if governance mutated the route
    route_mutated = (governed_route != raw_decision.adaptive_recovery_route)

    # Build with empty hash, then compute
    preliminary = GovernedSteeringBundle(
        raw_decision=raw_decision,
        governed_route=governed_route,
        policy_state=policy_state,
        governance_applied=governance_applied,
        route_mutated=route_mutated,
        cycle_index=cycle_index,
        stable_hash="",
    )
    stable_hash = _compute_bundle_hash(preliminary)

    return GovernedSteeringBundle(
        raw_decision=raw_decision,
        governed_route=governed_route,
        policy_state=policy_state,
        governance_applied=governance_applied,
        route_mutated=route_mutated,
        cycle_index=cycle_index,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _bundle_to_canonical_dict(b: GovernedSteeringBundle) -> Dict[str, Any]:
    """Convert a GovernedSteeringBundle to a canonical dict for hashing.

    Shape mirrors export minus layer/version/stable_hash metadata.
    Policy state fields are flat with ``policy_`` prefix.
    Includes raw_decision_stable_hash for full upstream identity.
    """
    if b.policy_state is not None:
        p_consecutive_low = b.policy_state.consecutive_low_risk_cycles
        p_consecutive_rec = b.policy_state.consecutive_recovery_cycles
        p_current_route = b.policy_state.current_route
        p_last_escalation = b.policy_state.last_escalation_level
        p_oscillation = b.policy_state.oscillation_count
        p_cycle_index = b.policy_state.policy_cycle_index
        p_prior_route = b.policy_state.prior_route
    else:
        p_consecutive_low = 0
        p_consecutive_rec = 0
        p_current_route = ""
        p_last_escalation = 0
        p_oscillation = 0
        p_cycle_index = 0
        p_prior_route = ""

    return {
        "adaptive_decoder_bias": list(b.raw_decision.adaptive_decoder_bias),
        "adaptive_escalation_level": b.raw_decision.adaptive_escalation_level,
        "adaptive_recovery_route": b.raw_decision.adaptive_recovery_route,
        "adaptive_rollback_weight": _round(b.raw_decision.adaptive_rollback_weight),
        "cycle_index": b.cycle_index,
        "forecast_label": b.raw_decision.forecast_label,
        "forecast_risk_score": _round(b.raw_decision.forecast_risk_score),
        "governance_applied": b.governance_applied,
        "governed_route": b.governed_route,
        "policy_consecutive_low_risk_cycles": p_consecutive_low,
        "policy_consecutive_recovery_cycles": p_consecutive_rec,
        "policy_current_route": p_current_route,
        "policy_cycle_index": p_cycle_index,
        "policy_last_escalation_level": p_last_escalation,
        "policy_oscillation_count": p_oscillation,
        "policy_prior_route": p_prior_route,
        "precollapse_detected": b.raw_decision.precollapse_detected,
        "raw_decision_stable_hash": b.raw_decision.stable_hash,
        "route_mutated": b.route_mutated,
    }


def _compute_bundle_hash(b: GovernedSteeringBundle) -> str:
    """SHA-256 of canonical JSON of a governed steering bundle."""
    payload = _bundle_to_canonical_dict(b)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(ledger: GovernedSteeringLedger) -> str:
    """SHA-256 of canonical JSON of a governed steering ledger."""
    payload = {
        "bundle_count": ledger.bundle_count,
        "bundles": [_bundle_to_canonical_dict(b) for b in ledger.bundles],
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Ledger operations
# ---------------------------------------------------------------------------

def build_governed_ledger(
    bundles: Tuple[GovernedSteeringBundle, ...] = (),
) -> GovernedSteeringLedger:
    """Build an immutable governed steering ledger from a tuple of bundles."""
    bundles = tuple(bundles)
    tmp = GovernedSteeringLedger(
        bundles=bundles,
        bundle_count=len(bundles),
        stable_hash="",
    )
    stable_hash = _compute_ledger_hash(tmp)
    return GovernedSteeringLedger(
        bundles=bundles,
        bundle_count=len(bundles),
        stable_hash=stable_hash,
    )


def append_governed_bundle(
    bundle: GovernedSteeringBundle,
    ledger: GovernedSteeringLedger,
) -> GovernedSteeringLedger:
    """Append a bundle to the ledger, returning a new immutable ledger."""
    new_bundles = ledger.bundles + (bundle,)
    return build_governed_ledger(new_bundles)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_governed_steering_bundle(
    bundle: GovernedSteeringBundle,
) -> Dict[str, Any]:
    """Export a governed steering bundle as a canonical JSON-safe dict.

    Deterministic: same bundle always produces byte-identical export.

    Includes:
      - raw route
      - governed route
      - whether route mutated
      - policy memory counters
      - oscillation count
      - cycle index

    Parameters
    ----------
    bundle : GovernedSteeringBundle
        The governed steering bundle to export.

    Returns
    -------
    dict
        Stable dictionary with sorted keys, suitable for JSON serialization.
    """
    # Canonical payload mirrors _bundle_to_canonical_dict shape
    canonical = _bundle_to_canonical_dict(bundle)

    # Export = canonical + layer/version/stable_hash metadata
    result: Dict[str, Any] = dict(canonical)
    result["layer"] = "governed_steering_integration"
    result["stable_hash"] = bundle.stable_hash
    result["version"] = GOVERNED_STEERING_VERSION

    return result
