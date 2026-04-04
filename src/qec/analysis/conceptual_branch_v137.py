"""
Conceptual Branch Module — v137.0.0

Read-only conceptual integration layer unifying:

    v136.9.x governed steering control stack
    +
    v136.10.0 cross-domain quantization framework

under a single architectural vocabulary.

Core conceptual law:

    continuous system state
    -> quantized symbolic state
    -> governed control action
    -> temporal supervisory memory
    -> stable replay identity

This is an architecture layer.
NOT a new controller.
NOT a new control law.

Layer: analysis (Layer 4) — additive conceptual integration.
Never imports or mutates decoder internals.

All outputs are deterministic, frozen, and byte-identical on replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

CONCEPTUAL_BRANCH_VERSION: str = "v137.0.0"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConceptualBranchState:
    """Immutable branch-era state descriptor.

    Unifies governed steering control (v136.9.x) and cross-domain
    quantization (v136.10.0) into a single canonical state artifact.

    This is the primary conceptual output of v137.0.0.
    """
    quantized_risk_band: str
    governed_route: str
    policy_cycle_index: int
    oscillation_count: int
    phase_bin_index: Tuple[int, int]
    quantization_domain: str
    replay_hash_chain: str
    branch_epoch: str = CONCEPTUAL_BRANCH_VERSION


@dataclass(frozen=True)
class ConceptualBranchLedger:
    """Immutable ordered ledger of conceptual branch states."""
    states: Tuple[ConceptualBranchState, ...]
    state_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Canonical JSON helpers
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


# ---------------------------------------------------------------------------
# Hash chain computation
# ---------------------------------------------------------------------------

def _compute_replay_hash_chain(
    governed_bundle_hash: str,
    quantization_decision_hash: str,
    policy_memory_hash: Optional[str] = None,
) -> str:
    """Compute a compositional replay hash chain.

    Deterministic SHA-256 over the ordered composition of upstream
    stable hashes.  This formalizes multi-layer replay identity.

    Parameters
    ----------
    governed_bundle_hash : str
        SHA-256 stable_hash from a GovernedSteeringBundle.
    quantization_decision_hash : str
        SHA-256 stable_hash from a QuantizationDecision.
    policy_memory_hash : str or None
        Optional SHA-256 stable_hash from a PolicyMemoryLedger.

    Returns
    -------
    str
        SHA-256 hex digest of the composed hash chain.
    """
    chain_elements = [
        ("governed_bundle_hash", governed_bundle_hash),
        ("quantization_decision_hash", quantization_decision_hash),
    ]
    if policy_memory_hash is not None:
        chain_elements.append(("policy_memory_hash", policy_memory_hash))

    payload = _canonical_json(chain_elements)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def _state_to_canonical_dict(s: ConceptualBranchState) -> Dict[str, Any]:
    """Convert a ConceptualBranchState to a canonical dict for hashing."""
    return {
        "branch_epoch": s.branch_epoch,
        "governed_route": s.governed_route,
        "oscillation_count": s.oscillation_count,
        "phase_bin_index": list(s.phase_bin_index),
        "policy_cycle_index": s.policy_cycle_index,
        "quantization_domain": s.quantization_domain,
        "quantized_risk_band": s.quantized_risk_band,
        "replay_hash_chain": s.replay_hash_chain,
    }


def _compute_ledger_hash(states: Tuple[ConceptualBranchState, ...]) -> str:
    """SHA-256 over canonical JSON of all states in the ledger."""
    payload = {
        "state_count": len(states),
        "states": [_state_to_canonical_dict(s) for s in states],
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def unify_control_and_quantization(
    governed_route: str,
    quantized_risk_band: str,
    policy_cycle_index: int,
    oscillation_count: int,
    phase_bin_index: Tuple[int, int],
    quantization_domain: str,
    governed_bundle_hash: str,
    quantization_decision_hash: str,
    policy_memory_hash: Optional[str] = None,
) -> ConceptualBranchState:
    """Unify governed control and quantization into a single branch state.

    This is the primary integration function of v137.0.0.
    It consumes upstream artifacts and emits a single unified
    conceptual state descriptor with a compositional replay hash chain.

    Parameters
    ----------
    governed_route : str
        The governed route from v136.9.4 (e.g. PRIMARY, RECOVERY).
    quantized_risk_band : str
        The quantized risk band from v136.10.0 (e.g. LOW, CRITICAL).
    policy_cycle_index : int
        Current policy memory cycle index.
    oscillation_count : int
        Current oscillation count from policy memory.
    phase_bin_index : tuple of (int, int)
        Phase-space bin indices (iq, ip) from quantization.
    quantization_domain : str
        The quantization domain that produced the risk band.
    governed_bundle_hash : str
        SHA-256 stable_hash from the GovernedSteeringBundle.
    quantization_decision_hash : str
        SHA-256 stable_hash from the QuantizationDecision.
    policy_memory_hash : str or None
        Optional SHA-256 stable_hash from a PolicyMemoryLedger.

    Returns
    -------
    ConceptualBranchState
        Frozen, deterministic conceptual branch state.
    """
    replay_hash_chain = _compute_replay_hash_chain(
        governed_bundle_hash=governed_bundle_hash,
        quantization_decision_hash=quantization_decision_hash,
        policy_memory_hash=policy_memory_hash,
    )

    return ConceptualBranchState(
        quantized_risk_band=quantized_risk_band,
        governed_route=governed_route,
        policy_cycle_index=policy_cycle_index,
        oscillation_count=oscillation_count,
        phase_bin_index=tuple(phase_bin_index),
        quantization_domain=quantization_domain,
        replay_hash_chain=replay_hash_chain,
    )


# ---------------------------------------------------------------------------
# Ledger operations
# ---------------------------------------------------------------------------

def build_branch_ledger(
    states: Tuple[ConceptualBranchState, ...] = (),
) -> ConceptualBranchLedger:
    """Build an immutable conceptual branch ledger from states.

    Parameters
    ----------
    states : tuple of ConceptualBranchState
        Ordered conceptual branch states.

    Returns
    -------
    ConceptualBranchLedger
        Frozen ledger with stable hash.
    """
    states = tuple(states)
    stable_hash = _compute_ledger_hash(states)
    return ConceptualBranchLedger(
        states=states,
        state_count=len(states),
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_conceptual_branch_bundle(
    state: ConceptualBranchState,
) -> Dict[str, Any]:
    """Export a conceptual branch state as a canonical JSON-safe dict.

    Deterministic: same state always produces byte-identical export.

    Parameters
    ----------
    state : ConceptualBranchState
        The conceptual branch state to export.

    Returns
    -------
    dict
        Stable dictionary with sorted keys, suitable for JSON serialization.
    """
    result = _state_to_canonical_dict(state)
    result["layer"] = "conceptual_branch_v137"
    result["version"] = CONCEPTUAL_BRANCH_VERSION
    return result


def export_conceptual_branch_ledger(
    ledger: ConceptualBranchLedger,
) -> str:
    """Export a conceptual branch ledger as canonical JSON string.

    Deterministic serialization — sorted keys, minimal separators.
    Byte-identical on replay.

    Parameters
    ----------
    ledger : ConceptualBranchLedger
        Immutable conceptual branch ledger.

    Returns
    -------
    str
        Canonical JSON string.
    """
    entries = [_state_to_canonical_dict(s) for s in ledger.states]
    bundle = {
        "stable_hash": ledger.stable_hash,
        "state_count": ledger.state_count,
        "states": entries,
        "version": CONCEPTUAL_BRANCH_VERSION,
    }
    return _canonical_json(bundle)
