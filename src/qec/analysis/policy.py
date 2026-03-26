"""v103.4.0 — Policy framework for hierarchical control.

Provides:
- Policy class: reusable, composable policy objects
- Policy composition: deterministic merging of multiple policies
- Policy registry: named policy lookup and registration

Replaces dict-based policy definitions with first-class objects
while preserving exact behavioral equivalence.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- rule-based only (no stochastic routing, no learning)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from qec.analysis.hierarchical_control import (
    DEFAULT_INSTABILITY_THRESHOLD,
    DEFAULT_SYNC_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------


class Policy:
    """A reusable, composable control policy.

    Encapsulates a control policy with mode, priority, and thresholds.
    Provides a ``decide`` method that returns the routing mode
    (``"local"``, ``"global"``, or ``"hybrid"``) given current state.

    Parameters
    ----------
    name : str
        Human-readable policy name.
    mode : str
        Base routing mode (``"local"``, ``"global"``, or ``"hybrid"``).
    priority : str
        Priority type (``"stability"``, ``"synchronization"``, or
        ``"balanced"``).
    thresholds : dict
        Threshold values for policy decisions. Expected keys:
        ``"instability"`` and ``"sync"``.
    """

    def __init__(
        self,
        name: str,
        mode: str,
        priority: str,
        thresholds: Dict[str, float],
    ) -> None:
        self.name = name
        self.mode = mode
        self.priority = priority
        self.thresholds = dict(thresholds)

    def decide(self, _state: Dict[str, Any], global_state: Dict[str, Any]) -> str:
        """Determine control mode ('local', 'global', 'hybrid').

        Parameters
        ----------
        _state : dict
            Local state (currently unused, retained for API compatibility).
        global_state : dict
            Aggregated system state used for decision.

        Returns
        -------
        str
            One of ``"local"``, ``"global"``, or ``"hybrid"``.

        Notes
        -----
        The ``_state`` parameter is intentionally unused to preserve
        compatibility with earlier interfaces and allow future extension.
        """
        if self.mode in ("local", "global"):
            return self.mode

        avg_stability = float(global_state.get("avg_stability", 0.5))
        avg_sync = float(global_state.get("avg_sync", 0.5))
        inst_thresh = float(self.thresholds.get("instability", 0.5))
        sync_thresh = float(self.thresholds.get("sync", 0.5))

        if self.priority == "stability":
            return "local" if avg_stability >= inst_thresh else "global"

        if self.priority == "synchronization":
            return "local" if avg_sync >= sync_thresh else "global"

        # balanced: check both metrics
        stable = avg_stability >= inst_thresh
        synced = avg_sync >= sync_thresh

        if stable and synced:
            return "local"
        if not stable and not synced:
            return "global"
        return "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the legacy dict format for backward compatibility.

        Returns
        -------
        dict
            Dict with ``"mode"``, ``"priority"``, and ``"thresholds"``.
        """
        return {
            "mode": self.mode,
            "priority": self.priority,
            "thresholds": dict(self.thresholds),
        }

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "Policy":
        """Create a Policy from a legacy dict.

        Parameters
        ----------
        name : str
            Policy name.
        d : dict
            Dict with ``"mode"``, ``"priority"``, and ``"thresholds"``.

        Returns
        -------
        Policy
        """
        return cls(
            name=name,
            mode=str(d.get("mode", "hybrid")),
            priority=str(d.get("priority", "balanced")),
            thresholds=dict(d.get("thresholds", {})),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Policy):
            return NotImplemented
        return (
            self.name == other.name
            and self.mode == other.mode
            and self.priority == other.priority
            and self.thresholds == other.thresholds
        )

    def __repr__(self) -> str:
        return (
            f"Policy(name={self.name!r}, mode={self.mode!r}, "
            f"priority={self.priority!r}, thresholds={self.thresholds!r})"
        )


# ---------------------------------------------------------------------------
# Policy composition
# ---------------------------------------------------------------------------

# Priority ranking for resolution (higher = wins).
_PRIORITY_RANK = {
    "stability": 2,
    "synchronization": 1,
    "balanced": 0,
}


def compose_policies(policies: List[Policy]) -> Policy:
    """Compose multiple policies into a single composite policy.

    Composition rules (deterministic):

    - **Thresholds**: averaged across all policies.
    - **Priority**: highest-ranked priority wins
      (stability > synchronization > balanced).
    - **Mode**: if all agree, use that mode; otherwise ``"hybrid"``.
    - **Name**: joined with ``" + "``.

    Parameters
    ----------
    policies : list of Policy
        Policies to compose. Must be non-empty.

    Returns
    -------
    Policy
        The composed policy.

    Raises
    ------
    ValueError
        If *policies* is empty.
    """
    if not policies:
        raise ValueError("Cannot compose an empty list of policies")

    if len(policies) == 1:
        p = policies[0]
        return Policy(
            name=p.name,
            mode=p.mode,
            priority=p.priority,
            thresholds=dict(p.thresholds),
        )

    # Combine names.
    name = " + ".join(p.name for p in policies)

    # Resolve priority: highest rank wins.
    best_priority = max(
        (p.priority for p in policies),
        key=lambda pr: _PRIORITY_RANK.get(pr, -1),
    )

    # Resolve mode: unanimous → that mode, otherwise hybrid.
    modes = sorted(set(p.mode for p in policies))
    mode = modes[0] if len(modes) == 1 else "hybrid"

    # Average thresholds deterministically.
    all_keys = sorted(
        set(k for p in policies for k in p.thresholds),
    )
    thresholds: Dict[str, float] = {}
    for key in all_keys:
        values = [p.thresholds[key] for p in policies if key in p.thresholds]
        thresholds[key] = round(sum(values) / len(values), 12)

    return Policy(
        name=name,
        mode=mode,
        priority=best_priority,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------

POLICY_REGISTRY: Dict[str, Policy] = {}


def register_policy(policy: Policy) -> None:
    """Register a policy in the global registry.

    Parameters
    ----------
    policy : Policy
        Policy to register. Overwrites any existing policy with the
        same name.
    """
    POLICY_REGISTRY[policy.name] = policy


def get_policy(name: str) -> Policy:
    """Retrieve a policy from the registry by name.

    Parameters
    ----------
    name : str
        Policy name.

    Returns
    -------
    Policy

    Raises
    ------
    KeyError
        If no policy with *name* is registered.
    """
    if name not in POLICY_REGISTRY:
        raise KeyError(
            f"Unknown policy: {name!r}. "
            f"Registered: {sorted(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[name]


def list_policies() -> List[str]:
    """Return sorted list of registered policy names.

    Returns
    -------
    list of str
    """
    return sorted(POLICY_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Register built-in policies on import
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Register the three built-in policies."""
    _defaults = {
        "instability": DEFAULT_INSTABILITY_THRESHOLD,
        "sync": DEFAULT_SYNC_THRESHOLD,
    }
    register_policy(Policy(
        name="stability_first",
        mode="hybrid",
        priority="stability",
        thresholds=dict(_defaults),
    ))
    register_policy(Policy(
        name="sync_first",
        mode="hybrid",
        priority="synchronization",
        thresholds=dict(_defaults),
    ))
    register_policy(Policy(
        name="balanced",
        mode="hybrid",
        priority="balanced",
        thresholds=dict(_defaults),
    ))


_register_builtins()
