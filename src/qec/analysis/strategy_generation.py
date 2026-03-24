"""v101.6.0 — Deterministic strategy generation.

Produces a structured set of 27 candidate strategies by enumerating
a bounded, interpretable design space over three operators:

  - Confidence scaling (CONF_SCALES)
  - Neutral bias (NEUTRAL_BIAS)
  - Iteration depth (DEPTHS)

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib + copy.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Operator value sets — fixed, ordered, deterministic
# ---------------------------------------------------------------------------

CONF_SCALES: Tuple[float, ...] = (0.75, 1.0, 1.25)
NEUTRAL_BIAS: Tuple[float, ...] = (-0.1, 0.0, 0.1)
DEPTHS: Tuple[int, ...] = (3, 4, 5)

EXPECTED_COUNT: int = len(CONF_SCALES) * len(NEUTRAL_BIAS) * len(DEPTHS)  # 27


# ---------------------------------------------------------------------------
# Strategy name construction
# ---------------------------------------------------------------------------

def _build_name(
    scale: float,
    bias: float,
    depth: int,
    *,
    state_system: str = "ternary",
) -> str:
    """Build a deterministic, sortable strategy name."""
    return f"{state_system}__conf_{scale}__bias_{bias:+.1f}__depth_{depth}"


# ---------------------------------------------------------------------------
# Operator application
# ---------------------------------------------------------------------------

def _apply_confidence_scale(config: Dict[str, Any], scale: float) -> None:
    """Apply confidence scaling to config (in-place on fresh copy)."""
    raw = float(config.get("confidence_scale", 1.0))
    config["confidence_scale"] = min(1.0, max(0.0, raw * scale))


def _apply_neutral_bias(config: Dict[str, Any], bias: float) -> None:
    """Apply neutral bias to config (in-place on fresh copy)."""
    config["neutral_bias"] = float(bias)


def _apply_depth(config: Dict[str, Any], depth: int) -> None:
    """Apply iteration depth to config (in-place on fresh copy)."""
    config["rounds"] = int(depth)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def generate_strategies(
    base_strategy: Dict[str, Any],
    *,
    include_quaternary: bool = False,
) -> List[Dict[str, Any]]:
    """Generate deterministically ordered candidate strategies.

    Enumerates all combinations of confidence scaling, neutral bias,
    and iteration depth applied to a deep copy of the base strategy
    config.

    When *include_quaternary* is True, generates both ternary (27) and
    quaternary (27) strategies for a total of 54 candidates.

    Parameters
    ----------
    base_strategy : dict
        Must contain at least a ``"config"`` key with a dict value.
        The config is deep-copied for each candidate.
    include_quaternary : bool
        If True, also generate quaternary variants (54 total).

    Returns
    -------
    list of dict
        27 or 54 strategies, each with:
        - ``"name"``: deterministic sortable name
        - ``"config"``: modified config dict
        - ``"origin"``: tuple of (confidence_scale, neutral_bias, depth)
        - ``"state_system"``: ``"ternary"`` or ``"quaternary"``
    """
    base_config = base_strategy.get("config", {})

    state_systems = ["ternary"]
    if include_quaternary:
        state_systems.append("quaternary")

    candidates: List[Dict[str, Any]] = []

    for state_system in state_systems:
        # Fixed operator order: CONF -> BIAS -> DEPTH
        for scale in CONF_SCALES:
            for bias in NEUTRAL_BIAS:
                for depth in DEPTHS:
                    cfg = copy.deepcopy(base_config)

                    _apply_confidence_scale(cfg, scale)
                    _apply_neutral_bias(cfg, bias)
                    _apply_depth(cfg, depth)
                    cfg["state_system"] = state_system

                    name = _build_name(
                        scale, bias, depth, state_system=state_system,
                    )

                    candidates.append({
                        "name": name,
                        "config": cfg,
                        "origin": (scale, bias, depth),
                        "state_system": state_system,
                    })

    # Sort by name for deterministic ordering
    candidates.sort(key=lambda c: c["name"])

    expected = EXPECTED_COUNT * len(state_systems)
    assert len(candidates) == expected
    return candidates


__all__ = [
    "CONF_SCALES",
    "NEUTRAL_BIAS",
    "DEPTHS",
    "EXPECTED_COUNT",
    "generate_strategies",
]
