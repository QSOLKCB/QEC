"""Deterministic baseline strategies for benchmarking — v101.0.0.

Provides simple, deterministic strategy selection baselines for comparison
against the full QEC adaptive pipeline. All baselines are pure functions
with no hidden state or randomness.

Dependencies: hashlib (stdlib).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List


def random_strategy_deterministic(
    seed: int,
    strategies: List[str],
    step: int,
) -> str:
    """Select a strategy using seeded deterministic pseudo-randomness.

    Uses SHA-256 to derive a deterministic index from seed and step,
    ensuring identical outputs for identical inputs across all platforms.

    Parameters
    ----------
    seed : int
        Base seed for reproducibility.
    strategies : list of str
        Sorted list of strategy identifiers.
    step : int
        Current step index.

    Returns
    -------
    str
        Selected strategy identifier.

    Raises
    ------
    ValueError
        If strategies is empty.
    """
    if not strategies:
        raise ValueError("strategies must not be empty")
    # Deterministic sub-seed via SHA-256
    digest = hashlib.sha256(f"{seed}:{step}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(strategies)
    return strategies[index]


def fixed_strategy(strategy_id: str) -> str:
    """Return a fixed strategy regardless of step or state.

    Parameters
    ----------
    strategy_id : str
        The strategy identifier to always return.

    Returns
    -------
    str
        The same strategy_id passed in.
    """
    return strategy_id


def round_robin_strategy(step: int, strategies: List[str]) -> str:
    """Select a strategy by deterministic cycling through the list.

    Parameters
    ----------
    step : int
        Current step index.
    strategies : list of str
        Sorted list of strategy identifiers.

    Returns
    -------
    str
        Selected strategy identifier.

    Raises
    ------
    ValueError
        If strategies is empty.
    """
    if not strategies:
        raise ValueError("strategies must not be empty")
    return strategies[step % len(strategies)]
