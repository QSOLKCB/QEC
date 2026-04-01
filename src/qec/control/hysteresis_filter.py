"""v132.3.0 — Deterministic hysteresis filter.

Anti-chatter switching suppression using fixed threshold bands.
No adaptive thresholds. No learned thresholds. No probabilistic logic.

The hysteresis band is defined by enter_threshold > exit_threshold.
A mode is entered when the metric rises to enter_threshold,
and exited only when it falls to exit_threshold.
"""

from __future__ import annotations

from typing import Union

Numeric = Union[int, float]


def passes_hysteresis(
    current_metric: Numeric,
    enter_threshold: Numeric,
    exit_threshold: Numeric,
    current_mode: str,
) -> bool:
    """Evaluate whether a mode transition passes hysteresis filtering.

    Parameters
    ----------
    current_metric:
        The observed metric value driving the switching decision.
    enter_threshold:
        Metric must reach this level to enter the candidate mode.
    exit_threshold:
        Metric must fall to this level to exit the current mode.
    current_mode:
        Either ``"active"`` (currently in the candidate mode) or
        ``"inactive"`` (not yet in the candidate mode).

    Returns
    -------
    bool
        True if the candidate mode should be active after this
        evaluation. False if it should be inactive.

    The hysteresis band between exit_threshold and enter_threshold
    prevents rapid oscillation: once entered, the mode persists
    until the metric drops to exit_threshold.
    """
    if enter_threshold < exit_threshold:
        raise ValueError(
            "enter_threshold must be >= exit_threshold for valid hysteresis band"
        )

    if current_mode == "active":
        # Already in the mode — stay unless metric drops to exit
        return current_metric > exit_threshold
    else:
        # Not in the mode — enter only if metric reaches enter threshold
        return current_metric >= enter_threshold
