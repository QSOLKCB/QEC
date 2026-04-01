"""v132.3.0 — Deterministic dwell-time guard.

Pure function that validates whether sufficient time has elapsed
since the last supervisory mode switch. Uses explicit integer
arithmetic only. No floating point. No system clock reads.

Wrap-safe for 64-bit monotonic counters.
"""

from __future__ import annotations


def can_switch_now(
    last_switch_time_ms: int,
    current_time_ms: int,
    dwell_time_ms: int,
) -> bool:
    """Return True iff the dwell-time constraint is satisfied.

    Parameters
    ----------
    last_switch_time_ms:
        Monotonic timestamp (ms) of the most recent mode switch.
    current_time_ms:
        Monotonic timestamp (ms) of the current evaluation point.
    dwell_time_ms:
        Minimum required dwell time (ms) before a new switch is legal.

    Returns
    -------
    bool
        True if elapsed >= dwell_time_ms, False otherwise.

    All arithmetic is explicit integer. Wrap-safe: uses modular
    subtraction under a 64-bit unsigned mask so that monotonic
    counter wraparound is handled correctly.
    """
    if dwell_time_ms < 0:
        raise ValueError("dwell_time_ms must be non-negative")

    # 64-bit unsigned wrap-safe elapsed computation
    _MASK_64 = (1 << 64) - 1
    elapsed = (current_time_ms - last_switch_time_ms) & _MASK_64

    return elapsed >= dwell_time_ms
