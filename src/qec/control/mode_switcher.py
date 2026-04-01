"""v132.3.0 — Atomic deterministic mode switcher.

Combines dwell-time guard, hysteresis filter, and fail-safe latch
into a single atomic evaluation. Returns a frozen dataclass summary
for every switching decision — fully deterministic and replayable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Tuple

from .dwell_time_guard import can_switch_now
from .hysteresis_filter import passes_hysteresis


# ---------------------------------------------------------------------------
# Fail-safe constants
# ---------------------------------------------------------------------------

#: Maximum blocked attempts within a dwell window before fail-safe latches.
MAX_BLOCKED_BEFORE_LATCH = 3


@dataclass(frozen=True)
class ModeSwitchResult:
    """Immutable summary of a deterministic mode-switch evaluation."""

    current_mode: str
    candidate_mode: str
    switch_allowed: bool
    dwell_guard_passed: bool
    hysteresis_passed: bool
    fail_safe_latched: bool
    new_mode: str
    reason_code: str


@dataclass(frozen=True)
class FailSafeState:
    """Explicit bounded state for fail-safe latch tracking.

    All fields are immutable.  A new instance is returned on every
    state transition — no hidden mutation.
    """

    blocked_count: int
    latched: bool
    recent_modes: Tuple[str, ...]  # bounded ring of last N modes

    #: Ring buffer capacity — only the last N modes are retained.
    _RING_CAP: ClassVar[int] = 6

    @staticmethod
    def initial() -> "FailSafeState":
        return FailSafeState(blocked_count=0, latched=False, recent_modes=())

    def record_blocked(self, candidate_mode: str) -> "FailSafeState":
        """Record a blocked switch attempt; latch if threshold exceeded."""
        new_count = self.blocked_count + 1
        new_ring = (self.recent_modes + (candidate_mode,))[-self._RING_CAP :]
        should_latch = self.latched or new_count >= MAX_BLOCKED_BEFORE_LATCH
        # Also latch on A-B-A-B chatter pattern (length >= 4)
        if not should_latch and len(new_ring) >= 4:
            should_latch = _detect_chatter(new_ring)
        return FailSafeState(
            blocked_count=new_count,
            latched=should_latch,
            recent_modes=new_ring,
        )

    def record_success(self) -> "FailSafeState":
        """Reset blocked counter on a successful switch."""
        return FailSafeState(
            blocked_count=0,
            latched=self.latched,  # latch is permanent until explicit reset
            recent_modes=(),
        )

    def reset(self) -> "FailSafeState":
        """Explicit full reset — only for external supervisory override."""
        return FailSafeState.initial()


def _detect_chatter(modes: Tuple[str, ...]) -> bool:
    """Detect alternating A↔B chatter in the recent mode ring.

    Returns True if the last 4+ entries alternate between exactly
    two distinct values (e.g. A-B-A-B).
    """
    if len(modes) < 4:
        return False
    tail: Tuple[str, ...] = modes[-4:]
    # Alternating iff positions 0,2 match and positions 1,3 match
    # and the two groups differ.
    return (
        tail[0] == tail[2]
        and tail[1] == tail[3]
        and tail[0] != tail[1]
    )


# ---------------------------------------------------------------------------
# Primary evaluation function
# ---------------------------------------------------------------------------


def evaluate_mode_switch(
    current_mode: str,
    candidate_mode: str,
    last_switch_time_ms: int,
    current_time_ms: int,
    dwell_time_ms: int,
    metric_value: float,
    enter_threshold: float,
    exit_threshold: float,
    fail_safe: FailSafeState | None = None,
) -> Tuple[ModeSwitchResult, FailSafeState]:
    """Atomically evaluate a proposed supervisory mode switch.

    Parameters
    ----------
    current_mode / candidate_mode:
        The present and proposed supervisory modes.
    last_switch_time_ms / current_time_ms / dwell_time_ms:
        Integer monotonic timestamps and minimum dwell (ms).
    metric_value:
        The observed metric driving the transition.
    enter_threshold / exit_threshold:
        Hysteresis band boundaries.
    fail_safe:
        Explicit fail-safe tracking state.  Pass ``None`` on the
        first call to initialise automatically.

    Returns
    -------
    (ModeSwitchResult, FailSafeState)
        A frozen summary and the updated fail-safe state.
    """
    if fail_safe is None:
        fail_safe = FailSafeState.initial()

    # 1. Fail-safe latch check — overrides everything.
    if fail_safe.latched:
        return (
            ModeSwitchResult(
                current_mode=current_mode,
                candidate_mode=candidate_mode,
                switch_allowed=False,
                dwell_guard_passed=False,
                hysteresis_passed=False,
                fail_safe_latched=True,
                new_mode=current_mode,
                reason_code="fail_safe_latched",
            ),
            fail_safe,
        )

    # 2. No-op when modes are identical.
    if current_mode == candidate_mode:
        return (
            ModeSwitchResult(
                current_mode=current_mode,
                candidate_mode=candidate_mode,
                switch_allowed=False,
                dwell_guard_passed=True,
                hysteresis_passed=True,
                fail_safe_latched=False,
                new_mode=current_mode,
                reason_code="no_change",
            ),
            fail_safe,
        )

    # 3. Dwell-time guard.
    dwell_ok = can_switch_now(last_switch_time_ms, current_time_ms, dwell_time_ms)
    if not dwell_ok:
        fail_safe = fail_safe.record_blocked(candidate_mode)
        return (
            ModeSwitchResult(
                current_mode=current_mode,
                candidate_mode=candidate_mode,
                switch_allowed=False,
                dwell_guard_passed=False,
                hysteresis_passed=False,
                fail_safe_latched=fail_safe.latched,
                new_mode=current_mode,
                reason_code="fail_safe_latched" if fail_safe.latched else "dwell_blocked",
            ),
            fail_safe,
        )

    # 4. Hysteresis filter.
    hyst_ok = passes_hysteresis(
        current_metric=metric_value,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        current_mode="inactive",  # evaluating entry into candidate
    )
    if not hyst_ok:
        fail_safe = fail_safe.record_blocked(candidate_mode)
        return (
            ModeSwitchResult(
                current_mode=current_mode,
                candidate_mode=candidate_mode,
                switch_allowed=False,
                dwell_guard_passed=True,
                hysteresis_passed=False,
                fail_safe_latched=fail_safe.latched,
                new_mode=current_mode,
                reason_code="fail_safe_latched" if fail_safe.latched else "hysteresis_blocked",
            ),
            fail_safe,
        )

    # 5. All guards passed — switch allowed.
    fail_safe = fail_safe.record_success()
    return (
        ModeSwitchResult(
            current_mode=current_mode,
            candidate_mode=candidate_mode,
            switch_allowed=True,
            dwell_guard_passed=True,
            hysteresis_passed=True,
            fail_safe_latched=False,
            new_mode=candidate_mode,
            reason_code="switched",
        ),
        fail_safe,
    )
