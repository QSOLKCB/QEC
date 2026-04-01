"""v132.3.0 — Deterministic dwell-time and mode switching layer.

Provides deterministic supervisory switching guards:
- Dwell-time guard: prevents rapid mode oscillation
- Hysteresis filter: anti-chatter threshold logic
- Mode switcher: atomic deterministic mode transitions

All logic is pure, deterministic, and clock-explicit.
No system clock reads. No floating point in timing.
No mutation of decoder internals.
"""

from .dwell_time_guard import can_switch_now
from .hysteresis_filter import passes_hysteresis
from .mode_switcher import evaluate_mode_switch, ModeSwitchResult

__all__ = [
    "can_switch_now",
    "passes_hysteresis",
    "evaluate_mode_switch",
    "ModeSwitchResult",
]
