"""v122.0.0 — Deterministic hybrid automata layer."""

from __future__ import annotations

from dataclasses import dataclass

MODE_NOMINAL = "nominal"
MODE_WARNING = "warning"
MODE_CRITICAL = "critical"
MODE_RECOVERY = "recovery"


@dataclass(frozen=True)
class HybridState:
    """Immutable hybrid state with discrete and continuous components."""

    mode: str
    continuous_value: float
    drift_rate: float
    risk_score: float


class HybridAutomata:
    """Minimal deterministic coupling of discrete mode and continuous metric."""

    def __init__(self) -> None:
        self._valid_modes = (
            MODE_NOMINAL,
            MODE_WARNING,
            MODE_CRITICAL,
            MODE_RECOVERY,
        )

    def transition_mode(self, state: HybridState) -> str:
        """Deterministically classify the next mode from continuous value."""
        self._validate_mode(state.mode)

        value = float(state.continuous_value)
        if state.mode == MODE_CRITICAL and value < 0.5:
            return MODE_RECOVERY
        if value < 0.3:
            return MODE_NOMINAL
        if value < 0.7:
            return MODE_WARNING
        return MODE_CRITICAL

    def update_continuous(self, state: HybridState) -> float:
        """Deterministically advance and clamp the continuous value."""
        next_value = (
            float(state.continuous_value)
            + float(state.drift_rate)
            + 0.25 * float(state.risk_score)
        )
        return self._clamp01(next_value)

    def step(self, state: HybridState) -> dict:
        """Execute one deterministic hybrid transition."""
        self._validate_mode(state.mode)

        previous_mode = state.mode
        previous_value = float(state.continuous_value)
        next_value = self.update_continuous(state)

        if (
            previous_mode == MODE_CRITICAL
            and previous_value >= 0.5
            and next_value < 0.5
        ):
            next_mode = MODE_RECOVERY
        else:
            next_mode = self.transition_mode(
                HybridState(
                    mode=previous_mode,
                    continuous_value=next_value,
                    drift_rate=state.drift_rate,
                    risk_score=state.risk_score,
                )
            )

        return {
            "previous_mode": previous_mode,
            "next_mode": next_mode,
            "continuous_value": next_value,
            "mode_transition": self._transition_label(previous_mode, next_mode),
            "hybrid_stable": previous_mode == next_mode,
        }

    def _validate_mode(self, mode: str) -> None:
        if mode not in self._valid_modes:
            raise ValueError(f"unknown mode: {mode}")

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _transition_label(previous_mode: str, next_mode: str) -> str:
        transition_labels = {
            (MODE_NOMINAL, MODE_NOMINAL): "remain_nominal",
            (MODE_NOMINAL, MODE_WARNING): "nominal_to_warning",
            (MODE_NOMINAL, MODE_CRITICAL): "nominal_to_critical",
            (MODE_WARNING, MODE_NOMINAL): "warning_to_nominal",
            (MODE_WARNING, MODE_WARNING): "remain_warning",
            (MODE_WARNING, MODE_CRITICAL): "warning_to_critical",
            (MODE_CRITICAL, MODE_CRITICAL): "remain_critical",
            (MODE_CRITICAL, MODE_RECOVERY): "critical_to_recovery",
            (MODE_RECOVERY, MODE_NOMINAL): "recovery_to_nominal",
            (MODE_RECOVERY, MODE_WARNING): "recovery_to_warning",
            (MODE_RECOVERY, MODE_CRITICAL): "recovery_to_critical",
            (MODE_RECOVERY, MODE_RECOVERY): "remain_recovery",
            (MODE_CRITICAL, MODE_WARNING): "critical_to_warning",
            (MODE_CRITICAL, MODE_NOMINAL): "critical_to_nominal",
            (MODE_WARNING, MODE_RECOVERY): "warning_to_recovery",
            (MODE_NOMINAL, MODE_RECOVERY): "nominal_to_recovery",
        }
        label = transition_labels.get((previous_mode, next_mode))
        if label is None:
            raise ValueError(
                f"unsupported mode transition: ({previous_mode}, {next_mode})"
            )
        return label
