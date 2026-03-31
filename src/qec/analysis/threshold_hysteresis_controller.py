"""v125.0.0 — Deterministic threshold hysteresis controller."""

from __future__ import annotations

from dataclasses import dataclass

BANDS = (
    "nominal",
    "warning",
    "critical",
    "recovery",
)

TRANSITION_LABELS = {
    ("nominal", "nominal"): "remain_nominal",
    ("nominal", "warning"): "nominal_to_warning",
    ("warning", "warning"): "remain_warning",
    ("warning", "critical"): "warning_to_critical",
    ("warning", "nominal"): "warning_to_nominal",
    ("critical", "critical"): "remain_critical",
    ("critical", "warning"): "critical_to_warning",
    ("critical", "recovery"): "critical_to_recovery",
    ("recovery", "recovery"): "remain_recovery",
    ("recovery", "nominal"): "recovery_to_nominal",
}


@dataclass(frozen=True)
class HysteresisState:
    current_band: str
    warning_counter: int
    critical_counter: int
    recovery_counter: int


class ThresholdHysteresisController:
    """Deterministic threshold + hysteresis controller with explicit counters."""

    def __init__(self) -> None:
        self._nominal_threshold = 0.40
        self._critical_threshold = 0.70
        self._critical_recovery_threshold = 0.50
        self._recovery_nominal_threshold = 0.30
        self._persistence = 2

    def classify_band(self, score: float) -> str:
        """Classify a score into threshold bands without hysteresis."""
        if score < self._nominal_threshold:
            return "nominal"
        if score < self._critical_threshold:
            return "warning"
        return "critical"

    def apply_hysteresis(self, state: HysteresisState, score: float) -> str:
        """Apply deterministic hysteresis persistence rules and return next stable band."""
        if state.current_band not in BANDS:
            raise ValueError(f"unknown band: {state.current_band}")

        observed_band = self.classify_band(score)

        if state.current_band == "nominal":
            if observed_band == "warning" and state.warning_counter + 1 >= self._persistence:
                return "warning"
            return "nominal"

        if state.current_band == "warning":
            if observed_band == "critical" and state.critical_counter + 1 >= self._persistence:
                return "critical"
            if observed_band == "nominal" and state.recovery_counter + 1 >= self._persistence:
                return "nominal"
            return "warning"

        if state.current_band == "critical":
            if score < self._critical_recovery_threshold and state.recovery_counter + 1 >= self._persistence:
                return "recovery"
            if observed_band == "warning" and state.warning_counter + 1 >= self._persistence:
                return "warning"
            return "critical"

        if score < self._recovery_nominal_threshold and state.recovery_counter + 1 >= self._persistence:
            return "nominal"
        return "recovery"

    def next_state(self, state: HysteresisState, score: float) -> HysteresisState:
        """Return deterministic next hysteresis state with stale-counter reset ownership."""
        observed_band = self.classify_band(score)
        next_band = self.apply_hysteresis(state, score)

        warning_counter = 0
        critical_counter = 0
        recovery_counter = 0

        if next_band == "nominal":
            if state.current_band == "nominal" and observed_band == "warning":
                warning_counter = state.warning_counter + 1
        elif next_band == "warning":
            if state.current_band == "warning":
                if observed_band == "critical":
                    critical_counter = state.critical_counter + 1
                elif observed_band == "nominal":
                    recovery_counter = state.recovery_counter + 1
        elif next_band == "critical":
            if state.current_band == "critical":
                if score < self._critical_recovery_threshold:
                    recovery_counter = state.recovery_counter + 1
                elif observed_band == "warning":
                    warning_counter = state.warning_counter + 1
        elif next_band == "recovery":
            if score < self._recovery_nominal_threshold:
                recovery_counter = state.recovery_counter + 1

        return HysteresisState(
            current_band=next_band,
            warning_counter=warning_counter,
            critical_counter=critical_counter,
            recovery_counter=recovery_counter,
        )

    def step(self, state: HysteresisState, score: float) -> dict:
        """Compute deterministic transition metadata for one control cycle."""
        previous_band = state.current_band
        observed_band = self.classify_band(score)
        next_band = self.apply_hysteresis(state, score)

        return {
            "previous_band": previous_band,
            "next_band": next_band,
            "band_transition": TRANSITION_LABELS.get((previous_band, next_band), f"{previous_band}_to_{next_band}"),
            "threshold_crossed": observed_band != previous_band,
            "hysteresis_active": observed_band != previous_band and next_band == previous_band,
        }
