"""
v80.0.0 — QEC FSM Controller (Deterministic Experiment Orchestrator).

A finite state machine that orchestrates experiment execution through
invariant-aware transitions.  Tracks system state across iterations and
enables reproducible experiment cycles.

FSM State     QEC Analog
----------    --------------------
ANALYZE       syndrome extraction
PERTURB       noise probing
INVARIANT     decoding insight
EVALUATE      correction decision
ACCEPT        logical state preserved
REJECT        error detected

Layer 8 — Controller.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from qec.experiments.perturbation_probe import run_perturbation_probe
from qec.experiments.invariant_engine import run_invariant_analysis


# ---------------------------------------------------------------------------
# FSM States
# ---------------------------------------------------------------------------

INIT = "INIT"
ANALYZE = "ANALYZE"
PERTURB = "PERTURB"
INVARIANT = "INVARIANT"
EVALUATE = "EVALUATE"
ACCEPT = "ACCEPT"
REJECT = "REJECT"
TERMINATE = "TERMINATE"

_VALID_STATES = frozenset({
    INIT, ANALYZE, PERTURB, INVARIANT,
    EVALUATE, ACCEPT, REJECT, TERMINATE,
})

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "stability_threshold": 0.5,
    "boundary_crossing_threshold": 2,
    "max_reject_cycles": 3,
    "epsilon": 1e-3,
    "n_perturbations": 9,
    "drift_threshold": 1e-4,
}


# ---------------------------------------------------------------------------
# QEC FSM Controller
# ---------------------------------------------------------------------------

class QECFSM:
    """Deterministic finite state machine for experiment orchestration.

    Parameters
    ----------
    config : dict, optional
        Configuration overrides.  Missing keys fall back to defaults.

    Attributes
    ----------
    state : str
        Current FSM state.
    history : list[dict]
        Full trace of state transitions and associated data.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        merged = dict(_DEFAULT_CONFIG)
        if config is not None:
            merged.update(config)
        self._config: Dict[str, Any] = merged
        self.state: str = INIT
        self.history: List[Dict[str, Any]] = []
        self._reject_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, state: str, data: dict) -> Tuple[str, dict]:
        """Execute a single FSM transition.

        Parameters
        ----------
        state : str
            Current state.
        data : dict
            Mutable working data dict (deep-copied internally where needed).

        Returns
        -------
        tuple[str, dict]
            (next_state, updated_data)

        Raises
        ------
        ValueError
            If *state* is not a valid FSM state.
        """
        if state not in _VALID_STATES:
            raise ValueError(f"Invalid FSM state: {state!r}")

        handler = {
            INIT: self._step_init,
            ANALYZE: self._step_analyze,
            PERTURB: self._step_perturb,
            INVARIANT: self._step_invariant,
            EVALUATE: self._step_evaluate,
            ACCEPT: self._step_accept,
            REJECT: self._step_reject,
            TERMINATE: self._step_terminate,
        }[state]

        next_state, updated = handler(data)

        self.history.append({
            "from_state": state,
            "to_state": next_state,
            "stability_score": updated.get("stability_score"),
            "phase": updated.get("phase"),
        })

        return next_state, updated

    def run(self, data: dict, *, max_steps: int = 10) -> dict:
        """Run the FSM to completion or until *max_steps* is reached.

        Parameters
        ----------
        data : dict
            Initial input data.  Deep-copied; the original is never mutated.
        max_steps : int
            Hard upper bound on transition count (default ``10``).

        Returns
        -------
        dict
            Structured result with ``final_state``, ``steps``, and ``history``.
        """
        working = copy.deepcopy(data)
        self.state = INIT
        self.history = []
        self._reject_count = 0
        steps = 0

        while self.state != TERMINATE and steps < max_steps:
            self.state, working = self.step(self.state, working)
            steps += 1

        # If we ran out of steps without terminating, force terminate.
        if self.state != TERMINATE:
            self.history.append({
                "from_state": self.state,
                "to_state": TERMINATE,
                "stability_score": working.get("stability_score"),
                "phase": working.get("phase"),
                "reason": "max_steps_reached",
            })
            self.state = TERMINATE

        return {
            "final_state": working.get("verdict", TERMINATE),
            "steps": steps,
            "history": list(self.history),
        }

    # ------------------------------------------------------------------
    # State handlers (private)
    # ------------------------------------------------------------------

    def _step_init(self, data: dict) -> Tuple[str, dict]:
        """INIT → ANALYZE: validate and prepare input."""
        out = dict(data)
        out.setdefault("iteration", 0)
        return ANALYZE, out

    def _step_analyze(self, data: dict) -> Tuple[str, dict]:
        """ANALYZE → PERTURB: build analysis result from input features."""
        out = dict(data)
        # Build a sonic-analysis-like result dict from input features.
        out["analysis_result"] = _build_analysis_result(out)
        return PERTURB, out

    def _step_perturb(self, data: dict) -> Tuple[str, dict]:
        """PERTURB → INVARIANT: run perturbation probe."""
        out = dict(data)
        result = out.get("analysis_result", {})
        probe = run_perturbation_probe(
            result,
            epsilon=self._config["epsilon"],
            n=self._config["n_perturbations"],
        )
        out["probe"] = probe
        return INVARIANT, out

    def _step_invariant(self, data: dict) -> Tuple[str, dict]:
        """INVARIANT → EVALUATE: run invariant analysis."""
        out = dict(data)
        probe = out.get("probe", {})
        analysis = run_invariant_analysis(
            probe,
            drift_threshold=self._config["drift_threshold"],
        )
        out["stability_score"] = analysis["stability_score"]
        out["phase"] = analysis["phase"]
        out["invariants"] = analysis["invariants"]
        out["feature_ranking"] = analysis["feature_ranking"]
        return EVALUATE, out

    def _step_evaluate(self, data: dict) -> Tuple[str, dict]:
        """EVALUATE → ACCEPT | REJECT | ANALYZE based on invariant results."""
        out = dict(data)
        score = out.get("stability_score", float("inf"))
        phase = out.get("phase", "")
        probe = out.get("probe", {})
        crossings = probe.get("boundary_crossings", 0)

        threshold = self._config["stability_threshold"]
        crossing_threshold = self._config["boundary_crossing_threshold"]

        if score < threshold and phase == "stable_region":
            return ACCEPT, out
        elif crossings >= crossing_threshold:
            return REJECT, out
        else:
            # Continue loop — re-analyze with incremented iteration.
            out["iteration"] = out.get("iteration", 0) + 1
            return ANALYZE, out

    def _step_accept(self, data: dict) -> Tuple[str, dict]:
        """ACCEPT → TERMINATE: mark result as stable."""
        out = dict(data)
        out["verdict"] = ACCEPT
        return TERMINATE, out

    def _step_reject(self, data: dict) -> Tuple[str, dict]:
        """REJECT → ANALYZE | TERMINATE: mark unstable, optionally retry."""
        out = dict(data)
        self._reject_count += 1
        max_rejects = self._config["max_reject_cycles"]

        if self._reject_count >= max_rejects:
            out["verdict"] = REJECT
            return TERMINATE, out

        # Deterministic adjustment: tighten epsilon for next cycle.
        self._config["epsilon"] = self._config["epsilon"] * 0.5
        out["iteration"] = out.get("iteration", 0) + 1
        return ANALYZE, out

    def _step_terminate(self, data: dict) -> Tuple[str, dict]:
        """TERMINATE → TERMINATE (absorbing state)."""
        out = dict(data)
        out.setdefault("verdict", TERMINATE)
        return TERMINATE, out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_analysis_result(data: dict) -> dict:
    """Build a sonic-analysis-compatible result dict from FSM input data.

    If *data* already contains a ``"result"`` key, it is used directly.
    Otherwise, feature values are extracted from *data* with defaults.
    """
    if "result" in data:
        return copy.deepcopy(data["result"])

    return {
        "rms_energy": float(data.get("rms_energy", 0.01)),
        "spectral_centroid_hz": float(data.get("spectral_centroid_hz", 500.0)),
        "spectral_spread_hz": float(data.get("spectral_spread_hz", 200.0)),
        "zero_crossing_rate": float(data.get("zero_crossing_rate", 0.05)),
        "fft_top_peaks": list(data.get("fft_top_peaks", [
            {"frequency_hz": 100.0, "magnitude": 0.5},
            {"frequency_hz": 200.0, "magnitude": 0.3},
        ])),
    }
