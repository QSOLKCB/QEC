from __future__ import annotations

import numpy as np

from qec.discovery.adaptive_mutation_controller import (
    AdaptiveMutationConfig,
    AdaptiveMutationController,
)


class _StubDiagnostics:
    def __init__(self, states: list[str]) -> None:
        self._states = list(states)
        self._index = 0

    def classify_state(self, _H: np.ndarray) -> dict[str, str]:
        state = self._states[min(self._index, len(self._states) - 1)]
        self._index += 1
        return {"basin_state": state}


class _FlowStub:
    def __init__(self) -> None:
        self.calls = 0

    def mutate(self, H: np.ndarray, *, iterations: int) -> tuple[np.ndarray, list[dict[str, int]]]:
        self.calls += 1
        out = H.copy()
        out[0, 0] = 0.0
        out[1, 1] = 0.0
        out[0, 1] = 1.0
        out[1, 0] = 1.0
        return out, [{"flow_mode_index": iterations}]


class _BeamStub:
    def __init__(self) -> None:
        self.calls = 0
        self.beam_width = 0

    def mutate(self, H: np.ndarray, *, steps: int) -> tuple[np.ndarray, list[dict[str, int]]]:
        self.calls += 1
        out = H.copy()
        out[0, 2] = 0.0
        out[1, 0] = 0.0
        out[0, 0] = 1.0
        out[1, 2] = 1.0
        return out, [{"flow_mode_index": steps}]


def _matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_strategy_selection_and_metadata() -> None:
    flow = _FlowStub()
    beam = _BeamStub()
    controller = AdaptiveMutationController(
        config=AdaptiveMutationConfig(
            enabled=True,
            flow_iterations=2,
            beam_iterations=3,
            trap_beam_width=8,
        ),
        basin_diagnostics=_StubDiagnostics(["free_descent", "localized_trap", "converged"]),
        flow_mutator=flow,
        beam_mutator=beam,
    )

    H_out, trajectory = controller.mutate(_matrix(), max_iterations=3)

    assert flow.calls == 1
    assert beam.calls == 1
    assert H_out.shape == _matrix().shape
    assert trajectory[0]["mutation_strategy"] == "nb_flow"
    assert trajectory[1]["mutation_strategy"] == "beam_search"
    assert trajectory[1]["beam_width"] == 8
    assert trajectory[2]["mutation_strategy"] == "terminated"


def test_determinism_repeated_runs_identical() -> None:
    cfg = AdaptiveMutationConfig(enabled=True, flow_iterations=1, beam_iterations=1)
    states = ["localized_trap", "converged"]

    c1 = AdaptiveMutationController(
        config=cfg,
        basin_diagnostics=_StubDiagnostics(states),
        flow_mutator=_FlowStub(),
        beam_mutator=_BeamStub(),
    )
    c2 = AdaptiveMutationController(
        config=cfg,
        basin_diagnostics=_StubDiagnostics(states),
        flow_mutator=_FlowStub(),
        beam_mutator=_BeamStub(),
    )

    r1 = c1.mutate(_matrix(), max_iterations=2)
    r2 = c2.mutate(_matrix(), max_iterations=2)

    assert np.array_equal(r1[0], r2[0])
    assert r1[1] == r2[1]


def test_trap_escape_activates_beam_search() -> None:
    beam = _BeamStub()
    controller = AdaptiveMutationController(
        config=AdaptiveMutationConfig(enabled=True, trap_beam_width=9),
        basin_diagnostics=_StubDiagnostics(["localized_trap"]),
        flow_mutator=_FlowStub(),
        beam_mutator=beam,
    )

    _, trajectory = controller.mutate(_matrix(), max_iterations=1)

    assert beam.calls == 1
    assert beam.beam_width == 9
    assert trajectory[0]["mutation_strategy"] == "beam_search"


def test_degree_preservation_and_binary_with_stub_flow() -> None:
    H = _matrix()
    controller = AdaptiveMutationController(
        config=AdaptiveMutationConfig(enabled=True, flow_iterations=1),
        basin_diagnostics=_StubDiagnostics(["free_descent"]),
        flow_mutator=_FlowStub(),
        beam_mutator=_BeamStub(),
    )

    out, trajectory = controller.mutate(H, max_iterations=1)

    assert trajectory
    assert np.all((out == 0.0) | (out == 1.0))
    assert np.array_equal(np.sum(H, axis=0), np.sum(out, axis=0))
    assert np.array_equal(np.sum(H, axis=1), np.sum(out, axis=1))
