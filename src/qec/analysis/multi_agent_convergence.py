"""Deterministic multi-agent convergence analysis for arbitration rounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.canonical_identity import _require_sha256_hex, canonical_hash_identity


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


@dataclass(frozen=True)
class ConvergenceRound:
    round_index: int
    decision_hashes: tuple[str, ...]
    selected_decision_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int) or self.round_index < 0:
            raise _invalid_input()
        decision_hashes = canonical_hash_identity(self.decision_hashes)
        selected_decision_hash = _require_sha256_hex(self.selected_decision_hash)
        if selected_decision_hash not in decision_hashes:
            raise _invalid_input()
        object.__setattr__(self, "decision_hashes", decision_hashes)
        object.__setattr__(self, "selected_decision_hash", selected_decision_hash)


@dataclass(frozen=True)
class ConvergenceState:
    rounds: tuple[ConvergenceRound, ...]
    disagreement_count: int
    arbitration_stability: float
    convergence_depth: int
    converged: bool

    def __post_init__(self) -> None:
        if not isinstance(self.rounds, tuple) or len(self.rounds) == 0:
            raise _invalid_input()
        previous_index: int | None = None
        for round_state in self.rounds:
            if not isinstance(round_state, ConvergenceRound):
                raise _invalid_input()
            if previous_index is not None and round_state.round_index <= previous_index:
                raise _invalid_input()
            previous_index = round_state.round_index

        if isinstance(self.disagreement_count, bool) or not isinstance(self.disagreement_count, int) or self.disagreement_count < 0:
            raise _invalid_input()
        if not isinstance(self.arbitration_stability, float) or not (0.0 <= self.arbitration_stability <= 1.0):
            raise _invalid_input()
        if isinstance(self.convergence_depth, bool) or not isinstance(self.convergence_depth, int) or self.convergence_depth < 0:
            raise _invalid_input()
        if not isinstance(self.converged, bool):
            raise _invalid_input()

        selected = tuple(round_state.selected_decision_hash for round_state in self.rounds)
        if len(selected) == 1:
            expected_converged = True
        else:
            expected_converged = selected[-1] == selected[-2]
        if self.converged and not expected_converged:
            raise _invalid_input()


@dataclass(frozen=True)
class ConvergenceReceipt:
    convergence_state: ConvergenceState
    input_memory_hashes: tuple[str, ...]
    protocol_hashes: tuple[str, ...]
    convergence_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.convergence_state, ConvergenceState):
            raise _invalid_input()
        input_memory_hashes = canonical_hash_identity(self.input_memory_hashes)
        protocol_hashes = canonical_hash_identity(self.protocol_hashes)
        object.__setattr__(self, "input_memory_hashes", input_memory_hashes)
        object.__setattr__(self, "protocol_hashes", protocol_hashes)
        expected_hash = _convergence_hash(
            convergence_state=self.convergence_state,
            input_memory_hashes=input_memory_hashes,
            protocol_hashes=protocol_hashes,
        )
        if self.convergence_hash != expected_hash:
            raise _invalid_input()


def _convergence_hash(*, convergence_state: ConvergenceState, input_memory_hashes: tuple[str, ...], protocol_hashes: tuple[str, ...]) -> str:
    return sha256_hex(
        {
            "convergence_state": {
                "rounds": [
                    {
                        "round_index": round_state.round_index,
                        "decision_hashes": round_state.decision_hashes,
                        "selected_decision_hash": round_state.selected_decision_hash,
                    }
                    for round_state in convergence_state.rounds
                ],
                "disagreement_count": convergence_state.disagreement_count,
                "arbitration_stability": convergence_state.arbitration_stability,
                "convergence_depth": convergence_state.convergence_depth,
                "converged": convergence_state.converged,
            },
            "input_memory_hashes": input_memory_hashes,
            "protocol_hashes": protocol_hashes,
        }
    )


def analyze_multi_agent_convergence(
    input_memory_hashes: tuple[str, ...],
    protocol_hashes: tuple[str, ...],
    rounds: Sequence[ConvergenceRound],
) -> ConvergenceReceipt:
    identity = canonical_hash_identity(input_memory_hashes)
    protocols = canonical_hash_identity(protocol_hashes)
    if isinstance(rounds, tuple):
        round_tuple = rounds
    else:
        round_tuple = tuple(rounds)
    if len(round_tuple) == 0 or any(not isinstance(round_state, ConvergenceRound) for round_state in round_tuple):
        raise _invalid_input()

    selected_hashes = tuple(round_state.selected_decision_hash for round_state in round_tuple)
    if len(round_tuple) > 1 and selected_hashes[-1] != selected_hashes[-2]:
        raise _invalid_input()

    if len(round_tuple) == 1:
        convergence_depth = 0
        arbitration_stability = 1.0
        converged = True
    else:
        final_selected = selected_hashes[-1]
        convergence_depth = -1
        for idx in range(len(selected_hashes)):
            if selected_hashes[idx] == final_selected and all(candidate == final_selected for candidate in selected_hashes[idx:]):
                convergence_depth = idx
                break
        if convergence_depth < 0:
            raise _invalid_input()
        total_transitions = len(selected_hashes) - 1
        stable_transitions = sum(
            1 for idx in range(total_transitions) if selected_hashes[idx] == selected_hashes[idx + 1]
        )
        arbitration_stability = round(stable_transitions / total_transitions, 12)
        converged = True

    final_round = round_tuple[-1]
    convergence_state = ConvergenceState(
        rounds=round_tuple,
        disagreement_count=len(final_round.decision_hashes),
        arbitration_stability=arbitration_stability,
        convergence_depth=convergence_depth,
        converged=converged,
    )
    convergence_hash = _convergence_hash(
        convergence_state=convergence_state,
        input_memory_hashes=identity,
        protocol_hashes=protocols,
    )
    return ConvergenceReceipt(
        convergence_state=convergence_state,
        input_memory_hashes=identity,
        protocol_hashes=protocols,
        convergence_hash=convergence_hash,
    )


__all__ = [
    "ConvergenceRound",
    "ConvergenceState",
    "ConvergenceReceipt",
    "analyze_multi_agent_convergence",
]
