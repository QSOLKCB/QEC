from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.multi_agent_convergence import (
    ConvergenceReceipt,
    ConvergenceRound,
    analyze_multi_agent_convergence,
)


def _h(seed: str) -> str:
    return sha256_hex({"seed": seed})


def _round(idx: int, seeds: tuple[str, ...], selected: str) -> ConvergenceRound:
    decision_hashes = tuple(sorted(_h(seed) for seed in seeds))
    return ConvergenceRound(round_index=idx, decision_hashes=decision_hashes, selected_decision_hash=_h(selected))


def test_deterministic_replay_100_runs_identical_receipt() -> None:
    rounds = (
        _round(0, ("a", "b"), "a"),
        _round(1, ("a", "b"), "a"),
    )
    input_memory_hashes = tuple(sorted((_h("m0"), _h("m1"))))
    protocol_hashes = tuple(sorted((_h("p0"), _h("p1"))))

    receipts = [analyze_multi_agent_convergence(input_memory_hashes, protocol_hashes, rounds) for _ in range(100)]
    assert all(receipt == receipts[0] for receipt in receipts)


def test_one_round_convergence_metrics() -> None:
    receipt = analyze_multi_agent_convergence(
        input_memory_hashes=(_h("m0"),),
        protocol_hashes=(_h("p0"),),
        rounds=(_round(0, ("x", "y"), "x"),),
    )
    assert receipt.convergence_state.converged is True
    assert receipt.convergence_state.convergence_depth == 0
    assert receipt.convergence_state.arbitration_stability == 1.0


def test_multi_round_convergence_final_two_stable() -> None:
    rounds = (
        _round(0, ("a", "b"), "b"),
        _round(1, ("a", "b"), "a"),
        _round(2, ("a", "b"), "a"),
    )
    receipt = analyze_multi_agent_convergence((_h("m0"),), (_h("p0"),), rounds)
    assert receipt.convergence_state.converged is True


def test_non_convergence_when_final_two_differ_fails() -> None:
    rounds = (
        _round(0, ("a", "b"), "a"),
        _round(1, ("a", "b"), "b"),
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        analyze_multi_agent_convergence((_h("m0"),), (_h("p0"),), rounds)


def test_disagreement_count_matches_final_round_size() -> None:
    rounds = (
        _round(0, ("a", "b"), "a"),
        _round(1, ("a", "b", "c"), "a"),
    )
    receipt = analyze_multi_agent_convergence((_h("m0"),), (_h("p0"),), rounds)
    assert receipt.convergence_state.disagreement_count == 3


def test_arbitration_stability_ratio_rounded_12() -> None:
    rounds = (
        _round(0, ("a", "b"), "a"),
        _round(1, ("a", "b"), "a"),
        _round(2, ("a", "b"), "b"),
        _round(3, ("a", "b"), "b"),
        _round(4, ("a", "b"), "b"),
    )
    receipt = analyze_multi_agent_convergence((_h("m0"),), (_h("p0"),), rounds)
    assert receipt.convergence_state.arbitration_stability == 0.75


def test_convergence_depth_first_stable_point() -> None:
    rounds = (
        _round(0, ("a", "b"), "a"),
        _round(1, ("a", "b"), "b"),
        _round(2, ("a", "b"), "b"),
        _round(3, ("a", "b"), "b"),
    )
    receipt = analyze_multi_agent_convergence((_h("m0"),), (_h("p0"),), rounds)
    assert receipt.convergence_state.convergence_depth == 1


@pytest.mark.parametrize(
    "input_memory_hashes,protocol_hashes",
    [
        ((_h("m1"), _h("m0")), (_h("p0"),)),
        ((_h("m0"), _h("m0")), (_h("p0"),)),
        (("bad",), (_h("p0"),)),
        ((_h("m0"),), tuple(sorted((_h("p0"), _h("p1")), reverse=True))),
        ((_h("m0"),), (_h("p0"), _h("p0"))),
        ((_h("m0"),), ("bad",)),
    ],
)
def test_canonical_identity_enforced(input_memory_hashes: tuple[str, ...], protocol_hashes: tuple[str, ...]) -> None:
    rounds = (_round(0, ("a",), "a"),)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        analyze_multi_agent_convergence(input_memory_hashes, protocol_hashes, rounds)


def test_round_validation_failures() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ConvergenceRound(round_index=0, decision_hashes=(_h("a"),), selected_decision_hash=_h("b"))

    round0 = _round(0, ("a",), "a")
    round1 = _round(1, ("a",), "a")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        analyze_multi_agent_convergence((_h("m0"),), (_h("p0"),), (round1, round0))

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        analyze_multi_agent_convergence((_h("m0"),), (_h("p0"),), (round0, replace(round0)))


def test_immutability() -> None:
    receipt = analyze_multi_agent_convergence((_h("m0"),), (_h("p0"),), (_round(0, ("a",), "a"),))
    with pytest.raises(FrozenInstanceError):
        receipt.convergence_state = receipt.convergence_state


def test_hash_stability_recompute_exact() -> None:
    receipt = analyze_multi_agent_convergence((_h("m0"),), (_h("p0"),), (_round(0, ("a",), "a"),))
    recomputed = ConvergenceReceipt(
        convergence_state=receipt.convergence_state,
        input_memory_hashes=receipt.input_memory_hashes,
        protocol_hashes=receipt.protocol_hashes,
        convergence_hash=receipt.convergence_hash,
    )
    assert recomputed.convergence_hash == receipt.convergence_hash
