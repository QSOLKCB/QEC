from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.governance_stability_validation import (
    GovernanceStabilityReceipt,
    GovernanceStabilityScenario,
    validate_governance_stability,
)


def _h(label: str) -> str:
    return sha256_hex(label)


def _scenario(
    *,
    scenario_id: str,
    mem: tuple[str, ...],
    decisions: tuple[str, ...],
    selected: str,
) -> GovernanceStabilityScenario:
    return GovernanceStabilityScenario(
        scenario_id=scenario_id,
        input_memory_hashes=mem,
        decision_hashes=decisions,
        selected_decision_hash=selected,
    )


def test_non_sequence_input_rejected() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_governance_stability(123)  # type: ignore[arg-type]


def test_empty_scenarios_rejected() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_governance_stability(())


def test_non_scenario_element_rejected() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_governance_stability(("not-a-scenario",))  # type: ignore[arg-type]


def test_deterministic_replay_100_runs() -> None:
    mem = (_h("m1"), _h("m2"))
    decisions = (_h("d1"), _h("d2"), _h("d3"))
    scenarios = (
        _scenario(scenario_id="s2", mem=mem, decisions=decisions, selected=decisions[1]),
        _scenario(scenario_id="s1", mem=mem, decisions=decisions, selected=decisions[1]),
    )
    baseline = validate_governance_stability(scenarios)
    for _ in range(100):
        replay = validate_governance_stability(scenarios)
        assert replay == baseline
        assert replay.stability_hash == baseline.stability_hash


def test_identical_replay_stability() -> None:
    mem = (_h("m1"),)
    decisions = (_h("d1"), _h("d2"))
    scenarios = (
        _scenario(scenario_id="same", mem=mem, decisions=decisions, selected=decisions[0]),
    )
    receipt = validate_governance_stability(scenarios)
    assert receipt.result.stable is True
    assert receipt.result.stability_score == 1.0


def test_reordered_input_stability() -> None:
    mem = (_h("m1"), _h("m2"))
    decisions = (_h("d1"), _h("d2"))
    left = (
        _scenario(scenario_id="b", mem=mem, decisions=decisions, selected=decisions[1]),
        _scenario(scenario_id="a", mem=mem, decisions=decisions, selected=decisions[1]),
    )
    right = tuple(reversed(left))
    assert validate_governance_stability(left) == validate_governance_stability(right)


def test_equivalent_decision_set_stability() -> None:
    mem = (_h("m1"), _h("m2"))
    unordered = (_h("d2"), _h("d1"), _h("d1"))
    canonical = tuple(sorted(set(unordered)))
    scenarios = (
        _scenario(scenario_id="s1", mem=mem, decisions=unordered, selected=canonical[0]),
        _scenario(scenario_id="s2", mem=mem, decisions=canonical, selected=canonical[0]),
    )
    receipt = validate_governance_stability(scenarios)
    assert receipt.result.stable is True
    assert receipt.result.scenario_count == 2
    assert receipt.result.perturbation_count == 1


def test_instability_rejected() -> None:
    mem = (_h("m1"),)
    decisions = (_h("d1"), _h("d2"))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_governance_stability(
            (
                _scenario(scenario_id="s1", mem=mem, decisions=decisions, selected=decisions[0]),
                _scenario(scenario_id="s2", mem=mem, decisions=decisions, selected=decisions[1]),
            )
        )


def test_context_mismatch_rejected() -> None:
    decisions = (_h("d1"),)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_governance_stability(
            (
                _scenario(scenario_id="s1", mem=(_h("m1"),), decisions=decisions, selected=decisions[0]),
                _scenario(scenario_id="s2", mem=(_h("m2"),), decisions=decisions, selected=decisions[0]),
            )
        )


def test_decision_mismatch_rejected() -> None:
    mem = (_h("m1"),)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_governance_stability(
            (
                _scenario(scenario_id="s1", mem=mem, decisions=(_h("d1"),), selected=_h("d1")),
                _scenario(scenario_id="s2", mem=mem, decisions=(_h("d2"),), selected=_h("d2")),
            )
        )


def test_duplicate_scenario_id_rejected() -> None:
    mem = (_h("m1"),)
    decisions = (_h("d1"),)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_governance_stability(
            (
                _scenario(scenario_id="dup", mem=mem, decisions=decisions, selected=decisions[0]),
                _scenario(scenario_id="dup", mem=mem, decisions=decisions, selected=decisions[0]),
            )
        )


def test_invalid_selected_hash_rejected() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _scenario(
            scenario_id="s1",
            mem=(_h("m1"),),
            decisions=(_h("d1"),),
            selected="abc",
        )


def test_selected_not_in_decision_set_rejected() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _scenario(
            scenario_id="s1",
            mem=(_h("m1"),),
            decisions=(_h("d1"),),
            selected=_h("other"),
        )


def test_hash_recomputation_stability() -> None:
    mem = (_h("m1"),)
    decisions = (_h("d1"),)
    receipt = validate_governance_stability(
        (_scenario(scenario_id="s1", mem=mem, decisions=decisions, selected=decisions[0]),)
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        GovernanceStabilityReceipt(
            scenarios=receipt.scenarios,
            result=receipt.result,
            stability_hash="0" * 64,
        )


def test_immutability() -> None:
    mem = (_h("m1"),)
    decisions = (_h("d1"),)
    receipt = validate_governance_stability(
        (_scenario(scenario_id="s1", mem=mem, decisions=decisions, selected=decisions[0]),)
    )
    with pytest.raises(FrozenInstanceError):
        receipt.stability_hash = "0" * 64
