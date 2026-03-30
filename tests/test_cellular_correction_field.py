"""Deterministic tests for v112.0.0 cellular correction field."""

from __future__ import annotations

import pytest

from qec.analysis.cellular_correction_field import run_cellular_correction_field


REQUIRED_KEYS = {
    "chain_length",
    "dispatch_result",
    "initial_field",
    "corrected_field",
    "field_drift_score",
    "local_stability_score",
    "correction_field_class",
}


def _dispatch_stub(correction_action: str) -> dict[str, object]:
    return {
        "chain_lengths": (9,),
        "spectral_result": {},
        "correction_action": correction_action,
        "dispatch_urgency_score": 0.2,
        "dispatch_cycle_budget": 1,
        "correction_policy_class": "monitor_policy",
        "action_stability_score": 1.0,
    }


def test_exact_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("local_stabilize"),
    )

    out1 = run_cellular_correction_field(chain_length=3, chain_state=[0.0, 0.5, 1.0], automata_steps=2)
    out2 = run_cellular_correction_field(chain_length=3, chain_state=[0.0, 0.5, 1.0], automata_steps=2)

    assert out1 == out2


def test_hold_state_no_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("hold_state"),
    )

    out = run_cellular_correction_field(chain_length=3, chain_state=[0.1, 0.4, 0.9], automata_steps=3)

    assert set(out.keys()) == REQUIRED_KEYS
    assert out["corrected_field"] == out["initial_field"]
    assert out["field_drift_score"] == 0.0
    assert out["local_stability_score"] == 1.0
    assert out["correction_field_class"] == "stable_field"


def test_local_rule_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("local_stabilize"),
    )

    out = run_cellular_correction_field(chain_length=3, chain_state=[0.0, 1.0, 0.0], automata_steps=1)

    assert out["corrected_field"] == pytest.approx((0.5, 1.0 / 3.0, 0.5))
    assert out["correction_field_class"] == "adaptive_field"


def test_boundary_intervention_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("boundary_intervene"),
    )

    out = run_cellular_correction_field(chain_length=5, chain_state=[0.0, 1.0, 0.0, 1.0, 0.0], automata_steps=1)

    assert out["corrected_field"][0] == pytest.approx(0.5)
    assert out["corrected_field"][-1] == pytest.approx(0.5)
    assert out["corrected_field"][2] == pytest.approx(2.0 / 3.0)
    assert out["correction_field_class"] == "intervention_field"


def test_boundedness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("spectral_rebalance"),
    )

    out = run_cellular_correction_field(chain_length=3, chain_state=[-1.0, 2.0, 0.5], automata_steps=2)

    assert all(0.0 <= value <= 1.0 for value in out["initial_field"])
    assert all(0.0 <= value <= 1.0 for value in out["corrected_field"])
    assert 0.0 <= out["field_drift_score"] <= 1.0
    assert 0.0 <= out["local_stability_score"] <= 1.0


def test_trace_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("local_stabilize"),
    )

    out = run_cellular_correction_field(
        chain_length=3,
        chain_state=[0.0, 1.0, 0.0],
        automata_steps=2,
        return_trace=True,
    )

    assert "automata_trace" in out
    assert len(out["automata_trace"]) == 2
    assert out["automata_trace"][0] == pytest.approx((0.5, 1.0 / 3.0, 0.5))


def test_short_chain_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("boundary_intervene"),
    )

    out = run_cellular_correction_field(chain_length=1, chain_state=[0.7], automata_steps=4, return_trace=True)

    assert out["chain_length"] == 1
    assert out["initial_field"] == (0.7,)
    assert out["corrected_field"] == (0.7,)
    assert all(state == (0.7,) for state in out["automata_trace"])


def test_empty_chain_state_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("hold_state"),
    )

    with pytest.raises(ValueError, match="chain_state must be non-empty"):
        run_cellular_correction_field(chain_length=0, chain_state=[])


def test_nested_chain_state_length_consistency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("hold_state"),
    )

    out = run_cellular_correction_field(chain_length=3, chain_state=[[0.1], [0.2], [0.3]], automata_steps=0)

    assert out["chain_length"] == 3
    assert out["initial_field"] == pytest.approx((0.1, 0.2, 0.3))


def test_chain_length_mismatch_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("hold_state"),
    )

    with pytest.raises(ValueError, match="chain_length must match flattened chain_state length"):
        run_cellular_correction_field(chain_length=2, chain_state=[0.1, 0.2, 0.3])


def test_unknown_correction_action_raises_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.cellular_correction_field.run_correction_dispatch",
        lambda **_: _dispatch_stub("unknown_action"),
    )

    with pytest.raises(KeyError, match="unknown correction_action: unknown_action"):
        run_cellular_correction_field(chain_length=3, chain_state=[0.1, 0.2, 0.3], automata_steps=1)
