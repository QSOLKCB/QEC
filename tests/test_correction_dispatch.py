"""Deterministic tests for v111.2.0 correction dispatch control semantics."""

from __future__ import annotations

import pytest

from qec.analysis.correction_dispatch import run_correction_dispatch


REQUIRED_KEYS = {
    "chain_lengths",
    "spectral_result",
    "correction_action",
    "dispatch_urgency_score",
    "dispatch_cycle_budget",
    "correction_policy_class",
    "action_stability_score",
}


def _base_spectral_stub(**_: object) -> dict[str, object]:
    return {
        "chain_lengths": (4, 8, 16),
        "phase_boundary_class": "drifting_boundary",
        "dominant_component": "balanced_components",
        "spectral_stability_score": 0.5,
        "boundary_shift_score": 0.3,
    }


def test_exact_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.correction_dispatch.run_spectral_phase_boundary", _base_spectral_stub)

    out1 = run_correction_dispatch(diffusion_steps=4)
    out2 = run_correction_dispatch(diffusion_steps=4)

    assert out1 == out2


def test_boundedness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.correction_dispatch.run_spectral_phase_boundary", _base_spectral_stub)

    out = run_correction_dispatch(diffusion_steps=4)

    assert set(out.keys()) == REQUIRED_KEYS
    assert 0.0 <= out["dispatch_urgency_score"] <= 1.0
    assert 1 <= out["dispatch_cycle_budget"] <= 4
    assert 0.0 <= out["action_stability_score"] <= 1.0


def test_stable_boundary_selects_hold_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.correction_dispatch.run_spectral_phase_boundary",
        lambda **_: {
            "chain_lengths": (4, 8),
            "phase_boundary_class": "stable_boundary",
            "dominant_component": "onset_dominant",
            "spectral_stability_score": 0.8,
            "boundary_shift_score": 0.1,
        },
    )

    out = run_correction_dispatch(diffusion_steps=4)

    assert out["correction_action"] == "hold_state"
    assert out["correction_policy_class"] == "monitor_policy"


def test_onset_dominant_selects_local_stabilize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.correction_dispatch.run_spectral_phase_boundary",
        lambda **_: {
            "chain_lengths": (4, 8),
            "phase_boundary_class": "drifting_boundary",
            "dominant_component": "onset_dominant",
            "spectral_stability_score": 0.2,
            "boundary_shift_score": 0.2,
        },
    )

    out = run_correction_dispatch(diffusion_steps=4)

    assert out["correction_action"] == "local_stabilize"
    assert out["correction_policy_class"] == "local_policy"


def test_spectral_dominant_selects_spectral_rebalance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.correction_dispatch.run_spectral_phase_boundary",
        lambda **_: {
            "chain_lengths": (4, 8),
            "phase_boundary_class": "drifting_boundary",
            "dominant_component": "spectral_dominant",
            "spectral_stability_score": 0.2,
            "boundary_shift_score": 0.2,
        },
    )

    out = run_correction_dispatch(diffusion_steps=4)

    assert out["correction_action"] == "spectral_rebalance"
    assert out["correction_policy_class"] == "spectral_policy"


def test_critical_boundary_selects_boundary_intervene(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.correction_dispatch.run_spectral_phase_boundary",
        lambda **_: {
            "chain_lengths": (4, 8),
            "phase_boundary_class": "critical_boundary",
            "dominant_component": "balanced_components",
            "spectral_stability_score": 0.1,
            "boundary_shift_score": 0.95,
        },
    )

    out = run_correction_dispatch(diffusion_steps=4)

    assert out["correction_action"] == "boundary_intervene"
    assert out["correction_policy_class"] == "intervention_policy"


@pytest.mark.parametrize(
    ("urgency", "expected_budget"),
    [
        (0.0, 1),
        (0.2499, 1),
        (0.25, 2),
        (0.4999, 2),
        (0.5, 3),
        (0.7499, 3),
        (0.75, 4),
        (1.0, 4),
    ],
)
def test_dispatch_cycle_budget_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    urgency: float,
    expected_budget: int,
) -> None:
    monkeypatch.setattr(
        "qec.analysis.correction_dispatch.run_spectral_phase_boundary",
        lambda **_: {
            "chain_lengths": (4, 8, 16),
            "phase_boundary_class": "drifting_boundary",
            "dominant_component": "balanced_components",
            "spectral_stability_score": 0.2,
            "boundary_shift_score": urgency,
        },
    )

    out = run_correction_dispatch(diffusion_steps=4)

    assert out["dispatch_urgency_score"] == pytest.approx(urgency)
    assert out["dispatch_cycle_budget"] == expected_budget


def test_optional_action_table_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("qec.analysis.correction_dispatch.run_spectral_phase_boundary", _base_spectral_stub)

    out = run_correction_dispatch(diffusion_steps=4, return_action_table=True)

    assert "action_table" in out
    assert out["action_table"] == {
        "stable_boundary": "hold_state",
        "onset_dominant": "local_stabilize",
        "spectral_dominant": "spectral_rebalance",
        "critical_boundary": "boundary_intervene",
    }


def test_short_chain_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.correction_dispatch.run_spectral_phase_boundary",
        lambda **_: {
            "chain_lengths": (5,),
            "phase_boundary_class": "drifting_boundary",
            "dominant_component": "balanced_components",
            "spectral_stability_score": 1.0,
            "boundary_shift_score": 0.0,
        },
    )

    out = run_correction_dispatch(chain_lengths=[5], diffusion_steps=4)

    assert out["chain_lengths"] == (5,)
    assert out["dispatch_urgency_score"] == 0.0
    assert out["dispatch_cycle_budget"] == 1
