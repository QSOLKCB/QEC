from __future__ import annotations

import json

import pytest

from qec.analysis.adaptive_thermal_control_kernel import ThermalControlReceipt, ThermalNodeDecision
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.distributed_timing_mesh import NodeTimingDecision, TimingMeshReceipt
from qec.analysis.hardware_feedback_consensus_bridge import (
    HARDWARE_FEEDBACK_CONSENSUS_BRIDGE_VERSION,
    HardwareConsensusReceipt,
    HardwareFeedbackInputs,
    evaluate_hardware_feedback_consensus_bridge,
)
from qec.analysis.latency_stabilization_loop import LatencyControlReceipt, LatencyNodeDecision
from qec.analysis.power_aware_control_modulation import PowerControlReceipt, PowerNodeDecision


def _thermal_receipt(*, action: str, pressure: float, stability: float) -> ThermalControlReceipt:
    return ThermalControlReceipt(
        version="v140.0",
        node_decisions=(
            ThermalNodeDecision(
                node_id="n1",
                thermal_pressure=pressure,
                cooling_bias=min(1.0, pressure),
                workload_derate=min(1.0, pressure),
                stability_score=stability,
                action_label=action,
            ),
        ),
        mesh_thermal_pressure=pressure,
        mesh_stability_score=stability,
        hotspot_count=0,
        control_mode="thermal_advisory",
        observatory_only=True,
    )


def _latency_receipt(*, action: str, pressure: float, stability: float) -> LatencyControlReceipt:
    decision = LatencyNodeDecision(
        node_id="n1",
        instability_pressure=pressure,
        correction_strength=min(1.0, pressure),
        stability_score=stability,
        action_label=action,
    )
    payload = {
        "version": "v140.1",
        "node_decisions": (decision.to_dict(),),
        "mesh_instability_pressure": pressure,
        "mesh_stability_score": stability,
        "instability_count": 0,
        "control_mode": "latency_advisory",
        "observatory_only": True,
    }
    return LatencyControlReceipt(
        version="v140.1",
        node_decisions=(decision,),
        mesh_instability_pressure=pressure,
        mesh_stability_score=stability,
        instability_count=0,
        control_mode="latency_advisory",
        observatory_only=True,
        stable_hash=sha256_hex(payload),
    )


def _timing_receipt(*, action: str, pressure: float, stability: float) -> TimingMeshReceipt:
    return TimingMeshReceipt(
        version="v140.2",
        node_decisions=(
            NodeTimingDecision(
                node_id="n1",
                timing_drift=pressure,
                alignment_error=1.0 - stability,
                correction_offset_ms=0.0,
                action_label=action,
            ),
        ),
        mesh_timing_drift=pressure,
        mesh_alignment_error=1.0 - stability,
        synchronization_confidence=stability,
        mesh_stability=stability,
        control_mode="timing_mesh_advisory",
        observatory_only=True,
    )


def _power_receipt(*, action: str, pressure: float, stability: float) -> PowerControlReceipt:
    return PowerControlReceipt(
        version="v140.3",
        node_decisions=(
            PowerNodeDecision(
                node_id="n1",
                power_pressure=pressure,
                load_balance_score=stability,
                modulation_strength=min(1.0, pressure),
                efficiency_score=stability,
                action_label=action,
            ),
        ),
        mesh_power_pressure=pressure,
        mesh_efficiency_score=stability,
        overload_count=0,
        control_mode="power_advisory",
        observatory_only=True,
    )


def _inputs(
    *,
    thermal_action: str,
    latency_action: str,
    timing_action: str,
    power_action: str,
    thermal_pressure: float,
    latency_pressure: float,
    timing_pressure: float,
    power_pressure: float,
    stability: float = 0.9,
) -> HardwareFeedbackInputs:
    return HardwareFeedbackInputs(
        thermal_receipt=_thermal_receipt(action=thermal_action, pressure=thermal_pressure, stability=stability),
        latency_receipt=_latency_receipt(action=latency_action, pressure=latency_pressure, stability=stability),
        timing_receipt=_timing_receipt(action=timing_action, pressure=timing_pressure, stability=stability),
        power_receipt=_power_receipt(action=power_action, pressure=power_pressure, stability=stability),
    )


def test_deterministic_replay_identical_json_and_hash() -> None:
    inputs = _inputs(
        thermal_action="pre_cool",
        latency_action="adjust",
        timing_action="adjust",
        power_action="balance",
        thermal_pressure=0.3,
        latency_pressure=0.4,
        timing_pressure=0.3,
        power_pressure=0.2,
    )

    first = evaluate_hardware_feedback_consensus_bridge(inputs)
    second = evaluate_hardware_feedback_consensus_bridge(inputs)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_vote_ordering_is_fixed() -> None:
    receipt = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="hold",
            latency_action="stable",
            timing_action="stable",
            power_action="stable",
            thermal_pressure=0.0,
            latency_pressure=0.0,
            timing_pressure=0.0,
            power_pressure=0.0,
        )
    )

    assert tuple(vote.signal_name for vote in receipt.signal_votes) == ("thermal", "latency", "timing", "power")


def test_dominant_signal_tie_break_by_pressure_then_fixed_order() -> None:
    pressure_winner = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="derate",
            latency_action="correct",
            timing_action="correct",
            power_action="reduce",
            thermal_pressure=0.61,
            latency_pressure=0.77,
            timing_pressure=0.60,
            power_pressure=0.59,
        )
    )
    assert pressure_winner.decision.dominant_signal == "latency"

    fixed_order_winner = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="derate",
            latency_action="correct",
            timing_action="correct",
            power_action="reduce",
            thermal_pressure=0.7,
            latency_pressure=0.7,
            timing_pressure=0.7,
            power_pressure=0.7,
        )
    )
    assert fixed_order_winner.decision.dominant_signal == "thermal"


def test_conflict_count_tracks_distinct_action_labels() -> None:
    receipt = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="hold",
            latency_action="adjust",
            timing_action="correct",
            power_action="critical",
            thermal_pressure=0.1,
            latency_pressure=0.2,
            timing_pressure=0.3,
            power_pressure=0.4,
        )
    )

    assert receipt.decision.conflict_count == 3


def test_consensus_confidence_drops_with_disagreement() -> None:
    low_disagreement = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="hold",
            latency_action="stable",
            timing_action="stable",
            power_action="stable",
            thermal_pressure=0.2,
            latency_pressure=0.2,
            timing_pressure=0.2,
            power_pressure=0.2,
        )
    )
    high_disagreement = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="hold",
            latency_action="critical",
            timing_action="correct",
            power_action="balance",
            thermal_pressure=0.0,
            latency_pressure=1.0,
            timing_pressure=0.8,
            power_pressure=0.3,
        )
    )

    assert high_disagreement.decision.consensus_confidence < low_disagreement.decision.consensus_confidence


def test_emergency_override_forces_emergency_align_result() -> None:
    receipt = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="critical",
            latency_action="stable",
            timing_action="stable",
            power_action="stable",
            thermal_pressure=0.1,
            latency_pressure=0.1,
            timing_pressure=0.1,
            power_pressure=0.1,
        )
    )

    assert receipt.decision.action_label == "emergency_align"


def test_reduce_load_path_from_vote_or_high_consensus_pressure() -> None:
    via_vote = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="hold",
            latency_action="correct",
            timing_action="stable",
            power_action="stable",
            thermal_pressure=0.1,
            latency_pressure=0.1,
            timing_pressure=0.1,
            power_pressure=0.1,
        )
    )
    via_pressure = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="hold",
            latency_action="stable",
            timing_action="stable",
            power_action="stable",
            thermal_pressure=0.7,
            latency_pressure=0.7,
            timing_pressure=0.7,
            power_pressure=0.7,
        )
    )

    assert via_vote.decision.action_label == "reduce_load"
    assert via_pressure.decision.action_label == "reduce_load"


def test_stable_path_all_stable_votes() -> None:
    receipt = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="hold",
            latency_action="stable",
            timing_action="stable",
            power_action="stable",
            thermal_pressure=0.1,
            latency_pressure=0.1,
            timing_pressure=0.1,
            power_pressure=0.1,
        )
    )

    assert receipt.decision.action_label == "stable"


def test_validation_incorrect_receipt_types_raise_value_error() -> None:
    with pytest.raises(ValueError, match="thermal_receipt"):
        HardwareFeedbackInputs(
            thermal_receipt=object(),  # type: ignore[arg-type]
            latency_receipt=_latency_receipt(action="stable", pressure=0.1, stability=0.9),
            timing_receipt=_timing_receipt(action="stable", pressure=0.1, stability=0.9),
            power_receipt=_power_receipt(action="stable", pressure=0.1, stability=0.9),
        )


def test_hash_stability_repeated_runs_identical() -> None:
    inputs = _inputs(
        thermal_action="pre_cool",
        latency_action="adjust",
        timing_action="adjust",
        power_action="balance",
        thermal_pressure=0.33,
        latency_pressure=0.31,
        timing_pressure=0.29,
        power_pressure=0.35,
    )

    hashes = [evaluate_hardware_feedback_consensus_bridge(inputs).stable_hash for _ in range(5)]
    assert len(set(hashes)) == 1


def test_canonical_serialization_replay_safe() -> None:
    receipt = evaluate_hardware_feedback_consensus_bridge(
        _inputs(
            thermal_action="pre_cool",
            latency_action="adjust",
            timing_action="adjust",
            power_action="balance",
            thermal_pressure=0.4,
            latency_pressure=0.3,
            timing_pressure=0.2,
            power_pressure=0.1,
        )
    )

    parsed = json.loads(receipt.to_canonical_json())
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) == receipt.to_canonical_json()
    assert isinstance(receipt, HardwareConsensusReceipt)
    assert receipt.version == HARDWARE_FEEDBACK_CONSENSUS_BRIDGE_VERSION
    assert receipt.control_mode == "hardware_consensus_advisory"
    assert receipt.observatory_only is True
