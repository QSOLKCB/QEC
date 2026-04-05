from __future__ import annotations

import json

import pytest

from qec.analysis.quantum_communication_governance_layer import (
    NodeState,
    build_secure_replay_signature_chain,
    compute_cross_node_drift_provenance,
    run_quantum_communication_governance_layer,
    synchronize_node_states,
)


def _node_states() -> tuple[NodeState, ...]:
    return (
        NodeState(
            node_id="node-b",
            epoch=1,
            state_hash="state-b1",
            metrics=(("latency", 0.20), ("fidelity", 0.97)),
            governance_flags=("trusted",),
        ),
        NodeState(
            node_id="node-a",
            epoch=0,
            state_hash="state-a0",
            metrics=(("latency", 0.10), ("fidelity", 0.99)),
            governance_flags=("trusted", "stable"),
        ),
        NodeState(
            node_id="node-a",
            epoch=2,
            state_hash="state-a2",
            metrics=(("latency", 0.12), ("fidelity", 0.995)),
            governance_flags=("trusted",),
        ),
    )


def test_repeated_run_determinism_and_same_bytes() -> None:
    states = _node_states()
    a = run_quantum_communication_governance_layer(states)
    b = run_quantum_communication_governance_layer(states)
    assert a == b
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_canonical_json_stability() -> None:
    report = run_quantum_communication_governance_layer(_node_states())
    canonical_json = report.to_canonical_json()
    reparsed = json.loads(canonical_json)
    assert json.dumps(reparsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True) == canonical_json


def test_stable_replay_signature_chain() -> None:
    synchronized = synchronize_node_states(_node_states())
    chain_a = build_secure_replay_signature_chain(synchronized)
    chain_b = build_secure_replay_signature_chain(synchronized)
    assert chain_a == chain_b
    assert chain_a.replay_identity == chain_b.replay_identity
    assert len(chain_a.entries) == len(synchronized)


def test_deterministic_node_state_synchronization() -> None:
    synchronized = synchronize_node_states(_node_states())
    assert tuple(state.node_id for state in synchronized) == ("node-a", "node-b")
    assert synchronized[0].epoch == 2


def test_governance_trust_score_is_bounded() -> None:
    report = run_quantum_communication_governance_layer(_node_states())
    assert 0.0 <= report.governance_trust_score <= 1.0


def test_fail_fast_invalid_node_state_input() -> None:
    bad_state = NodeState(node_id="", epoch=1, state_hash="ok", metrics=(("fidelity", 1.0),), governance_flags=())
    with pytest.raises(ValueError):
        run_quantum_communication_governance_layer((bad_state,))


def test_stable_cross_node_drift_provenance_output() -> None:
    synchronized = synchronize_node_states(_node_states())
    drift_a = compute_cross_node_drift_provenance(synchronized)
    drift_b = compute_cross_node_drift_provenance(synchronized)
    assert drift_a == drift_b
    assert tuple((d.node_pair, d.metric_key, d.provenance_hash) for d in drift_a) == tuple(
        (d.node_pair, d.metric_key, d.provenance_hash) for d in drift_b
    )
