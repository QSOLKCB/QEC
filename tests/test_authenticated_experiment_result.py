"""Exact-result signing controls for v81/v82 consumers.

Ephemeral signing keys: no private key is checked into the repository.
"""

from __future__ import annotations

import copy

import pytest

from qec.controller.execution_proof import authenticate_experiment_result
from qec.experiments.midi_cube_bridge import run_midi_cube_experiment
from qec.experiments.uff_bridge import run_uff_experiment


@pytest.fixture()
def _signing():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PrivateFormat, PublicFormat, NoEncryption,
    )
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    public = key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    return {"private_key_pem": private, "public_key_pem": public,
            "trusted_signers": {"operator": public}, "signer_id": "operator"}


def test_uff_producer_binds_history_features_consensus_and_theta(_signing):
    theta = [120.0, 3.0, 1.5]
    result = run_uff_experiment(theta, **_signing)
    trust = _signing["trusted_signers"]
    assert authenticate_experiment_result(result, trust, domain="uff", identity=theta)
    assert not authenticate_experiment_result(
        result, trust, domain="uff", identity=[121.0, 3.0, 1.5]
    )
    for section, key, new_value in (
        ("features", "energy", 9999.0),
        ("consensus", "consensus", False),
        ("verification", "match", False),
    ):
        forged = copy.deepcopy(result)
        forged[section][key] = new_value
        assert not authenticate_experiment_result(
            forged, trust, domain="uff", identity=theta
        )


def test_midi_producer_binds_actual_sequence_and_reported_metrics(tmp_path, _signing):
    mido = pytest.importorskip("mido")
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    path = str(tmp_path / "signed.mid")
    midi.save(path)
    result = run_midi_cube_experiment(path, **_signing)
    trust = _signing["trusted_signers"]
    assert authenticate_experiment_result(
        result, trust, domain="sequence", identity=path
    )
    forged = copy.deepcopy(result)
    forged["trajectory"]["classification"] = "fabricated"
    assert not authenticate_experiment_result(
        forged, trust, domain="sequence", identity=path
    )
    assert not authenticate_experiment_result(
        result, trust, domain="sequence", identity=path + "-other"
    )


def test_converged_uff_terminal_record_is_authenticated(_signing):
    """FSM convergence adds a terminal history record without another step."""
    config = {
        "stability_threshold": 0.0,
        "boundary_crossing_threshold": 100,
    }
    theta = [120.0, 3.0, 1.5]
    result = run_uff_experiment(theta, config=config, **_signing)
    history = result["probe"]["history"]
    steps = result["verification"]["steps"]

    assert result["verification"]["match"] is True
    assert result["verification"]["mismatch_index"] is None
    assert result["probe"]["steps"] == steps
    assert len(history) == steps + 1
    assert history[-1]["to_state"] == "TERMINATE"
    assert history[-1]["reason"] == "converged"
    assert authenticate_experiment_result(
        result, _signing["trusted_signers"], domain="uff", identity=theta
    )


def test_terminal_record_convention_rejects_forged_extra_rows():
    """No arbitrary extra rows, incorrect reason, or bogus step counts."""
    from qec.controller.execution_proof import _fsm_step_count_matches_history
    from qec.controller.qec_fsm import QECFSM

    run = QECFSM().run({}, max_steps=2)
    history = run["history"]
    assert run["steps"] == 2
    assert len(history) == 3
    assert history[-1]["reason"] == "max_steps_reached"
    assert _fsm_step_count_matches_history(run["steps"], history)

    invalid = copy.deepcopy(history)
    invalid[-1]["reason"] = "fabricated"
    assert not _fsm_step_count_matches_history(run["steps"], invalid)
    invalid = copy.deepcopy(history)
    invalid[-1]["from_state"] = "unrelated"
    assert not _fsm_step_count_matches_history(run["steps"], invalid)
    invalid = copy.deepcopy(history)
    invalid.append(dict(history[-1]))
    assert not _fsm_step_count_matches_history(run["steps"], invalid)
    assert not _fsm_step_count_matches_history(True, history)
    assert not _fsm_step_count_matches_history(-1, history)
