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
