"""Tests for deterministic spectral sonification utilities."""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.spectral_sonifier import sonify_experiment_artifact, sonify_spectrum


def test_sonify_basic() -> None:
    eigs = [1.0, 2.0, 3.0]
    audio = sonify_spectrum(eigs)
    assert len(audio) > 0


def test_sonify_output_length_is_deterministic() -> None:
    eigs = [0.1, 0.2, 0.3, 0.4]
    duration = 0.125
    sr = 8000

    audio = sonify_spectrum(eigs, duration=duration, sr=sr)

    expected = len(eigs) * int(sr * duration)
    assert len(audio) == expected


def test_sonify_waveform_is_deterministic() -> None:
    eigs = [0.25, 0.5, 0.75]
    ipr = [0.1, 0.4, 0.9]

    first = sonify_spectrum(eigs, ipr=ipr, duration=0.05, sr=4000)
    second = sonify_spectrum(eigs, ipr=ipr, duration=0.05, sr=4000)

    assert np.array_equal(first, second)


def test_sonify_artifact_writes_wav(tmp_path) -> None:
    artifact = {
        "nb_eigenvalues": [0.1, 0.2],
        "ipr_values": [0.4, 0.6],
    }
    output_path = tmp_path / "artifact.wav"

    sonify_experiment_artifact(artifact, str(output_path))

    assert output_path.exists()
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 44100
        assert wav_file.getnframes() > 0
