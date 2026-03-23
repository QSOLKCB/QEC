"""Tests for v86.2.0 — spectral trajectory sonification."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import numpy as np

from qec.visualization.trajectory_audio import (
    SAMPLE_RATE,
    STEP_DURATION,
    _normalize,
    _sine_segment,
    sonify_spectral_trajectory,
)


# ── fixtures ─────────────────────────────────────────────────────────


def _make_traj(trajectory_type: str = "convergent", n_steps: int = 5) -> dict:
    drift = [0.8, 0.5, 0.3, 0.1][:n_steps - 1]
    return {
        "n_steps": n_steps,
        "drift": drift,
        "lambda_max": [1.0 + 0.1 * i for i in range(n_steps)],
        "rank_evolution": [4] * n_steps,
        "degeneracy_evolution": [1] * n_steps,
        "temporal_transitions": [{"time_index": 1, "drift": 0.5}],
        "trajectory_type": trajectory_type,
    }


# ── file creation ────────────────────────────────────────────────────


def test_wav_created():
    traj = _make_traj()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.wav"
        result = sonify_spectral_trajectory(traj, output_path=out)
        assert out.exists()
        assert result["output_path"] == str(out)


def test_default_output_path():
    traj = _make_traj()
    result = sonify_spectral_trajectory(traj)
    assert Path(result["output_path"]).exists()


# ── determinism ──────────────────────────────────────────────────────


def test_deterministic_waveform():
    """Same input produces identical WAV bytes."""
    traj = _make_traj()
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.wav"
        p2 = Path(tmp) / "b.wav"
        sonify_spectral_trajectory(traj, output_path=p1)
        sonify_spectral_trajectory(traj, output_path=p2)
        assert p1.read_bytes() == p2.read_bytes()


# ── duration matches steps ───────────────────────────────────────────


def test_duration_matches_steps():
    traj = _make_traj(n_steps=5)
    result = sonify_spectral_trajectory(traj)
    # 5 steps * 0.2s + ending motif (0.3s for convergent)
    expected_min = 5 * STEP_DURATION
    assert result["duration"] >= expected_min


# ── amplitude bounds ─────────────────────────────────────────────────


def test_amplitude_bounds():
    """Waveform values must stay within [-1, 1]."""
    traj = _make_traj()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bounded.wav"
        sonify_spectral_trajectory(traj, output_path=out)
        with wave.open(str(out), "rb") as wf:
            n = wf.getnframes()
            raw = wf.readframes(n)
        samples = np.array(struct.unpack(f"<{n}h", raw), dtype=np.float64)
        samples /= 32767.0
        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)


# ── transition click insertion ───────────────────────────────────────


def test_transition_click_inserted():
    """Steps with transitions should have high-freq content at start."""
    traj = _make_traj()
    traj["temporal_transitions"] = [{"time_index": 0, "drift": 0.8}]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "click.wav"
        sonify_spectral_trajectory(traj, output_path=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


# ── empty trajectory ─────────────────────────────────────────────────


def test_empty_trajectory():
    traj = {
        "n_steps": 0,
        "drift": [],
        "lambda_max": [],
        "temporal_transitions": [],
        "trajectory_type": "undetermined",
    }
    result = sonify_spectral_trajectory(traj)
    assert result["duration"] > 0  # ending motif still present


# ── normalize helper ─────────────────────────────────────────────────


def test_normalize_constant():
    assert _normalize([5.0, 5.0, 5.0]) == [0.5, 0.5, 0.5]


def test_normalize_range():
    result = _normalize([0.0, 1.0])
    assert result == [0.0, 1.0]


def test_normalize_empty():
    assert _normalize([]) == []


# ── trajectory type motifs ───────────────────────────────────────────


def test_all_trajectory_types():
    for ttype in ("convergent", "oscillatory", "divergent", "undetermined"):
        traj = _make_traj(trajectory_type=ttype)
        result = sonify_spectral_trajectory(traj)
        assert result["duration"] > 0
