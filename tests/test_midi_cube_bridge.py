"""Tests for MIDI/Cube/DNA-to-QEC bridge layer (v82.3.0)."""

from __future__ import annotations

import os
import tempfile

import mido
import numpy as np
import pytest

from qec.experiments.midi_cube_bridge import (
    apply_event,
    build_sample,
    dna_roundtrip,
    extract_features,
    init_cube,
    load_midi_sequence,
    run_midi_cube_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_midi_file(events, path):
    """Write a minimal MIDI file with the given note_on events."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for note, velocity, time in events:
        track.append(mido.Message("note_on", note=note, velocity=velocity, time=time))
    mid.save(path)
    return path


@pytest.fixture()
def midi_path(tmp_path):
    """Create a simple MIDI file with 3 note events."""
    events = [
        (60, 100, 0),
        (64, 80, 100),
        (67, 120, 50),
    ]
    return _make_midi_file(events, str(tmp_path / "test.mid"))


@pytest.fixture()
def empty_midi_path(tmp_path):
    """Create a MIDI file with no note_on events."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("control_change", control=64, value=127, time=0))
    path = str(tmp_path / "empty.mid")
    mid.save(path)
    return path


@pytest.fixture()
def single_event_midi_path(tmp_path):
    """Create a MIDI file with exactly one note event."""
    return _make_midi_file([(72, 90, 0)], str(tmp_path / "single.mid"))


# ---------------------------------------------------------------------------
# MIDI loading tests
# ---------------------------------------------------------------------------


class TestLoadMidiSequence:
    """Tests for load_midi_sequence."""

    def test_deterministic_loading(self, midi_path):
        e1 = load_midi_sequence(midi_path)
        e2 = load_midi_sequence(midi_path)
        assert e1 == e2

    def test_ignores_non_note_on(self, empty_midi_path):
        events = load_midi_sequence(empty_midi_path)
        assert events == []

    def test_correct_event_count(self, midi_path):
        events = load_midi_sequence(midi_path)
        assert len(events) == 3

    def test_event_structure(self, midi_path):
        events = load_midi_sequence(midi_path)
        for e in events:
            assert "note" in e
            assert "velocity" in e
            assert "time" in e


# ---------------------------------------------------------------------------
# Cube tests
# ---------------------------------------------------------------------------


class TestCube:
    """Tests for cube init and event application."""

    def test_init_shape(self):
        cube = init_cube()
        assert cube.shape == (8, 8, 8, 3)
        assert cube.dtype == np.uint8

    def test_deterministic_updates(self):
        cube1 = init_cube()
        cube2 = init_cube()
        event = {"note": 60, "velocity": 100, "time": 0.0}
        apply_event(cube1, event)
        apply_event(cube2, event)
        np.testing.assert_array_equal(cube1, cube2)

    def test_no_mutation_of_event(self):
        cube = init_cube()
        event = {"note": 60, "velocity": 100, "time": 0.0}
        event_copy = dict(event)
        apply_event(cube, event)
        assert event == event_copy

    def test_cube_modified_after_event(self):
        cube = init_cube()
        event = {"note": 60, "velocity": 100, "time": 0.0}
        apply_event(cube, event)
        assert np.any(cube > 0)


# ---------------------------------------------------------------------------
# DNA tests
# ---------------------------------------------------------------------------


class TestDnaRoundtrip:
    """Tests for DNA encode/decode roundtrip."""

    def test_roundtrip_no_corruption(self):
        cube = init_cube()
        event = {"note": 60, "velocity": 100, "time": 0.0}
        apply_event(cube, event)
        result = dna_roundtrip(cube, corrupt=False)
        np.testing.assert_array_equal(result, cube)

    def test_corruption_still_decodes(self):
        cube = init_cube()
        event = {"note": 60, "velocity": 100, "time": 0.0}
        apply_event(cube, event)
        result = dna_roundtrip(cube, corrupt=True)
        assert result.shape == cube.shape
        assert result.dtype == cube.dtype


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Tests for extract_features."""

    def test_returns_all_keys(self):
        cube = init_cube()
        apply_event(cube, {"note": 60, "velocity": 100, "time": 0.0})
        features = extract_features(cube)
        assert set(features.keys()) == {
            "energy", "spread", "zcr", "centroid",
            "gradient_energy", "curvature",
        }

    def test_finite_outputs(self):
        cube = init_cube()
        apply_event(cube, {"note": 60, "velocity": 100, "time": 0.0})
        features = extract_features(cube)
        for key, val in features.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_deterministic(self):
        cube = init_cube()
        apply_event(cube, {"note": 60, "velocity": 100, "time": 0.0})
        f1 = extract_features(cube)
        f2 = extract_features(cube)
        assert f1 == f2


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


class TestRunMidiCubeExperiment:
    """Tests for the full MIDI/Cube/DNA → QEC pipeline."""

    def test_deterministic_full_run(self, midi_path):
        r1 = run_midi_cube_experiment(midi_path)
        r2 = run_midi_cube_experiment(midi_path)
        assert r1["features"] == r2["features"]
        assert r1["verification"]["final_hash"] == r2["verification"]["final_hash"]

    def test_consensus_true(self, midi_path):
        result = run_midi_cube_experiment(midi_path)
        assert result["consensus"]["consensus"] is True

    def test_verified_true(self, midi_path):
        result = run_midi_cube_experiment(midi_path)
        assert result["proof"]["verified"] is True

    def test_finite_outputs(self, midi_path):
        result = run_midi_cube_experiment(midi_path)
        for val in result["features"].values():
            assert np.isfinite(val)

    def test_all_keys_present(self, midi_path):
        result = run_midi_cube_experiment(midi_path)
        assert "events" in result
        assert "features" in result
        assert "probe" in result
        assert "invariants" in result
        assert "trajectory" in result
        assert "verification" in result
        assert "proof" in result
        assert "consensus" in result

    def test_empty_midi(self, empty_midi_path):
        result = run_midi_cube_experiment(empty_midi_path)
        assert result["events"] == 0
        assert result["consensus"]["consensus"] is True

    def test_single_event(self, single_event_midi_path):
        result = run_midi_cube_experiment(single_event_midi_path)
        assert result["events"] == 1
        assert result["proof"]["verified"] is True

    def test_corrupt_mode(self, midi_path):
        result = run_midi_cube_experiment(midi_path, corrupt=True)
        assert result["consensus"]["consensus"] is True
        assert result["proof"]["verified"] is True
