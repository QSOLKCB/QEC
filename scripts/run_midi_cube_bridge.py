#!/usr/bin/env python3
"""
v82.3.0 — MIDI/Cube/DNA-to-QEC Bridge Runner Script.

Creates a synthetic MIDI file, runs it through the full QEC pipeline,
and prints a summary of the unified experiment artifact.

Usage
-----
    python scripts/run_midi_cube_bridge.py
    python scripts/run_midi_cube_bridge.py path/to/file.mid
"""

from __future__ import annotations

import sys
import tempfile

import mido

from qec.experiments.midi_cube_bridge import run_midi_cube_experiment


def _create_demo_midi(path: str) -> str:
    """Write a deterministic demo MIDI file."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    notes = [60, 64, 67, 72, 76, 60, 55, 48]
    for i, note in enumerate(notes):
        velocity = 80 + (i * 7) % 48
        track.append(mido.Message("note_on", note=note, velocity=velocity, time=i * 50))
    mid.save(path)
    return path


def main() -> None:
    """Run MIDI/Cube/DNA → QEC bridge and print summary."""
    if len(sys.argv) > 1:
        midi_path = sys.argv[1]
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        midi_path = _create_demo_midi(tmp.name)

    result = run_midi_cube_experiment(midi_path, corrupt=False)

    print("=== MIDI/Cube/DNA → QEC Bridge ===")
    print(f"EVENTS      : {result['events']}")
    print(f"ENERGY      : {result['features']['energy']:.6f}")
    print(f"SPREAD      : {result['features']['spread']:.6f}")
    print(f"PHASE       : {result['invariants']['final_state']}")
    print(f"CONSENSUS   : {result['consensus']['consensus']}")
    print(f"HASH        : {result['verification']['final_hash']}")
    print(f"VERIFIED    : {result['proof']['verified']}")


if __name__ == "__main__":
    main()
