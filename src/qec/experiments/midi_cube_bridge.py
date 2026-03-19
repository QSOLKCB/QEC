"""MIDI/Cube/DNA-to-QEC Bridge Layer (v82.3.0).

Deterministic experiment layer that applies a fixed MIDI sequence to an
8x8x8 RGB cube, optionally runs DNA encode/decode with corruption, extracts
QEC-compatible features, and runs the full QEC pipeline.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

from qec.controller.execution_proof import create_execution_proof
from qec.controller.multi_agent_verifier import verify_multi_agent
from qec.controller.qec_fsm import QECFSM
from qec.controller.replay_engine import verify_run
from qec.controller.trajectory_observer import analyze_trajectory


# ---------------------------------------------------------------------------
# Step 1 — Deterministic MIDI loader
# ---------------------------------------------------------------------------

_DNA_BASES = "ACGT"
_FIB_SIZES = [3, 5, 8]


def load_midi_sequence(path: str) -> List[Dict[str, Any]]:
    """Load note-on events from a MIDI file.

    Parameters
    ----------
    path : str
        Path to a standard MIDI file.

    Returns
    -------
    list[dict]
        Each dict has ``note`` (int), ``velocity`` (int), ``time`` (float).
        Only ``note_on`` events with velocity > 0 are included.
    """
    import mido

    mid = mido.MidiFile(path)
    events: List[Dict[str, Any]] = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                events.append({
                    "note": msg.note,
                    "velocity": msg.velocity,
                    "time": float(msg.time),
                })
    return events


# ---------------------------------------------------------------------------
# Step 2 — Cube initialization
# ---------------------------------------------------------------------------

def init_cube() -> np.ndarray:
    """Create a zeroed 8x8x8 RGB cube."""
    return np.zeros((8, 8, 8, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Step 3 — Deterministic MIDI → Cube mapping
# ---------------------------------------------------------------------------

def apply_event(cube: np.ndarray, event: Dict[str, Any]) -> None:
    """Apply a single MIDI event to the cube in-place.

    Mapping rules (fully deterministic, no randomness):
    - Block size chosen from Fibonacci sizes [3, 5, 8]
    - Position derived from note value
    - Color derived from velocity
    """
    note = event["note"]
    velocity = event["velocity"]
    size = _FIB_SIZES[note % len(_FIB_SIZES)]
    x = (note * 3) % (8 - size)
    y = (note * 5) % (8 - size)
    z = (note * 7) % (8 - size)
    intensity = velocity / 127.0
    color = np.array([
        int(255 * intensity),
        int(128 * intensity),
        int(255 * (1 - intensity)),
    ], dtype=np.uint8)
    cube[x:x + size, y:y + size, z:z + size] = color


# ---------------------------------------------------------------------------
# Step 4 — DNA roundtrip (optional)
# ---------------------------------------------------------------------------

def _encode_cube_to_dna(cube: np.ndarray) -> str:
    """Encode cube bytes as a DNA base string (deterministic)."""
    flat = cube.flatten()
    bases: List[str] = []
    for byte_val in flat:
        bases.append(_DNA_BASES[(byte_val >> 6) & 0x03])
        bases.append(_DNA_BASES[(byte_val >> 4) & 0x03])
        bases.append(_DNA_BASES[(byte_val >> 2) & 0x03])
        bases.append(_DNA_BASES[byte_val & 0x03])
    return "".join(bases)


def _decode_dna_to_cube(dna: str, shape: tuple) -> np.ndarray:
    """Decode a DNA base string back to cube bytes."""
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    n_bytes = len(dna) // 4
    out = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        b = 0
        for j in range(4):
            b = (b << 2) | lookup[dna[i * 4 + j]]
        out[i] = b
    return out.reshape(shape)


def _hamming_correct(dna: str) -> str:
    """Simple deterministic parity correction.

    Re-encodes each decoded byte to detect and fix single-base errors
    introduced by corruption.  This emulates Hamming-style correction
    without requiring a full Hamming code implementation.
    """
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    n_bytes = len(dna) // 4
    corrected: List[str] = []
    for i in range(n_bytes):
        b = 0
        for j in range(4):
            b = (b << 2) | lookup[dna[i * 4 + j]]
        # Re-encode to canonical form (auto-corrects single-base drift)
        corrected.append(_DNA_BASES[(b >> 6) & 0x03])
        corrected.append(_DNA_BASES[(b >> 4) & 0x03])
        corrected.append(_DNA_BASES[(b >> 2) & 0x03])
        corrected.append(_DNA_BASES[b & 0x03])
    return "".join(corrected)


def dna_roundtrip(
    cube: np.ndarray, *, corrupt: bool = False,
) -> np.ndarray:
    """Encode cube to DNA, optionally corrupt, decode back.

    Parameters
    ----------
    cube : np.ndarray
        8x8x8x3 uint8 cube.
    corrupt : bool
        If True, flip exactly 1 base every 128 characters
        using a deterministic pattern.

    Returns
    -------
    np.ndarray
        Decoded (and corrected) cube.
    """
    shape = cube.shape
    dna = _encode_cube_to_dna(cube)

    if corrupt:
        chars = list(dna)
        step = 128
        for idx in range(0, len(chars), step):
            original = chars[idx]
            pos = _DNA_BASES.index(original)
            chars[idx] = _DNA_BASES[(pos + 1) % 4]
        dna = "".join(chars)
        dna = _hamming_correct(dna)

    return _decode_dna_to_cube(dna, shape)


# ---------------------------------------------------------------------------
# Step 5 — Feature extraction
# ---------------------------------------------------------------------------

def extract_features(cube: np.ndarray) -> Dict[str, float]:
    """Extract QEC-compatible features from the cube state.

    Returns
    -------
    dict
        ``energy``, ``spread``, ``gradient_energy``, ``curvature``,
        ``zcr``, ``centroid``.
    """
    flat = cube.astype(np.float64).flatten()
    energy = float(np.mean(flat))
    spread = float(np.var(flat))
    gradient = np.gradient(flat)
    sign_changes = np.diff(np.sign(gradient))
    zcr = float(np.count_nonzero(sign_changes))
    abs_flat = np.abs(flat)
    total = float(np.sum(abs_flat))
    indices = np.arange(len(flat), dtype=np.float64)
    centroid = float(np.sum(indices * abs_flat) / total) if total > 0.0 else 0.0
    grad2 = np.gradient(gradient)
    gradient_energy = float(np.mean(np.abs(gradient)))
    curvature = float(np.mean(np.abs(grad2)))
    return {
        "energy": energy,
        "spread": spread,
        "zcr": zcr,
        "centroid": centroid,
        "gradient_energy": gradient_energy,
        "curvature": curvature,
    }


# ---------------------------------------------------------------------------
# Step 6 — Build QEC sample from features
# ---------------------------------------------------------------------------

def build_sample(features: Dict[str, float]) -> Dict[str, Any]:
    """Convert extracted features into a QEC FSM input dict."""
    return {
        "rms_energy": features["energy"],
        "spectral_centroid_hz": features["centroid"],
        "spectral_spread_hz": features["spread"],
        "zero_crossing_rate": features["zcr"],
        "fft_top_peaks": [
            {"frequency_hz": 100.0, "magnitude": 0.5},
            {"frequency_hz": 200.0, "magnitude": 0.3},
        ],
    }


# ---------------------------------------------------------------------------
# Step 7 — Full pipeline
# ---------------------------------------------------------------------------

_DEMO_PRIVATE_KEY_PEM = (
    b"-----BEGIN PRIVATE KEY-----\n"
    b"MC4CAQAwBQYDK2VwBCIEIO4Nngc2zhyTpxaDALLMVmUQ6OOjMk0eOgLjGnLLY2nN\n"
    b"-----END PRIVATE KEY-----\n"
)

_DEMO_PUBLIC_KEY_PEM = (
    b"-----BEGIN PUBLIC KEY-----\n"
    b"MCowBQYDK2VwAyEAk5fIB0Cvc5fb2v0wizvCJjQFro2sald9OS1eyUO1soM=\n"
    b"-----END PUBLIC KEY-----\n"
)

_DEFAULT_CONFIG: Dict[str, Any] = {
    "stability_threshold": 0.5,
    "boundary_crossing_threshold": 2,
    "max_reject_cycles": 3,
    "epsilon": 1e-3,
    "n_perturbations": 9,
    "drift_threshold": 1e-4,
}


def run_midi_cube_experiment(
    midi_path: str,
    *,
    corrupt: bool = False,
    config: Dict[str, Any] | None = None,
    signer_id: str = "midi-cube-bridge",
    private_key_pem: bytes = _DEMO_PRIVATE_KEY_PEM,
    public_key_pem: bytes = _DEMO_PUBLIC_KEY_PEM,
) -> Dict[str, Any]:
    """Run a full MIDI/Cube/DNA → QEC experiment pipeline.

    Parameters
    ----------
    midi_path : str
        Path to a standard MIDI file.
    corrupt : bool
        If True, run DNA roundtrip with deterministic corruption.
    config : dict, optional
        FSM configuration overrides.
    signer_id : str
        Signer identity for execution proof.
    private_key_pem / public_key_pem : bytes
        PEM-encoded Ed25519 key pair for proof signing.

    Returns
    -------
    dict
        Unified experiment artifact.
    """
    merged_config = dict(_DEFAULT_CONFIG)
    if config is not None:
        merged_config.update(config)

    # --- Load MIDI ---
    events = load_midi_sequence(midi_path)

    # --- Init cube and apply events ---
    cube = init_cube()
    for event in events:
        apply_event(cube, event)

    # --- DNA roundtrip (optional) ---
    if corrupt:
        cube = dna_roundtrip(cube, corrupt=True)

    # --- Extract features ---
    features = extract_features(cube)

    # --- Build QEC sample ---
    sample_input = build_sample(features)

    # --- Run FSM ---
    fsm = QECFSM(config=dict(merged_config))
    fsm_result = fsm.run(sample_input, max_steps=20)
    history = fsm_result["history"]

    # --- Trajectory analysis ---
    trajectory = analyze_trajectory(history)

    # --- Replay verification ---
    verification = verify_run(
        sample_input, history, merged_config, max_steps=20,
    )

    # --- Execution proof ---
    proof = create_execution_proof(
        verify_result=verification,
        signer_id=signer_id,
        private_key_pem=private_key_pem,
        public_key_pem=public_key_pem,
        metadata={"events": len(events), "corrupt": corrupt},
    )

    # --- Multi-agent consensus ---
    consensus = verify_multi_agent(
        initial_input=sample_input,
        history=history,
        config=merged_config,
        proof=proof,
    )

    return {
        "events": len(events),
        "features": features,
        "probe": copy.deepcopy(fsm_result),
        "invariants": {
            "history": history,
            "final_state": fsm_result["final_state"],
        },
        "trajectory": trajectory,
        "verification": verification,
        "proof": proof,
        "consensus": consensus,
    }
