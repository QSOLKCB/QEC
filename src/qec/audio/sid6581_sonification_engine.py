"""
QEC SID 6581 Sonification Engine — Deterministic SID-Inspired Synthesis (v136.8.8).

Hardware-faithful observability layer for QEC auditory state rendering,
inspired by the MOS Technology 6581 SID chip.

Voice mapping
-------------
voice1 -> cognition confidence carrier
voice2 -> gate severity / rollback risk
voice3 -> drift monitor state

Design invariants
-----------------
* frozen dataclasses only
* deterministic — same input always produces identical waveform and hash
* no hidden randomness
* no decoder imports
* stdlib + numpy only
* 3 deterministic voices
* ring modulation faithful to SID hardware
* filter modes: lowpass, highpass, bandpass
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENGINE_VERSION: str = "v136.8.8"

SAMPLE_RATE: int = 44100
DURATION: float = 0.25  # seconds
NUM_SAMPLES: int = int(SAMPLE_RATE * DURATION)

# SID frequency bounds (C-0 to ~4 kHz, faithful to 6581 range)
SID_FREQ_MIN: float = 16.35
SID_FREQ_MAX: float = 4000.0

# ADSR bounds (milliseconds, except sustain which is 0.0-1.0)
ATTACK_MS_MIN: float = 2.0
ATTACK_MS_MAX: float = 800.0
DECAY_MS_MIN: float = 6.0
DECAY_MS_MAX: float = 2400.0
SUSTAIN_MIN: float = 0.0
SUSTAIN_MAX: float = 1.0
RELEASE_MS_MIN: float = 6.0
RELEASE_MS_MAX: float = 2400.0

# Filter bounds
FILTER_CUTOFF_MIN: float = 30.0
FILTER_CUTOFF_MAX: float = 12000.0
RESONANCE_MIN: float = 0.0
RESONANCE_MAX: float = 1.0

VALID_WAVEFORMS: Tuple[str, ...] = ("triangle", "sawtooth", "pulse", "noise")
VALID_FILTER_MODES: Tuple[str, ...] = ("lowpass", "highpass", "bandpass")

# Pulse duty cycle for pulse waveform
PULSE_DUTY: float = 0.5

# Noise LFSR seed (deterministic)
NOISE_LFSR_SEED: int = 0x7FFFF8


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIDVoice:
    """Immutable SID voice configuration."""

    frequency_hz: float
    waveform: str
    attack_ms: float
    decay_ms: float
    sustain_level: float
    release_ms: float


@dataclass(frozen=True)
class SIDFrame:
    """Immutable SID frame — 3 voices plus filter configuration."""

    voice1: SIDVoice
    voice2: SIDVoice
    voice3: SIDVoice
    ring_mod_enabled: bool
    filter_mode: str
    cutoff_hz: float
    resonance: float


@dataclass(frozen=True)
class SIDRenderResult:
    """Immutable result of a SID render cycle."""

    audio_buffer: Tuple[float, ...]
    spectral_hash: str
    drift_overlay: float
    stable_hash: str


# ---------------------------------------------------------------------------
# Deterministic hash-to-float mapping
# ---------------------------------------------------------------------------


def _hash_to_float(data: str, salt: str = "") -> float:
    """Map a string deterministically to [0.0, 1.0) via SHA-256."""
    digest = hashlib.sha256(f"{salt}:{data}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / (2**32)


def _hash_to_range(data: str, low: float, high: float, salt: str = "") -> float:
    """Map a string deterministically to [low, high) via SHA-256."""
    return low + _hash_to_float(data, salt) * (high - low)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


# ---------------------------------------------------------------------------
# Waveform generators (deterministic, no hidden state)
# ---------------------------------------------------------------------------


def _generate_triangle(num_samples: int, frequency_hz: float,
                       sample_rate: int) -> Tuple[float, ...]:
    """Generate deterministic triangle waveform."""
    if frequency_hz <= 0.0:
        return tuple(0.0 for _ in range(num_samples))
    samples = []
    for i in range(num_samples):
        phase = (i * frequency_hz / sample_rate) % 1.0
        if phase < 0.25:
            val = 4.0 * phase
        elif phase < 0.75:
            val = 2.0 - 4.0 * phase
        else:
            val = -4.0 + 4.0 * phase
        samples.append(val)
    return tuple(samples)


def _generate_sawtooth(num_samples: int, frequency_hz: float,
                       sample_rate: int) -> Tuple[float, ...]:
    """Generate deterministic sawtooth waveform."""
    if frequency_hz <= 0.0:
        return tuple(0.0 for _ in range(num_samples))
    samples = []
    for i in range(num_samples):
        phase = (i * frequency_hz / sample_rate) % 1.0
        val = 2.0 * phase - 1.0
        samples.append(val)
    return tuple(samples)


def _generate_pulse(num_samples: int, frequency_hz: float,
                    sample_rate: int,
                    duty: float = PULSE_DUTY) -> Tuple[float, ...]:
    """Generate deterministic pulse waveform."""
    if frequency_hz <= 0.0:
        return tuple(0.0 for _ in range(num_samples))
    samples = []
    for i in range(num_samples):
        phase = (i * frequency_hz / sample_rate) % 1.0
        val = 1.0 if phase < duty else -1.0
        samples.append(val)
    return tuple(samples)


def _generate_noise(num_samples: int, seed_salt: str) -> Tuple[float, ...]:
    """Generate deterministic noise via LFSR seeded from salt."""
    digest = hashlib.sha256(seed_salt.encode("utf-8")).hexdigest()
    lfsr = int(digest[:6], 16) | 1  # Ensure non-zero 23-bit LFSR
    lfsr &= 0x7FFFFF
    if lfsr == 0:
        lfsr = NOISE_LFSR_SEED

    samples = []
    for _ in range(num_samples):
        # Galois LFSR feedback polynomial (23-bit, taps at 23 and 18)
        bit = lfsr & 1
        lfsr >>= 1
        if bit:
            lfsr ^= 0x400002  # Feedback taps
        # Map to [-1.0, 1.0] via modular reduction
        val = (lfsr % 0x400000) / 0x200000 - 1.0
        samples.append(val)
    return tuple(samples)


def _generate_waveform(waveform: str, num_samples: int,
                       frequency_hz: float, sample_rate: int,
                       noise_salt: str = "default") -> Tuple[float, ...]:
    """Dispatch to the correct deterministic waveform generator."""
    if waveform == "triangle":
        return _generate_triangle(num_samples, frequency_hz, sample_rate)
    elif waveform == "sawtooth":
        return _generate_sawtooth(num_samples, frequency_hz, sample_rate)
    elif waveform == "pulse":
        return _generate_pulse(num_samples, frequency_hz, sample_rate)
    elif waveform == "noise":
        return _generate_noise(num_samples, noise_salt)
    else:
        raise ValueError(f"Invalid waveform: {waveform!r}. "
                         f"Must be one of {VALID_WAVEFORMS}")


# ---------------------------------------------------------------------------
# ADSR envelope (deterministic)
# ---------------------------------------------------------------------------


def _apply_adsr_envelope(
    buffer: Tuple[float, ...],
    attack_ms: float,
    decay_ms: float,
    sustain_level: float,
    release_ms: float,
    sample_rate: int,
) -> Tuple[float, ...]:
    """Apply deterministic ADSR envelope to a buffer."""
    num_samples = len(buffer)
    attack_samples = int(attack_ms * sample_rate / 1000.0)
    decay_samples = int(decay_ms * sample_rate / 1000.0)
    release_samples = int(release_ms * sample_rate / 1000.0)

    sustain_samples = max(
        0, num_samples - attack_samples - decay_samples - release_samples
    )

    envelope = []
    for i in range(num_samples):
        if i < attack_samples:
            # Attack: linear ramp 0 -> 1
            env = i / max(attack_samples, 1)
        elif i < attack_samples + decay_samples:
            # Decay: linear ramp 1 -> sustain_level
            progress = (i - attack_samples) / max(decay_samples, 1)
            env = 1.0 - progress * (1.0 - sustain_level)
        elif i < attack_samples + decay_samples + sustain_samples:
            # Sustain: constant
            env = sustain_level
        else:
            # Release: linear ramp sustain_level -> 0
            rel_idx = i - attack_samples - decay_samples - sustain_samples
            progress = rel_idx / max(release_samples, 1)
            env = sustain_level * (1.0 - min(progress, 1.0))
        envelope.append(env)

    return tuple(buffer[i] * envelope[i] for i in range(num_samples))


# ---------------------------------------------------------------------------
# Voice rendering
# ---------------------------------------------------------------------------


def _render_voice(voice: SIDVoice, sample_rate: int,
                  num_samples: int,
                  noise_salt: str = "voice") -> Tuple[float, ...]:
    """Render a single SID voice to a deterministic audio buffer."""
    raw = _generate_waveform(
        voice.waveform, num_samples, voice.frequency_hz,
        sample_rate, noise_salt=noise_salt,
    )
    return _apply_adsr_envelope(
        raw,
        voice.attack_ms,
        voice.decay_ms,
        voice.sustain_level,
        voice.release_ms,
        sample_rate,
    )


# ---------------------------------------------------------------------------
# Ring modulation (SID hardware faithful)
# ---------------------------------------------------------------------------


def apply_sid_ring_modulation(
    buffer_a: Tuple[float, ...],
    buffer_b: Tuple[float, ...],
) -> Tuple[float, ...]:
    """Apply SID-style ring modulation: element-wise multiplication.

    On the real 6581, ring mod replaces the triangle output of one
    oscillator with the product of its triangle and another oscillator's
    output.  We faithfully model this as sample-by-sample multiplication.

    Both buffers must have the same length.
    """
    if len(buffer_a) != len(buffer_b):
        raise ValueError(
            f"Ring modulation requires equal-length buffers: "
            f"{len(buffer_a)} != {len(buffer_b)}"
        )
    return tuple(a * b for a, b in zip(buffer_a, buffer_b))


# ---------------------------------------------------------------------------
# SID filter (deterministic digital model)
# ---------------------------------------------------------------------------


def apply_sid_filter(
    buffer: Tuple[float, ...],
    mode: str,
    cutoff_hz: float,
    resonance: float,
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[float, ...]:
    """Apply deterministic SID-style state-variable filter.

    Models the 6581's state-variable filter topology with:
    - lowpass, highpass, bandpass modes
    - cutoff frequency and resonance parameters
    - fully deterministic (no hidden state between calls)

    Parameters
    ----------
    buffer : input audio samples
    mode : one of "lowpass", "highpass", "bandpass"
    cutoff_hz : filter cutoff frequency in Hz
    resonance : resonance amount in [0.0, 1.0]
    sample_rate : audio sample rate
    """
    if mode not in VALID_FILTER_MODES:
        raise ValueError(
            f"Invalid filter mode: {mode!r}. Must be one of {VALID_FILTER_MODES}"
        )

    # State-variable filter coefficients
    # f = 2 * sin(pi * cutoff / sample_rate), clamped for stability
    f = 2.0 * math.sin(math.pi * _clamp(cutoff_hz, 1.0, sample_rate * 0.45)
                        / sample_rate)
    # Q factor from resonance: higher resonance = sharper peak
    q = 1.0 - _clamp(resonance, 0.0, 0.95)

    # Filter state (initialized to zero — deterministic)
    lp = 0.0  # lowpass state
    bp = 0.0  # bandpass state

    output = []
    for sample in buffer:
        # State-variable filter update (Chamberlin topology)
        hp = sample - lp - q * bp
        bp = bp + f * hp
        lp = lp + f * bp

        if mode == "lowpass":
            output.append(lp)
        elif mode == "highpass":
            output.append(hp)
        else:  # bandpass
            output.append(bp)

    return tuple(output)


# ---------------------------------------------------------------------------
# Spectral hash
# ---------------------------------------------------------------------------


def compute_sid_spectral_hash(buffer: Tuple[float, ...]) -> str:
    """Compute deterministic spectral hash of an audio buffer.

    Uses canonical JSON serialization of rounded samples for
    byte-identical hashing across runs.
    """
    # Round to 10 decimal places to avoid floating-point noise
    canonical = json.dumps(
        [round(s, 10) for s in buffer],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frame construction from QEC state
# ---------------------------------------------------------------------------


def _extract_confidence(cognition_result: Any) -> float:
    """Extract confidence from cognition result (duck-typed)."""
    if hasattr(cognition_result, "match"):
        match = cognition_result.match
        if hasattr(match, "confidence"):
            return float(match.confidence)
    if isinstance(cognition_result, Mapping):
        if "match" in cognition_result:
            m = cognition_result["match"]
            if isinstance(m, Mapping) and "confidence" in m:
                return float(m["confidence"])
            if hasattr(m, "confidence"):
                return float(m.confidence)
        if "confidence" in cognition_result:
            return float(cognition_result["confidence"])
    return 0.5


def _extract_gate_severity(gate_result: Any) -> float:
    """Extract severity / rollback risk from gate result (duck-typed)."""
    if hasattr(gate_result, "decision"):
        dec = gate_result.decision
        if hasattr(dec, "confidence"):
            return 1.0 - float(dec.confidence)
    if isinstance(gate_result, Mapping):
        if "decision" in gate_result:
            d = gate_result["decision"]
            if hasattr(d, "confidence"):
                return 1.0 - float(d.confidence)
            if isinstance(d, Mapping) and "confidence" in d:
                return 1.0 - float(d["confidence"])
        if "severity" in gate_result:
            return float(gate_result["severity"])
    return 0.5


def _extract_drift_score(history_ledger: Any) -> float:
    """Extract drift score from history ledger (duck-typed)."""
    if hasattr(history_ledger, "drift_score"):
        return float(history_ledger.drift_score)
    if isinstance(history_ledger, Mapping):
        if "drift_score" in history_ledger:
            return float(history_ledger["drift_score"])
    return 0.0


def _build_state_hash(cognition_result: Any, gate_result: Any,
                      history_ledger: Any) -> str:
    """Build deterministic hash of the combined QEC state."""
    parts = []

    # Cognition hash
    if hasattr(cognition_result, "match") and hasattr(cognition_result.match, "identity"):
        parts.append(f"cog:{cognition_result.match.identity}")
    elif isinstance(cognition_result, Mapping):
        parts.append(f"cog:{json.dumps(cognition_result, sort_keys=True, separators=(',', ':'), default=str)}")
    else:
        parts.append(f"cog:{str(cognition_result)}")

    # Gate hash
    if hasattr(gate_result, "decision") and hasattr(gate_result.decision, "verdict"):
        parts.append(f"gate:{gate_result.decision.verdict}")
    elif isinstance(gate_result, Mapping):
        parts.append(f"gate:{json.dumps(gate_result, sort_keys=True, separators=(',', ':'), default=str)}")
    else:
        parts.append(f"gate:{str(gate_result)}")

    # Ledger hash
    if hasattr(history_ledger, "stable_hash"):
        parts.append(f"ledger:{history_ledger.stable_hash}")
    elif isinstance(history_ledger, Mapping) and "stable_hash" in history_ledger:
        parts.append(f"ledger:{history_ledger['stable_hash']}")
    else:
        parts.append(f"ledger:{str(history_ledger)}")

    combined = "|".join(sorted(parts))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def build_sid_frame_from_qec_state(
    cognition_result: Any,
    gate_result: Any,
    history_ledger: Any,
) -> SIDFrame:
    """Build a deterministic SID frame from QEC state inputs.

    Voice mapping:
        voice1 -> cognition confidence carrier
        voice2 -> gate severity / rollback risk
        voice3 -> drift monitor state

    All parameters are derived deterministically from the combined
    state hash, ensuring byte-identical output for identical input.
    """
    confidence = _clamp(_extract_confidence(cognition_result), 0.0, 1.0)
    severity = _clamp(_extract_gate_severity(gate_result), 0.0, 1.0)
    drift = _clamp(_extract_drift_score(history_ledger), 0.0, 1.0)

    state_hash = _build_state_hash(cognition_result, gate_result, history_ledger)

    # Voice 1: Cognition confidence carrier — triangle wave
    # Higher confidence -> higher frequency
    v1_freq = SID_FREQ_MIN + confidence * (SID_FREQ_MAX - SID_FREQ_MIN)
    v1_attack = _hash_to_range(state_hash, ATTACK_MS_MIN, ATTACK_MS_MAX, "v1a")
    v1_decay = _hash_to_range(state_hash, DECAY_MS_MIN, DECAY_MS_MAX, "v1d")
    v1_sustain = _clamp(confidence, SUSTAIN_MIN, SUSTAIN_MAX)
    v1_release = _hash_to_range(state_hash, RELEASE_MS_MIN, RELEASE_MS_MAX, "v1r")

    voice1 = SIDVoice(
        frequency_hz=v1_freq,
        waveform="triangle",
        attack_ms=v1_attack,
        decay_ms=v1_decay,
        sustain_level=v1_sustain,
        release_ms=v1_release,
    )

    # Voice 2: Gate severity / rollback risk — sawtooth wave
    # Higher severity -> lower frequency (ominous)
    v2_freq = SID_FREQ_MAX - severity * (SID_FREQ_MAX - SID_FREQ_MIN)
    v2_attack = _hash_to_range(state_hash, ATTACK_MS_MIN, ATTACK_MS_MAX, "v2a")
    v2_decay = _hash_to_range(state_hash, DECAY_MS_MIN, DECAY_MS_MAX, "v2d")
    v2_sustain = _clamp(severity, SUSTAIN_MIN, SUSTAIN_MAX)
    v2_release = _hash_to_range(state_hash, RELEASE_MS_MIN, RELEASE_MS_MAX, "v2r")

    voice2 = SIDVoice(
        frequency_hz=v2_freq,
        waveform="sawtooth",
        attack_ms=v2_attack,
        decay_ms=v2_decay,
        sustain_level=v2_sustain,
        release_ms=v2_release,
    )

    # Voice 3: Drift monitor state — pulse wave
    # Higher drift -> noise waveform (instability), otherwise pulse
    v3_waveform = "noise" if drift > 0.7 else "pulse"
    v3_freq = _hash_to_range(state_hash, SID_FREQ_MIN, SID_FREQ_MAX * 0.5, "v3f")
    v3_attack = _hash_to_range(state_hash, ATTACK_MS_MIN, ATTACK_MS_MAX, "v3a")
    v3_decay = _hash_to_range(state_hash, DECAY_MS_MIN, DECAY_MS_MAX, "v3d")
    v3_sustain = _clamp(1.0 - drift, SUSTAIN_MIN, SUSTAIN_MAX)
    v3_release = _hash_to_range(state_hash, RELEASE_MS_MIN, RELEASE_MS_MAX, "v3r")

    voice3 = SIDVoice(
        frequency_hz=v3_freq,
        waveform=v3_waveform,
        attack_ms=v3_attack,
        decay_ms=v3_decay,
        sustain_level=v3_sustain,
        release_ms=v3_release,
    )

    # Ring mod enabled when both severity and drift are elevated
    ring_mod = severity > 0.5 and drift > 0.3

    # Filter mode derived from state
    drift_bucket = _hash_to_float(state_hash, "filter_mode")
    if drift_bucket < 0.33:
        filt_mode = "lowpass"
    elif drift_bucket < 0.66:
        filt_mode = "bandpass"
    else:
        filt_mode = "highpass"

    # Cutoff and resonance from state hash
    cutoff = _hash_to_range(state_hash, FILTER_CUTOFF_MIN, FILTER_CUTOFF_MAX, "cutoff")
    reso = _hash_to_range(state_hash, RESONANCE_MIN, RESONANCE_MAX, "resonance")

    return SIDFrame(
        voice1=voice1,
        voice2=voice2,
        voice3=voice3,
        ring_mod_enabled=ring_mod,
        filter_mode=filt_mode,
        cutoff_hz=cutoff,
        resonance=reso,
    )


# ---------------------------------------------------------------------------
# Waveform rendering
# ---------------------------------------------------------------------------


def render_sid_waveform(
    frame: SIDFrame,
    sample_rate: int = SAMPLE_RATE,
    num_samples: int = NUM_SAMPLES,
) -> Tuple[float, ...]:
    """Render a SID frame into a deterministic audio buffer.

    Mixes 3 voices, optionally applies ring modulation between
    voice1 and voice3 (faithful to 6581 voice 3 ring mod routing),
    then applies the SID filter.
    """
    buf1 = _render_voice(frame.voice1, sample_rate, num_samples,
                         noise_salt="v1_noise")
    buf2 = _render_voice(frame.voice2, sample_rate, num_samples,
                         noise_salt="v2_noise")
    buf3 = _render_voice(frame.voice3, sample_rate, num_samples,
                         noise_salt="v3_noise")

    # Ring modulation: voice1 * voice3 replaces voice1 (6581 faithful)
    if frame.ring_mod_enabled:
        buf1 = apply_sid_ring_modulation(buf1, buf3)

    # Mix voices (equal weight, normalized)
    mixed = tuple(
        (buf1[i] + buf2[i] + buf3[i]) / 3.0
        for i in range(num_samples)
    )

    # Apply SID filter
    filtered = apply_sid_filter(
        mixed, frame.filter_mode, frame.cutoff_hz,
        frame.resonance, sample_rate,
    )

    return filtered


# ---------------------------------------------------------------------------
# Full sonification cycle
# ---------------------------------------------------------------------------


def run_sid_sonification_cycle(
    cognition_result: Any,
    gate_result: Any,
    history_ledger: Any,
) -> SIDRenderResult:
    """Execute a complete SID sonification cycle.

    Pipeline:
        QEC State -> Frame Build -> Waveform Render -> Hash -> Result

    Deterministic: identical inputs always produce identical results.
    """
    frame = build_sid_frame_from_qec_state(
        cognition_result, gate_result, history_ledger,
    )
    audio_buffer = render_sid_waveform(frame)
    spectral_hash = compute_sid_spectral_hash(audio_buffer)

    drift_overlay = _extract_drift_score(history_ledger)

    # Stable hash: covers frame config + audio content
    frame_canonical = json.dumps(
        {
            "v1_freq": frame.voice1.frequency_hz,
            "v1_waveform": frame.voice1.waveform,
            "v2_freq": frame.voice2.frequency_hz,
            "v2_waveform": frame.voice2.waveform,
            "v3_freq": frame.voice3.frequency_hz,
            "v3_waveform": frame.voice3.waveform,
            "ring_mod": frame.ring_mod_enabled,
            "filter_mode": frame.filter_mode,
            "cutoff_hz": frame.cutoff_hz,
            "resonance": frame.resonance,
            "spectral_hash": spectral_hash,
            "engine_version": ENGINE_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    stable_hash = hashlib.sha256(frame_canonical.encode("utf-8")).hexdigest()

    return SIDRenderResult(
        audio_buffer=audio_buffer,
        spectral_hash=spectral_hash,
        drift_overlay=drift_overlay,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Export bundle
# ---------------------------------------------------------------------------


def export_sid_bundle(result: SIDRenderResult) -> Dict[str, Any]:
    """Export a SID render result as a deterministic dictionary bundle.

    Suitable for serialization, logging, and replay verification.
    """
    return {
        "engine_version": ENGINE_VERSION,
        "sample_count": len(result.audio_buffer),
        "spectral_hash": result.spectral_hash,
        "drift_overlay": result.drift_overlay,
        "stable_hash": result.stable_hash,
        "audio_buffer_hash": compute_sid_spectral_hash(result.audio_buffer),
        "deterministic": True,
    }
