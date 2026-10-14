# SPDX-License-Identifier: MPL-2.0
"""Versioned integer-PCM harmonic alphabet and immutable source contracts.

The 48-entry quantized sine is normative, not runtime libm output. Integer
products divide by 32767 with Python floor division. Carrier phase resets at
block boundaries. This is a declared classical sampled-signal experiment,
not a model of quantum dynamics, microphones, or physical phonons.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

from ._common import CLAIM_BOUNDARY, digest, exact, integer, read, seal, sequence

SINE48 = (0, 4277, 8481, 12539, 16383, 19947, 23170, 25996, 28377, 30273, 31650, 32487,
          32767, 32487, 31650, 30273, 28377, 25996, 23170, 19947, 16383, 12539, 8481, 4277,
          0, -4277, -8481, -12539, -16383, -19947, -23170, -25996, -28377, -30273, -31650, -32487,
          -32767, -32487, -31650, -30273, -28377, -25996, -23170, -19947, -16383, -12539, -8481, -4277)
MAX_SYMBOLS = 32
MAX_SAMPLES = 73728


@dataclass(frozen=True)
class ChannelConfig:
    alphabet_size: int = 3
    sample_rate: int = 48000
    cycles: int = 4
    amplitude: int = 1024
    harmonics: tuple[int, ...] = (1, 2, 3, 4)
    repetitions: int = 3

    def __post_init__(self):
        integer(self.alphabet_size, 4, 3)
        if type(self.sample_rate) is not int or self.sample_rate not in (24000, 48000):
            raise ValueError("INVALID_INPUT")
        integer(self.cycles, 8, 1)
        integer(self.amplitude, 2048, 64)
        integer(self.repetitions, 3, 1)
        harmonics = sequence(self.harmonics, 8)
        if not harmonics:
            raise ValueError("INVALID_INPUT")
        for h in harmonics:
            integer(h, 8, 1)
        if harmonics != tuple(sorted(set(harmonics))):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "harmonics", harmonics)

    @property
    def block_samples(self) -> int:
        return 48 * self.cycles

    def as_dict(self):
        return seal("channel-config", alphabet_size=self.alphabet_size, sample_rate=self.sample_rate,
                    cycles=self.cycles, amplitude=self.amplitude, harmonics=list(self.harmonics),
                    repetitions=self.repetitions, pcm_layout="mono_signed_16bit_little_endian",
                    phase_origin="zero_at_each_repetition", rounding="integer_floor_divide_32767",
                    lut_sha256=hashlib.sha256(struct.pack("<48h", *SINE48)).hexdigest(),
                    carrier_hz=self.sample_rate // 48,
                    informative_orders=[h for h in self.harmonics if math.gcd(h, self.alphabet_size) == 1],
                    partial_orders=[h for h in self.harmonics if 1 < math.gcd(h, self.alphabet_size) < self.alphabet_size],
                    dark_orders=[h for h in self.harmonics if h % self.alphabet_size == 0])

    @classmethod
    def from_dict(cls, value):
        return read(cls, value, ("alphabet_size", "sample_rate", "cycles", "amplitude", "harmonics", "repetitions"))


@dataclass(frozen=True)
class PCM:
    samples: tuple[int, ...]
    sample_rate: int = 48000

    def __post_init__(self):
        samples = sequence(self.samples, MAX_SAMPLES)
        for sample in samples:
            integer(sample, 32767, -32768)
        if type(self.sample_rate) is not int or self.sample_rate not in (24000, 48000):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "samples", samples)

    @property
    def pcm16le(self) -> bytes:
        return struct.pack("<" + "h" * len(self.samples), *self.samples)

    def as_dict(self):
        return seal("pcm", sample_rate=self.sample_rate, sample_count=len(self.samples),
                    layout="mono_signed_16bit_little_endian", pcm_hex=self.pcm16le.hex(),
                    pcm_sha256=hashlib.sha256(self.pcm16le).hexdigest())

    @classmethod
    def from_dict(cls, value):
        try:
            raw = value["pcm_hex"]
            if type(raw) is not str or len(raw) > 4 * MAX_SAMPLES or len(raw) % 4:
                raise ValueError("INVALID_INPUT")
            data = bytes.fromhex(raw)
            result = cls(struct.unpack("<" + "h" * (len(data) // 2), data), value["sample_rate"])
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError, struct.error) as exc:
            raise ValueError("INVALID_INPUT") from exc


def symbol_block(symbol: int, config: ChannelConfig) -> tuple[int, ...]:
    if type(config) is not ChannelConfig:
        raise ValueError("INVALID_INPUT")
    integer(symbol, config.alphabet_size - 1)
    return tuple(sum(config.amplitude * SINE48[(h * (n + symbol * (48 // config.alphabet_size))) % 48] // 32767
                     for h in config.harmonics) for n in range(config.block_samples))


def encode_symbols(symbols, config: ChannelConfig = ChannelConfig()) -> PCM:
    if type(config) is not ChannelConfig:
        raise ValueError("INVALID_INPUT")
    symbols = sequence(symbols, MAX_SYMBOLS)
    if not symbols:
        raise ValueError("INVALID_INPUT")
    samples = []
    for symbol in symbols:
        block = symbol_block(symbol, config)
        for _ in range(config.repetitions):
            samples.extend(block)
    return PCM(samples, config.sample_rate)


def wav_bytes(pcm: PCM) -> bytes:
    if type(pcm) is not PCM:
        raise ValueError("INVALID_INPUT")
    data = pcm.pcm16le
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, pcm.sample_rate, pcm.sample_rate * 2, 2, 16) +
            b"data" + struct.pack("<I", len(data)) + data)


def read_wav(data: bytes) -> PCM:
    """Read only the exact bounded canonical WAV layout; reject extra chunks."""
    if type(data) is not bytes or not 44 <= len(data) <= 44 + 2 * MAX_SAMPLES or (len(data) - 44) % 2:
        raise ValueError("INVALID_INPUT")
    try:
        rate = struct.unpack_from("<I", data, 24)[0]
        pcm = PCM(struct.unpack("<" + "h" * ((len(data) - 44) // 2), data[44:]), rate)
        if wav_bytes(pcm) != data:
            raise ValueError("INVALID_INPUT")
        return pcm
    except (ValueError, struct.error) as exc:
        raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class AcousticSource:
    config: ChannelConfig
    measured_symbols: tuple[int, ...]
    source_sha256: str

    def __post_init__(self):
        if type(self.config) is not ChannelConfig:
            raise ValueError("INVALID_INPUT")
        symbols = sequence(self.measured_symbols, MAX_SYMBOLS)
        encode_symbols(symbols, self.config)
        digest(self.source_sha256)
        object.__setattr__(self, "measured_symbols", symbols)

    def as_dict(self):
        from qec.sonify.canonical import canonical_sha256
        return seal("source", config=self.config.as_dict(), measured_symbols=list(self.measured_symbols),
                    source_sha256=self.source_sha256,
                    fixture_sha256=canonical_sha256({"measured_symbols": list(self.measured_symbols)}),
                    encoded_sound=encode_symbols(self.measured_symbols, self.config).as_dict(),
                    source_authority="caller_declared_synthetic_fixture", claim_boundary=dict(CLAIM_BOUNDARY))

    @classmethod
    def from_dict(cls, value):
        return read(cls, value, ("config", "measured_symbols", "source_sha256"), {"config": ChannelConfig.from_dict})


def validate_source(value, *, expected_source_sha256: str) -> dict:
    digest(expected_source_sha256)
    source = AcousticSource.from_dict(value)
    if value["sha256"] != expected_source_sha256:
        raise ValueError("INVALID_INPUT")
    return seal("source-validation", source_sha256=value["sha256"], all_passed=True,
                encoded_pcm_recomputed=True, physical_provenance_authenticated=False)
