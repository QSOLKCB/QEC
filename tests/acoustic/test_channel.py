# SPDX-License-Identifier: MPL-2.0
from dataclasses import FrozenInstanceError, replace
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import wave

import pytest
from qec.experiments.acoustic.channel import AcousticSource, ChannelConfig, PCM, SINE48, encode_symbols, read_wav, symbol_block, validate_source, wav_bytes
from qec.sonify.canonical import canonical_json, canonical_sha256

FIXTURES = Path(__file__).parent / "fixtures"


def rehash(value):
    if isinstance(value, list):
        return [rehash(v) for v in value]
    if isinstance(value, dict):
        result = {k: rehash(v) for k, v in value.items() if k != "sha256"}
        if "sha256" in value:
            result["sha256"] = canonical_sha256(result)
        return result
    return value


@pytest.mark.parametrize("d", [3, 4])
def test_all_symbols_harmonic_classes_independent_clean_demodulation(d):
    config = ChannelConfig(d, harmonics=tuple(range(1, 9)))
    for s in range(d):
        block = symbol_block(s, config)
        # Independent float Fourier projection is TEST ONLY; it is neither the
        # normative integer generator nor production receiver implementation.
        for h in config.harmonics:
            cosine = sum(x * math.cos(2 * math.pi * h * n / 48) for n, x in enumerate(block))
            sine = sum(x * math.sin(2 * math.pi * h * n / 48) for n, x in enumerate(block))
            angle = math.atan2(cosine, sine)
            expected = 2 * math.pi * h * s / d
            assert abs(math.sin(angle - expected)) < 0.002
            assert math.cos(angle - expected) > 0.999
            assert math.hypot(cosine, sine) > config.amplitude * config.block_samples * 0.49
    meta = config.as_dict()
    assert meta["dark_orders"] == [h for h in range(1,9) if h % d == 0]
    assert meta["partial_orders"] == ([] if d == 3 else [2,6])


@pytest.mark.parametrize("d", [3, 4])
def test_frozen_clean_source_and_wav_known_answers(d):
    fixture = json.loads((FIXTURES / f"clean_q{d}.json").read_text())
    source = AcousticSource.from_dict(fixture["source"])
    pcm = encode_symbols(tuple(range(d)), ChannelConfig(d))
    assert source.as_dict() == fixture["source"]
    assert pcm.as_dict() == fixture["source"]["encoded_sound"]
    assert hashlib.sha256(wav_bytes(pcm)).hexdigest() == fixture["wav_sha256"]
    assert wav_bytes(pcm) == (FIXTURES / f"clean_q{d}.wav").read_bytes()
    assert read_wav(wav_bytes(pcm)) == pcm
    with wave.open(io.BytesIO(wav_bytes(pcm)), "rb") as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, 48000)
        assert stream.readframes(stream.getnframes()) == pcm.pcm16le
    assert validate_source(source.as_dict(), expected_source_sha256=source.as_dict()["sha256"])["all_passed"]


@pytest.mark.parametrize("changes", [{"alphabet_size":True},{"alphabet_size":2},{"alphabet_size":5},{"sample_rate":44100},
    {"cycles":0},{"cycles":9},{"amplitude":True},{"amplitude":2049},{"repetitions":4},{"repetitions":0},
    {"harmonics":()},{"harmonics":(1,1)},{"harmonics":(2,1)},{"harmonics":(0,)},{"harmonics":(9,)},
    {"harmonics":(1.0,)},{"harmonics":{1,2}}])
def test_configuration_rejects_invalid_values(changes):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ChannelConfig(**changes)


@pytest.mark.parametrize("symbols", [(),(True,),(3,),(-1,),(1.0,),range(3),(0,)*33])
def test_symbols_strict_bounded(symbols):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        encode_symbols(symbols)


@pytest.mark.parametrize("samples", [(True,),(-32769,),(32768,),(float("nan"),),(0,)*73729])
def test_pcm_rejects_invalid_values(samples):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        PCM(samples)


def test_immutable_detached_snapshots():
    symbols = [0,1,2]
    harmonics = [1,2,3,4]
    source = AcousticSource(ChannelConfig(harmonics=harmonics), symbols, "a"*64)
    symbols[0] = 1
    harmonics.clear()
    assert source.measured_symbols == (0,1,2)
    assert source.config.harmonics == (1,2,3,4)
    with pytest.raises(FrozenInstanceError):
        source.source_sha256 = "b"*64
    snapshot = source.as_dict()
    snapshot["measured_symbols"][0] = 2
    assert source.measured_symbols[0] == 0


def test_rehashed_source_tamper_unknown_fields_and_foreign_source_fail():
    source = AcousticSource(ChannelConfig(), (0,1,2), "a"*64).as_dict()
    for mutate in (lambda s: s.update(extra=True), lambda s: s["encoded_sound"].update(pcm_hex="0000"),
                   lambda s: s["claim_boundary"].update(physical_quantum_protection=True)):
        damaged = deepcopy(source)
        mutate(damaged)
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            AcousticSource.from_dict(rehash(damaged))
    foreign = AcousticSource(ChannelConfig(), (2,1,0), "a"*64).as_dict()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_source(foreign, expected_source_sha256=source["sha256"])
    bad = deepcopy(source)
    bad["config"]["alphabet_size"] = 3.0
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AcousticSource.from_dict(bad)


def test_wav_fail_closed_no_ambient_layout():
    raw = wav_bytes(encode_symbols((0,1)))
    for candidate in (b"",raw[:-1],raw+b"extra",b"NOPE"+raw[4:],raw[:22]+b"\x02\x00"+raw[24:]):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            read_wav(candidate)


def test_fresh_process_hash_seed_determinism():
    code = "from qec.experiments.acoustic.channel import *;print(AcousticSource(ChannelConfig(),(0,1,2),'a'*64).as_dict()['sha256'])"
    outputs = [subprocess.check_output([sys.executable,"-c",code], env={**os.environ,"PYTHONHASHSEED":seed}).strip() for seed in ("1","123")]
    assert outputs[0] == outputs[1]
