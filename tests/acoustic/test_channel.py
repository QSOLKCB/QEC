# SPDX-License-Identifier: MPL-2.0
from dataclasses import FrozenInstanceError, replace
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
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


def _exporter(path=None):
    import importlib.util
    if path is None:
        path=Path(__file__).resolve().parents[2]/"scripts/generate_acoustic_channel_evidence.py"
    spec=importlib.util.spec_from_file_location("acoustic_channel_export_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exporter_pins_independent_frozen_source_and_waveform(tmp_path):
    exporter=_exporter()
    identity=exporter.generate(tmp_path/"evidence")
    for d,row in zip((3,4),identity["sources"]):
        expected=json.loads((FIXTURES/f"clean_q{d}.json").read_text())
        assert row["source_sha256"]==expected["source"]["sha256"]
        assert row["wav_sha256"]==expected["wav_sha256"]
        assert (tmp_path/"evidence"/f"clean_q{d}.wav").read_bytes()==(FIXTURES/f"clean_q{d}.wav").read_bytes()


def test_exporter_rejects_self_consistent_substituted_source(tmp_path,monkeypatch):
    exporter=_exporter()
    real=exporter.AcousticSource
    monkeypatch.setattr(exporter,"AcousticSource",lambda config,symbols,source_sha256:real(config,(2,2,2),source_sha256))
    with pytest.raises(ValueError,match="INVALID_INPUT"):
        exporter.generate(tmp_path/"substituted-source")
    assert not (tmp_path/"substituted-source"/"channel_identity.json").exists()


def test_exporter_rejects_substituted_wav(tmp_path,monkeypatch):
    exporter=_exporter()
    real=exporter.wav_bytes
    monkeypatch.setattr(exporter,"wav_bytes",lambda pcm:real(PCM((0,)*len(pcm.samples),pcm.sample_rate)))
    with pytest.raises(ValueError,match="INVALID_INPUT"):
        exporter.generate(tmp_path/"substituted-wav")
    assert not (tmp_path/"substituted-wav"/"channel_identity.json").exists()


def test_exporter_metadata_uses_script_repository_not_caller_cwd(tmp_path,monkeypatch):
    exporter=_exporter()
    expected=subprocess.check_output(["git","rev-parse","HEAD"],cwd=exporter.ROOT,text=True).strip()
    unrelated=tmp_path/"unrelated"
    unrelated.mkdir()
    subprocess.run(["git","init","-q",str(unrelated)],check=True)
    monkeypatch.chdir(unrelated)
    exporter.generate(tmp_path/"evidence")
    metadata=json.loads((tmp_path/"evidence"/"environment.json").read_text())
    assert metadata["source_commit_sha"]==expected


@pytest.mark.parametrize("preexisting_change", ["none", "tracked", "untracked"])
def test_exporter_snapshots_git_before_unignored_output(tmp_path,monkeypatch,preexisting_change):
    # A real, committed Git fixture is needed: mocking status misses the output
    # directory itself changing Git's answer during generation.
    root=tmp_path/"checkout"
    (root/"scripts").mkdir(parents=True)
    script=Path(__file__).resolve().parents[2]/"scripts/generate_acoustic_channel_evidence.py"
    shutil.copy2(script,root/"scripts"/script.name)
    shutil.copytree(FIXTURES,root/"tests/acoustic/fixtures")
    tracked=root/"tracked.txt"
    tracked.write_text("committed input\n",encoding="utf-8")
    subprocess.run(["git","init","-q",str(root)],check=True)
    subprocess.run(["git","add","."],cwd=root,check=True)
    subprocess.run(["git","-c","user.name=Acoustic Fixture","-c","user.email=acoustic-fixture@example.invalid",
                    "-c","commit.gpgsign=false","-c","core.hooksPath=/dev/null",
                    "commit","-qm","Commit test fixture"],cwd=root,check=True)
    expected_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()
    if preexisting_change=="tracked":
        tracked.write_text("preexisting modification\n",encoding="utf-8")
    elif preexisting_change=="untracked":
        (root/"untracked.txt").write_text("preexisting untracked input\n",encoding="utf-8")
    # Loading the copied script binds its own repository root; bytecode output
    # must not itself dirty the fixture before the generator starts.
    monkeypatch.setattr(sys,"dont_write_bytecode",True)
    exporter=_exporter(root/"scripts"/script.name)
    before=subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()
    assert bool(before)==(preexisting_change!="none")
    unrelated=tmp_path/"caller"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    output=root/"unignored-evidence"
    assert not output.exists()
    identity=exporter.generate(output)
    metadata=json.loads((output/"environment.json").read_text(encoding="utf-8"))
    assert metadata["source_commit_sha"]==expected_commit
    assert metadata["clean_working_tree"]==(preexisting_change=="none")
    assert metadata["canonical_channel_identity_sha256"]==identity["sha256"]
    after=subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True)
    assert "?? unignored-evidence/" in after
    if preexisting_change=="tracked":
        assert " M tracked.txt" in after
    elif preexisting_change=="untracked":
        assert "?? untracked.txt" in after
    for d in (3,4):
        expected=json.loads((FIXTURES/f"clean_q{d}.json").read_text(encoding="utf-8"))
        assert json.loads((output/f"clean_q{d}.json").read_text(encoding="utf-8"))==expected["source"]
        assert (output/f"clean_q{d}.wav").read_bytes()==(FIXTURES/f"clean_q{d}.wav").read_bytes()
