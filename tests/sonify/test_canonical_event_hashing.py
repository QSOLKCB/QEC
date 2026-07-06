import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from qec.sonify import symbolic_sonification_runtime_skeleton_hash
from qec.sonify.canonical import canonical_json
from qec.sonify.events import (
    build_symbolic_event,
    build_symbolic_event_stream,
    symbolic_event_hash,
    symbolic_event_payload,
    symbolic_event_stream_payload,
    validate_symbolic_event,
    validate_symbolic_event_stream,
)

HEX = set("0123456789abcdef")


def assert_sha(value):
    assert len(value) == 64
    assert set(value) <= HEX


def event(event_id="e1", start=0, lane="a"):
    return build_symbolic_event(event_id, "SYMBOLIC_MARKER", "DIAG", start, 4, lane, {"level": 1}, ("tag",))


def stream(events=None):
    return build_symbolic_event_stream("s1", events or (event(),))


def test_event_and_stream_validate_and_hash_shapes():
    e = validate_symbolic_event(event())
    s = validate_symbolic_event_stream(stream((e,)))
    assert_sha(e.event_hash)
    assert_sha(s.stream_hash)
    assert_sha(symbolic_sonification_runtime_skeleton_hash())


def test_round_trip_payload_json_payload_object_preserves_stream_hash():
    original = stream((event("e2", 2, "b"), event("e1", 1, "a")))
    encoded = canonical_json(symbolic_event_stream_payload(original))
    payload = json.loads(encoded)
    rebuilt_events = [build_symbolic_event(**{k: v for k, v in item.items() if k != "event_hash"}) for item in payload["events"]]
    rebuilt = build_symbolic_event_stream(payload["stream_id"], rebuilt_events)
    assert rebuilt.stream_hash == original.stream_hash


def test_repeated_builds_and_different_input_order_are_deterministic():
    a = stream((event("e2", 5, "b"), event("e1", 0, "a")))
    b = stream((event("e1", 0, "a"), event("e2", 5, "b")))
    c = stream((event("e2", 5, "b"), event("e1", 0, "a")))
    assert a.stream_hash == b.stream_hash == c.stream_hash
    assert [e.event_id for e in a.events] == ["e1", "e2"]


def test_duplicate_event_ids_and_forged_counts_hashes_rejected():
    with pytest.raises(ValueError):
        stream((event("e1", 0, "a"), event("e1", 1, "a")))
    e = event()
    with pytest.raises(ValueError):
        validate_symbolic_event(replace(e, event_hash="0" * 64))
    s = stream((e,))
    with pytest.raises(ValueError):
        validate_symbolic_event_stream(replace(s, event_count=99))
    with pytest.raises(ValueError):
        validate_symbolic_event_stream(replace(s, stream_hash="0" * 64))


def test_event_hash_recomputed_not_copied():
    e = event()
    payload = symbolic_event_payload(e, include_hash=False)
    payload["symbolic_token"] = "Ouroboros"
    assert symbolic_event_hash(payload) != e.event_hash


def test_hash_seed_stability_subprocess():
    code = "from qec.sonify.events import build_symbolic_event, build_symbolic_event_stream; e=build_symbolic_event('e1','SYMBOLIC_MARKER','DIAG',0,1,'lane',{'a':1},('x',)); print(build_symbolic_event_stream('s1',(e,)).stream_hash)"
    outputs = []
    for seed in ("0", "1"):
        proc = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True, env={"PYTHONPATH": "src", "PYTHONHASHSEED": seed})
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1]


def test_no_forbidden_runtime_identity_terms_in_sources():
    text = "\n".join(Path(path).read_text(encoding="utf-8") for path in ("src/qec/sonify/__init__.py", "src/qec/sonify/events.py", "src/qec/sonify/mapping.py", "src/qec/sonify/canonical.py"))
    for token in ("random", "datetime", "uuid", "os.environ", "builtins.hash", " hash("):
        assert token not in text


def test_boundary_imports_and_absent_future_files():
    forbidden = {"qec.decoder", "qec.os", "qec.codes", "qec.syndrome", "socket", "requests", "urllib", "http.client", "mido", "pretty_midi", "music21", "pyaudio", "sounddevice", "wave", "openai", "anthropic", "google.generativeai", "numpy", "scipy", "pandas", "polars", "torch", "tensorflow", "jax", "qiskit", "stim", "pymatching", "qldpc"}
    for path in Path("src/qec/sonify").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert imports.isdisjoint(forbidden), (path, imports & forbidden)
    assert not Path("src/qec/sonify/packs.py").exists()
    assert not Path("src/qec/sonify/packs").exists()
    assert not Path("src/qec/sonify/midi.py").exists()
    assert not Path("src/qec/sonify/cli.py").exists()
