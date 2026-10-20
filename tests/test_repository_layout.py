"""Preservation checks for the root-layout migration, not scientific assertions."""
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "archive" / "root-cleanup-manifest.json"


def test_moved_files_match_manifest_and_preserve_historical_bytes():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_commit"] == "0df7fb3602c3c79b4e9ef0b367037bc951c679f3"
    originals, destinations = set(), set()
    for entry in manifest["moves"]:
        old, new = entry["original_path"], entry["path"]
        assert old not in originals and new not in destinations
        originals.add(old)
        destinations.add(new)
        assert not (ROOT / old).exists()
        path = ROOT / new
        payload = path.read_bytes()
        assert len(payload) == entry["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        if new.startswith(("archive/", "requirements/")):
            assert entry["byte_identical"]
            assert entry["sha256"] == entry["original_sha256"]
            blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
            assert hashlib.sha1(blob).hexdigest() == entry["original_git_blob"]
        if entry["git_mode"] == "100755":
            assert path.stat().st_mode & stat.S_IXUSR


def test_no_historical_media_or_profile_files_in_root():
    assert not any(ROOT.glob("*.mp3"))
    assert not any(ROOT.glob("*.zip"))
    assert not any(ROOT.glob("*.jpg"))
    assert not any(ROOT.glob("*requirements*.txt"))
    assert (ROOT / "AGENTS.md").is_file()
    assert (ROOT / "PACKAGE_README.md").is_file()


def test_audio_consumers_resolve_preserved_inputs_without_skip():
    from scripts.run_sonic_analysis import _AUDIO_FILES
    assert len(_AUDIO_FILES) == 3
    assert all(Path(p).is_file() for p in _AUDIO_FILES)
    audio = ROOT / "archive" / "sonification" / "audio"
    for name in ("Deterministic Traversal Mesh.mp3", "E8 Triality Topology.mp3",
                 "Quantum Coherence Threshold (v1).mp3", "Quantum Coherence Threshold (v2).mp3"):
        assert (audio / name).is_file()


def test_immutable_v3_baseline_keeps_original_location_and_hash():
    baseline = ROOT / "qec_data_prepared.csv"
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == (
        "80f1f74ad02c2ac7fdaf5e6a6df1611f55df3f4294cc11edfc889f5d7fe41b0a"
    )


def test_release_manifest_loads_from_organized_location():
    from scripts.update_release_notes import load_release_history
    assert load_release_history(ROOT)


def test_sonic_output_stays_outside_archive(monkeypatch):
    from scripts import run_sonic_analysis as script
    seen = {}
    def fake_run(inputs, output_root):
        seen["output_root"] = output_root
        return dict(n_files=len(inputs), mean_duration=0, mean_energy=0,
                    mean_centroid=0, variance_centroid=0)
    monkeypatch.setattr(script, "run_sonic_batch_analysis", fake_run)
    script.main()
    assert Path(seen["output_root"]) == ROOT / "artifacts" / "sonic"
