from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import zipfile

import pytest

import qec.analysis.game_world_intake_contract as mod
from qec.analysis.game_world_intake_contract import (
    GameWorldArchive,
    GameWorldCorpusManifest,
    GameWorldIntakeReceipt,
    _validate_archive_entry_path,
    build_game_world_archive,
    build_game_world_corpus_manifest,
    build_game_world_intake_receipt,
    validate_game_world_archive,
    validate_game_world_corpus_manifest,
    validate_game_world_intake_receipt,
)


def _make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)


def test_game_world_archive_determinism(tmp_path):
    p = tmp_path / "wolfenstein.zip"
    _make_zip(p, [("main.py", "print(1)"), ("a/readme.md", "x")])
    hashes = [build_game_world_archive(str(p)).archive_manifest_hash for _ in range(10)]
    assert len(set(hashes)) == 1


def test_archive_sha256_changes_when_zip_bytes_change(tmp_path):
    p1 = tmp_path / "a.zip"
    p2 = tmp_path / "b.zip"
    _make_zip(p1, [("main.py", "print(1)")])
    _make_zip(p2, [("main.py", "print(2)")])
    a1 = build_game_world_archive(str(p1))
    a2 = build_game_world_archive(str(p2))
    assert a1.archive_sha256 != a2.archive_sha256
    assert a1.archive_manifest_hash != a2.archive_manifest_hash


def test_zip_path_traversal_rejected(tmp_path):
    for i, name in enumerate(["../evil.py", "/abs/path.py", "C:/bad.py", "subdir/C:/evil.py", "nested/Z:/bad.py"]):
        p = tmp_path / f"bad{i}.zip"
        _make_zip(p, [(name, "x")])
        with pytest.raises(ValueError, match="UNSAFE_ARCHIVE_PATH"):
            build_game_world_archive(str(p))


def test_non_zip_rejected(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="INVALID_ARCHIVE_FORMAT"):
        build_game_world_archive(str(p))


def test_missing_archive_rejected(tmp_path):
    with pytest.raises(ValueError, match="ARCHIVE_NOT_FOUND"):
        build_game_world_archive(str(tmp_path / "missing.zip"))


def test_detected_languages_entrypoints_assets_are_sorted(tmp_path):
    p = tmp_path / "mix.zip"
    _make_zip(p, [("z/train.py", "x"), ("a/file.java", "x"), ("b/README.md", "x"), ("c/im.png", "x"), ("d/cfg.json", "{}")])
    a = build_game_world_archive(str(p))
    assert a.detected_languages == tuple(sorted(a.detected_languages))
    assert a.detected_entrypoints == tuple(sorted(a.detected_entrypoints))
    assert a.detected_asset_types == tuple(sorted(a.detected_asset_types))


def test_world_family_detection(tmp_path):
    cases = {
        "wolfenstein-main.zip": "RAYCAST_FPS",
        "doomengine.zip": "DOOMLIKE_2_5D",
        "atari.zip": "ATARI_RL",
        "easyAI-main.zip": "ABSTRACT_STRATEGY",
        "gdx-ai-main.zip": "GAME_AI_FRAMEWORK",
        "universe-main.zip": "PIXEL_ACTION_INTERFACE",
        "mug-diffusion.zip": "RHYTHM_GENERATIVE",
        "monopoly-main.zip": "BOARD_ECONOMIC_STRATEGY",
        "other.zip": "UNKNOWN",
    }
    for name, expected in cases.items():
        p = tmp_path / name
        _make_zip(p, [("x.txt", "x")])
        assert build_game_world_archive(str(p)).world_family == expected


def test_intake_warnings(tmp_path):
    p = tmp_path / "unknown.zip"
    _make_zip(p, [("main.py", "x"), ("m.ckpt", "x"), ("level.wad", "x"), ("n.ipynb", "{}"), ("build.gradle", "x")])
    warnings = build_game_world_archive(str(p)).intake_warnings
    assert "CONTAINS_EXECUTABLE_ENTRYPOINT" in warnings
    assert "CONTAINS_MODEL_WEIGHTS" in warnings
    assert "CONTAINS_DOOM_WAD" in warnings
    assert "CONTAINS_NATIVE_BUILD_FILES" in warnings
    assert "CONTAINS_JUPYTER_NOTEBOOK" in warnings
    assert "UNKNOWN_WORLD_FAMILY" in warnings


def test_corpus_manifest_sorting_determinism(tmp_path):
    a = tmp_path / "b.zip"
    b = tmp_path / "a.zip"
    _make_zip(a, [("x.txt", "1")])
    _make_zip(b, [("x.txt", "2")])
    m1 = build_game_world_corpus_manifest([str(a), str(b)])
    m2 = build_game_world_corpus_manifest([str(b), str(a)])
    assert m1.archives == m2.archives
    assert m1.corpus_manifest_hash == m2.corpus_manifest_hash


def test_duplicate_archive_name_rejected(tmp_path):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    p1 = d1 / "same.zip"
    p2 = d2 / "same.zip"
    _make_zip(p1, [("a.txt", "1")])
    _make_zip(p2, [("a.txt", "2")])
    with pytest.raises(ValueError, match="DUPLICATE_ARCHIVE"):
        build_game_world_corpus_manifest([str(p1), str(p2)])


def test_intake_receipt_execution_not_allowed(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    manifest = build_game_world_corpus_manifest([str(p)])
    with pytest.raises(ValueError, match="EXECUTION_NOT_ALLOWED"):
        build_game_world_intake_receipt(manifest, "a" * 64, execution_allowed=True)


def test_intake_receipt_determinism(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    manifest = build_game_world_corpus_manifest([str(p)])
    assert build_game_world_intake_receipt(manifest, "a" * 64).receipt_hash == build_game_world_intake_receipt(manifest, "a" * 64).receipt_hash


def test_hash_tamper_detected(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    a = build_game_world_archive(str(p))
    m = build_game_world_corpus_manifest([str(p)])
    r = build_game_world_intake_receipt(m, "a" * 64)
    object.__setattr__(a, "archive_manifest_hash", "b" * 64)
    object.__setattr__(m, "corpus_manifest_hash", "b" * 64)
    object.__setattr__(r, "receipt_hash", "b" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_game_world_archive(a)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_game_world_corpus_manifest(m)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_game_world_intake_receipt(r)


def test_malformed_hash_rejected(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    a = build_game_world_archive(str(p))
    object.__setattr__(a, "archive_manifest_hash", "XYZ")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_game_world_archive(a)


def test_artifacts_are_frozen(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    a = build_game_world_archive(str(p))
    with pytest.raises(FrozenInstanceError):
        a.archive_name = "x"


def test_no_execution_or_import_behavior():
    for name in ("run_game", "execute_archive", "import_world", "load_module_from_zip"):
        assert not hasattr(mod, name)
    src = (Path(mod.__file__)).read_text(encoding="utf-8").lower()
    for banned in (".extract(", ".extractall(", "importlib", "__import__(", "runpy", "subprocess", "exec(", "eval("):
        assert banned not in src
    assert "zipfile.zipfile(path, \"r\")" in src


def test_corpus_manifest_deep_validates_child_archives(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    archive = build_game_world_archive(str(p))
    bad_hash_archive = build_game_world_archive(str(p))
    object.__setattr__(bad_hash_archive, "archive_manifest_hash", "b" * 64)
    payload = {"archives": [bad_hash_archive.to_dict()], "total_archives": 1, "total_files": bad_hash_archive.file_count, "total_uncompressed_bytes": bad_hash_archive.total_uncompressed_bytes}
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        GameWorldCorpusManifest(archives=(bad_hash_archive,), total_archives=1, total_files=bad_hash_archive.file_count, total_uncompressed_bytes=bad_hash_archive.total_uncompressed_bytes, corpus_manifest_hash=mod.sha256_hex(payload))
    malformed_hash_archive = build_game_world_archive(str(p))
    object.__setattr__(malformed_hash_archive, "archive_manifest_hash", "bad")
    payload_bad = {"archives": [malformed_hash_archive.to_dict()], "total_archives": 1, "total_files": malformed_hash_archive.file_count, "total_uncompressed_bytes": malformed_hash_archive.total_uncompressed_bytes}
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        GameWorldCorpusManifest(archives=(malformed_hash_archive,), total_archives=1, total_files=malformed_hash_archive.file_count, total_uncompressed_bytes=malformed_hash_archive.total_uncompressed_bytes, corpus_manifest_hash=mod.sha256_hex(payload_bad))


def test_invalid_archive_tuple_contents(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    a = build_game_world_archive(str(p))
    for field, value in (("top_level_entries", None), ("detected_languages", 123), ("detected_entrypoints", ["a"]), ("detected_asset_types", ("a", 1)), ("intake_warnings", ("a", 1))):
        object.__setattr__(a, field, value)
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_game_world_archive(a)
        object.__setattr__(a, field, build_game_world_archive(str(p)).to_dict()[field])


def test_invalid_world_family_rejected(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    a = build_game_world_archive(str(p))
    object.__setattr__(a, "world_family", "BAD")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_game_world_archive(a)


def test_negative_count_fields_rejected(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    a = build_game_world_archive(str(p))
    object.__setattr__(a, "file_count", -1)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_game_world_archive(a)
    m = build_game_world_corpus_manifest([str(p)])
    object.__setattr__(m, "total_archives", -1)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_game_world_corpus_manifest(m)


def test_execution_allowed_must_be_exact_false(tmp_path):
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    manifest = build_game_world_corpus_manifest([str(p)])
    for bad in (0, "", [], None):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            build_game_world_intake_receipt(manifest, "a" * 64, execution_allowed=bad)
    with pytest.raises(ValueError, match="EXECUTION_NOT_ALLOWED"):
        build_game_world_intake_receipt(manifest, "a" * 64, execution_allowed=True)
    assert build_game_world_intake_receipt(manifest, "a" * 64, execution_allowed=False).execution_allowed is False
    r = build_game_world_intake_receipt(manifest, "a" * 64)
    object.__setattr__(r, "execution_allowed", 0)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_game_world_intake_receipt(r)


def test_build_intake_receipt_rejects_invalid_corpus_manifest(tmp_path):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_game_world_intake_receipt("not-a-manifest", "a" * 64)
    p = tmp_path / "a.zip"
    _make_zip(p, [("a.txt", "1")])
    manifest = build_game_world_corpus_manifest([str(p)])
    object.__setattr__(manifest, "corpus_manifest_hash", "b" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        build_game_world_intake_receipt(manifest, "a" * 64)


def test_native_build_warning_requires_build_file(tmp_path):
    src_only = tmp_path / "src_only.zip"
    _make_zip(src_only, [("a.java", "1"), ("b.c", "1"), ("c.cpp", "1")])
    assert "CONTAINS_NATIVE_BUILD_FILES" not in build_game_world_archive(str(src_only)).intake_warnings
    for idx, name in enumerate(["build.gradle", "pom.xml", "Makefile", "CMakeLists.txt"]):
        p = tmp_path / f"build{idx}.zip"
        _make_zip(p, [(name, "x")])
        assert "CONTAINS_NATIVE_BUILD_FILES" in build_game_world_archive(str(p)).intake_warnings


def test_validator_non_object_rejection():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_game_world_archive(123)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_game_world_corpus_manifest("bad")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_game_world_intake_receipt(None)


def test_private_path_validator_for_null_and_embedded_drive():
    with pytest.raises(ValueError, match="UNSAFE_ARCHIVE_PATH"):
        _validate_archive_entry_path("bad\x00name.py")
    with pytest.raises(ValueError, match="UNSAFE_ARCHIVE_PATH"):
        _validate_archive_entry_path("subdir/C:/evil.py")
