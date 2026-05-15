import json

import pytest

from scripts.update_release_notes import (
    GENERIC_SUMMARY,
    ReleaseHistoryError,
    _validate_release_history,
    build_release_notes_from_tags,
    generate_release_notes_from_history,
    load_release_history,
)


def _headings(notes: str) -> list[str]:
    return [line[3:] for line in notes.splitlines() if line.startswith("## ")]


def test_release_history_json_exact_set_match():
    history = [{"tag": "v165.3.2"}, {"tag": "43.0.0"}]
    notes = generate_release_notes_from_history(history)
    assert set(_headings(notes)) == {"v165.3.2", "43.0.0"}


def test_release_notes_exact_set_match():
    manifest_tags = ["v1.0.0", "v0.1"]
    generated_tags = ["v0.1", "v1.0.0"]
    _validate_release_history(manifest_tags, generated_tags)


def test_release_notes_no_synthetic_versions():
    notes = generate_release_notes_from_history([{"tag": "v165.3.2"}, {"tag": "43.0.0"}, {"tag": "v0.3"}])
    assert "v900." not in notes


def test_release_notes_preserves_early_history():
    notes = generate_release_notes_from_history([{"tag": "v0.1"}, {"tag": "v0.2"}, {"tag": "v0.3"}])
    assert _headings(notes) == ["v0.1", "v0.2", "v0.3"]


def test_release_notes_preserves_non_v_releases():
    notes = generate_release_notes_from_history([{"tag": "43.0.0"}, {"tag": "42.0.0"}, {"tag": "38.0.0"}])
    assert _headings(notes) == ["43.0.0", "42.0.0", "38.0.0"]


def test_release_notes_no_duplicate_entries():
    with pytest.raises(ReleaseHistoryError, match="DUPLICATE_RELEASE_ENTRY"):
        generate_release_notes_from_history([{"tag": "v1.0.0"}, {"tag": "v1.0.0"}])


def test_release_notes_idempotent_generation():
    history = [{"tag": "v0.3", "title": "t", "body": "b"}, {"tag": "43.0.0", "title": "", "body": ""}]
    assert generate_release_notes_from_history(history) == generate_release_notes_from_history(history)


def test_release_notes_reverse_chronological_order():
    history = [{"tag": "v3.0.0"}, {"tag": "v2.0.0"}, {"tag": "v1.0.0"}]
    notes = generate_release_notes_from_history(history)
    assert _headings(notes) == ["v3.0.0", "v2.0.0", "v1.0.0"]


def test_release_history_json_rejects_duplicates(tmp_path):
    p = tmp_path / "release_history.json"
    p.write_text(json.dumps([{"tag": "v1.0.0"}, {"tag": "v1.0.0"}]), encoding="utf-8")
    with pytest.raises(ReleaseHistoryError, match="DUPLICATE_RELEASE_ENTRY"):
        load_release_history(tmp_path)


def test_release_history_json_rejects_malformed_entries(tmp_path):
    p = tmp_path / "release_history.json"
    p.write_text("{bad", encoding="utf-8")
    with pytest.raises(ReleaseHistoryError, match="INVALID_RELEASE_HISTORY_JSON"):
        load_release_history(tmp_path)
    p.write_text(json.dumps({"tag": "v1.0.0"}), encoding="utf-8")
    with pytest.raises(ReleaseHistoryError, match="INVALID_RELEASE_HISTORY_SCHEMA"):
        load_release_history(tmp_path)
    p.write_text(json.dumps([{"title": "missing"}]), encoding="utf-8")
    with pytest.raises(ReleaseHistoryError, match="MALFORMED_RELEASE_ENTRY"):
        load_release_history(tmp_path)


def test_release_notes_preserves_full_body():
    """Verify that full body content is preserved without truncation."""
    body = "x" * 1000
    notes = generate_release_notes_from_history([{"tag": "v1.0.0", "body": body}])
    # Body should be fully preserved, not truncated
    assert body in notes


def test_release_notes_generation_from_canonical_archive(tmp_path):
    history = [{"tag": "v2.0.0", "title": "T2", "body": "B2"}, {"tag": "v1.0.0"}]
    (tmp_path / "release_history.json").write_text(json.dumps(history), encoding="utf-8")
    loaded = load_release_history(tmp_path)
    notes = generate_release_notes_from_history(loaded)
    assert _headings(notes) == ["v2.0.0", "v1.0.0"]


def test_release_notes_generation_offline(tmp_path):
    (tmp_path / "release_history.json").write_text(json.dumps([{"tag": "v1.0.0"}]), encoding="utf-8")
    loaded = load_release_history(tmp_path)
    notes = generate_release_notes_from_history(loaded)
    assert GENERIC_SUMMARY in notes


def test_release_notes_no_partial_overwrite():
    with pytest.raises(ReleaseHistoryError, match="RELEASE_HISTORY_INCOMPLETE"):
        build_release_notes_from_tags(["v1.0.0"], min_release_count=2)


def test_readme_boundary_enforcement_still_passes():
    from scripts.update_readme_release_metadata import update_readme

    readme = "## 📦 Release & Research\nCurrent release line: **v164.2**  "
    out = update_readme(readme, "v165.3.2", "v165.4 — Frontier", "v164.x — Arc")
    assert "Current release line: **v165.3.2**" in out
