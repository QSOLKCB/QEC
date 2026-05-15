import pytest

from scripts.update_release_notes import (
    MIN_CANONICAL_RELEASE_COUNT,
    ReleaseHistoryError,
    _validate_release_history,
    build_release_notes_from_tags,
    generate_release_notes_from_history,
)


def _headings(notes: str) -> list[str]:
    return [line[3:] for line in notes.splitlines() if line.startswith("## ")]


def test_release_notes_exact_release_set_match():
    history = [{"tag": "v165.3.2", "title": "", "body": ""}, {"tag": "43.0.0", "title": "", "body": ""}]
    notes = generate_release_notes_from_history(history)
    assert set(_headings(notes)) == {"v165.3.2", "43.0.0"}


def test_release_notes_preserve_early_and_non_v_tags():
    history = [
        {"tag": "v0.1", "title": "", "body": ""},
        {"tag": "v0.2", "title": "", "body": ""},
        {"tag": "v0.3", "title": "", "body": ""},
        {"tag": "38.0.0", "title": "", "body": ""},
        {"tag": "42.0.0", "title": "", "body": ""},
        {"tag": "43.0.0", "title": "", "body": ""},
    ]
    notes = generate_release_notes_from_history(history)
    headings = set(_headings(notes))
    assert {"v0.1", "v0.2", "v0.3", "38.0.0", "42.0.0", "43.0.0"}.issubset(headings)


def test_release_notes_no_synthetic_versions_generated():
    history = [{"tag": "v165.3.2", "title": "", "body": ""}, {"tag": "43.0.0", "title": "", "body": ""}, {"tag": "v0.3", "title": "", "body": ""}]
    notes = generate_release_notes_from_history(history)
    assert "v900." not in notes


def test_release_notes_idempotent_generation():
    history = [{"tag": "v0.3", "title": "", "body": ""}, {"tag": "43.0.0", "title": "", "body": ""}]
    assert generate_release_notes_from_history(history) == generate_release_notes_from_history(history)


def test_release_history_mismatch_and_duplicate_validation():
    with pytest.raises(ReleaseHistoryError, match="RELEASE_HISTORY_MISMATCH"):
        _validate_release_history(["v1.0.0"], ["v1.0.1"])
    with pytest.raises(ReleaseHistoryError, match="DUPLICATE_RELEASE_ENTRY"):
        _validate_release_history(["v1.0.0"], ["v1.0.0", "v1.0.0"])


def test_release_history_no_tags_and_incomplete_fail_closed():
    with pytest.raises(ReleaseHistoryError, match="NO_RELEASE_TAGS_FOUND"):
        build_release_notes_from_tags([], min_release_count=MIN_CANONICAL_RELEASE_COUNT)
    with pytest.raises(ReleaseHistoryError, match="RELEASE_HISTORY_INCOMPLETE"):
        build_release_notes_from_tags(["v1.0.0"], min_release_count=MIN_CANONICAL_RELEASE_COUNT)


def test_release_history_invalid_min_release_count_rejected():
    with pytest.raises(ReleaseHistoryError, match="INVALID_MIN_RELEASE_COUNT"):
        build_release_notes_from_tags(["v1.0.0"], min_release_count=MIN_CANONICAL_RELEASE_COUNT - 1)


def test_release_history_invalid_json_raises_controlled_error(tmp_path):
    from scripts.update_release_notes import load_release_history_tags
    # Test malformed JSON
    bad_json = tmp_path / "release_history.json"
    bad_json.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(ReleaseHistoryError, match="INVALID_RELEASE_HISTORY_JSON"):
        load_release_history_tags(tmp_path)
    # Test non-list JSON
    bad_json.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ReleaseHistoryError, match="INVALID_RELEASE_HISTORY_SCHEMA"):
        load_release_history_tags(tmp_path)
