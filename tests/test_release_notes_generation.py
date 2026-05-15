import pytest

from scripts.update_release_notes import (
    GENERIC_SUMMARY,
    HEADER,
    MAX_ENTRY_CHARS,
    ReleaseHistoryError,
    _validate_release_history,
    build_release_notes_from_tags,
    discover_release_tags,
    generate_release_notes,
)


def _headings(notes: str) -> list[str]:
    return [line[3:] for line in notes.splitlines() if line.startswith("## ")]


def test_release_notes_exact_release_set_match():
    discovered = discover_release_tags(["v165.3.1", "v165.3", "43.0.0", "v0.1"])
    notes = generate_release_notes(discovered)
    assert set(_headings(notes)) == set(discovered)


def test_release_notes_rejects_synthetic_versions():
    discovered = discover_release_tags(["v165.3.1", "v165.3", "v165.2.1"])
    notes = generate_release_notes(discovered)
    headings = set(_headings(notes))
    assert "v165.9" not in headings
    assert "v164.9" not in headings


def test_release_notes_preserves_early_history():
    discovered = discover_release_tags(["v0.1", "v0.2", "v0.3", "v165.3.1"])
    notes = generate_release_notes(discovered)
    headings = set(_headings(notes))
    assert {"v0.1", "v0.2", "v0.3"}.issubset(headings)


def test_release_notes_preserves_non_v_releases():
    discovered = discover_release_tags(["43.0.0", "42.0.0", "38.0.0", "v165.3.1"])
    notes = generate_release_notes(discovered)
    headings = set(_headings(notes))
    assert {"43.0.0", "42.0.0", "38.0.0"}.issubset(headings)


def test_release_notes_no_duplicate_entries():
    discovered = discover_release_tags(["v1.0", "v1.0", "v1.0.0", "v1.0.0"])
    notes = generate_release_notes(discovered)
    headings = _headings(notes)
    assert len(headings) == len(set(headings))


def test_release_notes_duplicate_entry_error():
    discovered = ["v1.0.0", "v1.0.0"]
    with pytest.raises(ReleaseHistoryError, match="DUPLICATE_RELEASE_ENTRY"):
        _validate_release_history(discovered, discovered)


def test_release_notes_reverse_chronological_order():
    discovered = discover_release_tags(["v0.3", "v165.2.1", "v165.3", "v165.3.1", "43.0.0"])
    notes = generate_release_notes(discovered)
    assert _headings(notes) == discovered


def test_release_notes_summary_length_bound():
    discovered = discover_release_tags(["v1.2.3"])
    notes = generate_release_notes(discovered)
    summary = notes.splitlines()[3]
    assert len(summary) <= MAX_ENTRY_CHARS


def test_release_notes_idempotent_generation():
    discovered = discover_release_tags(["v165.3.1", "v165.3", "43.0.0", "v0.3"])
    assert generate_release_notes(discovered) == generate_release_notes(discovered)


def test_release_notes_no_tags_fails_closed():
    with pytest.raises(ReleaseHistoryError, match="NO_RELEASE_TAGS_FOUND"):
        build_release_notes_from_tags([], min_release_count=1)


def test_release_notes_invalid_min_release_count():
    with pytest.raises(ReleaseHistoryError, match="INVALID_MIN_RELEASE_COUNT"):
        build_release_notes_from_tags(["v1.0.0"], min_release_count=0)
    with pytest.raises(ReleaseHistoryError, match="INVALID_MIN_RELEASE_COUNT"):
        build_release_notes_from_tags(["v1.0.0"], min_release_count=-1)


def test_release_notes_existing_history_not_destroyed_on_partial_source(tmp_path):
    existing = tmp_path / "RELEASE_NOTES.md"
    existing.write_text("# RELEASE NOTES\n\n## v999.0\nkeep\n", encoding="utf-8")

    with pytest.raises(ReleaseHistoryError, match="RELEASE_HISTORY_INCOMPLETE"):
        build_release_notes_from_tags(["v165.3.1"], min_release_count=2)

    assert existing.read_text(encoding="utf-8") == "# RELEASE NOTES\n\n## v999.0\nkeep\n"


def test_release_notes_count_matches_source():
    discovered = discover_release_tags(["v165.3.1", "v165.3", "43.0.0", "42.0.0", "v0.1"])
    notes = generate_release_notes(discovered)
    assert len(_headings(notes)) == len(discovered)


def test_release_notes_header_and_generic_summary_are_stable():
    discovered = discover_release_tags(["v1.0.0", "test-tag"])
    notes = generate_release_notes(discovered)
    assert notes.startswith(HEADER)
    assert GENERIC_SUMMARY in notes
