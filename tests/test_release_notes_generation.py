from scripts.update_release_notes import MAX_ENTRY_CHARS, generate_release_notes, parse_release_tags


def test_ordering_and_no_duplicates():
    tags = parse_release_tags(["v1.0", "v2.0", "v2.0.1", "v1.0.0", "junk"])
    notes = generate_release_notes(tags)
    headings = [ln for ln in notes.splitlines() if ln.startswith("## ")]
    assert headings == ["## v2.0.1", "## v2.0", "## v1.0"]
    assert notes.count("## v1.0") == 1


def test_idempotent_generation():
    tags = parse_release_tags(["v3.2", "v3.1"])
    a = generate_release_notes(tags)
    b = generate_release_notes(tags)
    assert a == b


def test_summary_bounded():
    tags = parse_release_tags(["v10.1.2"])
    notes = generate_release_notes(tags)
    summary = notes.splitlines()[3]
    assert len(summary) <= MAX_ENTRY_CHARS
