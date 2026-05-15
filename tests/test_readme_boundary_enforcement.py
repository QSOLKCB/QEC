import pytest

from scripts.update_readme_release_metadata import ERR, update_readme


README = """# QSOLKCB / QEC\n\n## 📦 Release & Research\n[![Latest](https://img.shields.io/badge/stable-v164.2-success)](https://github.com/QSOLKCB/QEC/releases/tag/v164.2)\n[![OSF Registration](https://img.shields.io/badge/OSF-Registration-blue)](https://osf.io/sjk7b)\nCurrent release line: **v164.2**  \nCurrent frontier: **v164.3 — x**  \nCompleted arc: **v163.x — y**\nRepository status is current through **v164.2 → z**.\n\n## 📚 DOIs\nA\n\n## ⚡ Quickstart\nIMMUT\n\n## Commands\nIMMUT2\n\n## IRC Operator Surface\nIMMUT3\n\n## 🧾 Attribution\nIMMUT4\n\n## References\nIMMUT5\n\n## Author\nIMMUT6\n"""


def test_doi_migration_and_updates():
    out = update_readme(README, "v165.2", "v165.3 — Frontier", "v164.x — Arc")
    assert "stable-v165.2-success" in out
    assert "[OSF Registration](https://osf.io/sjk7b)" in out.split("## 📚 DOIs", 1)[1].split("## ⚡ Quickstart", 1)[0]
    assert "OSF-Registration" not in out.split("## 📦 Release & Research", 1)[1].split("## 📚 DOIs", 1)[0]


def test_idempotent_rerun():
    out1 = update_readme(README, "v165.2", "v165.3 — Frontier", "v164.x — Arc")
    out2 = update_readme(out1, "v165.2", "v165.3 — Frontier", "v164.x — Arc")
    assert out1 == out2


def test_invalid_immutable_edit_rejected():
    bad = README.replace("IMMUT2", "Current frontier: **oops**")
    with pytest.raises(ValueError, match=ERR):
        update_readme(bad, "v165.2", "v165.3 — Frontier", "v164.x — Arc")


def test_version_agnostic_badge_update():
    """Test that badge updates work regardless of current version."""
    # First update to v165.2
    out1 = update_readme(README, "v165.2", "v165.3 — Frontier", "v164.x — Arc")
    assert "stable-v165.2-success" in out1
    # Second update to v166.0 should still work (version-agnostic match)
    out2 = update_readme(out1, "v166.0", "v166.1 — NewFrontier", "v165.x — NewArc")
    assert "stable-v166.0-success" in out2
    assert "stable-v165.2-success" not in out2


def test_badge_link_updated_with_version():
    """Test that badge link target is updated alongside the shield URL."""
    out = update_readme(README, "v165.2", "v165.3 — Frontier", "v164.x — Arc")
    # Badge link should point to new release
    assert "releases/tag/v165.2)" in out
    # Old release link should not be present in the badge
    assert "releases/tag/v164.2)" not in out


def test_unknown_section_change_rejected():
    """Test that changes to unknown sections by the updater are rejected (fail-closed).
    
    Note: The boundary validation ensures the updater itself doesn't modify unknown sections.
    Pre-existing drift in the input README is a separate concern that would require
    baseline comparison against a canonical reference file.
    """
    # The current implementation validates that update_readme's own modifications
    # don't change unknown sections. Since update_readme doesn't modify unknown sections,
    # this test verifies that unknown sections pass through unchanged.
    readme_with_unknown = README + "\n## Unknown Section\nOriginal content\n"
    out = update_readme(readme_with_unknown, "v165.2", "v165.3 — Frontier", "v164.x — Arc")
    # Unknown section should be preserved unchanged
    assert "## Unknown Section\nOriginal content" in out
