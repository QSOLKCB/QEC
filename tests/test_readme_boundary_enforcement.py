from pathlib import Path

import pytest

from scripts.update_readme_release_metadata import _validate_boundaries, update_readme

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEST_RELEASE = "v172.5"
FRONTIER = "v173.0 — Stored-Program Switch Skeleton (unreleased candidate)"
ACTIVE_ARC = "v171.x–v176.x — Deterministic Telecommunications Switching Lineage"
COMPLETED_ARC = "v172.5 — Strowger/Panel/Crossbar Equivalence Battery"
REPOSITORY_STATUS = "v172.5 → StrowgerPanelCrossbarEquivalenceBattery"


def _run_update(text: str) -> str:
    return update_readme(
        text,
        LATEST_RELEASE,
        FRONTIER,
        COMPLETED_ARC,
        ACTIVE_ARC,
        REPOSITORY_STATUS,
    )


def test_readme_updater_byte_identical_when_already_current():
    readme = (REPO_ROOT / "docs/archive/README_BEFORE_V177_1_CLEANUP.md").read_text(encoding="utf-8")
    out = _run_update(readme)
    assert out == readme


def test_readme_updater_mutates_stale_release_metadata():
    stale = """## 📦 Release & Research
[![Latest](https://img.shields.io/badge/stable-v164.2-success)](https://github.com/QSOLKCB/QEC/releases/tag/v164.2)
[![Branch](https://img.shields.io/badge/branch-v164.2%20canonical-purple)]()
Current release line: **v164.2**  
Current frontier: **v164.3 — x**  
Active arc: **v164.x — y**  
Completed arc: **v163.x — z**

Repository status is current through **v164.2 → OldStatus**.
"""
    out = _run_update(stale)
    assert "stable-v172.5-success" in out
    assert "releases/tag/v172.5" in out
    assert "branch-v172.5%20canonical-purple" in out
    assert "Current release line: **v172.5**" in out
    assert "Current frontier: **v173.0 — Stored-Program Switch Skeleton (unreleased candidate)**" in out
    assert "Active arc: **v171.x–v176.x — Deterministic Telecommunications Switching Lineage**" in out
    assert "Completed arc: **v172.5 — Strowger/Panel/Crossbar Equivalence Battery**" in out
    assert (
        "Repository status is current through "
        "**v172.5 → StrowgerPanelCrossbarEquivalenceBattery**."
    ) in out


def test_readme_updater_preserves_independent_lab_release_link():
    stale = """# QSOLKCB / QEC
<a href="https://github.com/QSOLKCB/QEC/releases/tag/v170.1.1">Lab release</a>

## 📦 Release & Research
[![Latest](https://img.shields.io/badge/stable-v164.2-success)](https://github.com/QSOLKCB/QEC/releases/tag/v164.2)
[![Branch](https://img.shields.io/badge/branch-v164.2%20canonical-purple)]()
Current release line: **v164.2**
Current frontier: **v164.3 — x**
Active arc: **v164.x — y**
Completed arc: **v163.x — z**

Repository status is current through **v164.2 → OldStatus**.
"""
    out = _run_update(stale)
    assert "releases/tag/v170.1.1\">Lab release" in out
    assert "[![Latest]" in out
    assert "releases/tag/v172.5" in out


def test_readme_updater_rejects_no_effect_when_stale():
    stale_without_patterns = """## 📦 Release & Research
No mutable release metadata tokens are present here.
"""
    with pytest.raises(ValueError, match="README_UPDATE_NO_EFFECT"):
        _run_update(stale_without_patterns)


def test_readme_immutable_sections_unchanged():
    readme = (REPO_ROOT / "docs/archive/README_BEFORE_V177_1_CLEANUP.md").read_text(encoding="utf-8")
    bad = readme.replace("## ⚡ Quickstart", "## ⚡ Quickstart\nMUTATED", 1)
    with pytest.raises(ValueError, match="README_BOUNDARY_VIOLATION"):
        _validate_boundaries(readme, bad)


def test_osf_stays_in_dois():
    readme = (REPO_ROOT / "docs/archive/README_BEFORE_V177_1_CLEANUP.md").read_text(encoding="utf-8")
    out = _run_update(readme)
    dois_idx = out.index("## 📚 DOIs")
    release_idx = out.index("## 📦 Release & Research")
    osf_idx = out.index("OSF-Registration")
    quickstart_idx = out.index("## ⚡ Quickstart")
    assert release_idx < dois_idx < osf_idx < quickstart_idx


def test_capability_summary_heading_current():
    readme = (REPO_ROOT / "docs/archive/README_BEFORE_V177_1_CLEANUP.md").read_text(encoding="utf-8")
    out = _run_update(readme)
    assert "## Capability Summary" in out
    assert "## v163.x → v164.x Capability Summary" not in out


def test_compact_readme_current_and_next_release():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert update_readme(readme, "v177.1", "unused", "unused") == readme
    updated = update_readme(readme, "v177.2", "unused", "unused")
    assert updated == readme.replace("stable-v177.1-success", "stable-v177.2-success").replace(
        "releases/tag/v177.1", "releases/tag/v177.2"
    ).replace("Current release line: **v177.1**", "Current release line: **v177.2**")
    assert update_readme(updated, "v177.2", "unused", "unused") == updated


@pytest.mark.parametrize("replacement", ["", "Current release line: **broken**"])
def test_compact_readme_missing_metadata_fails_closed(replacement):
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    bad = readme.replace("Current release line: **v177.1**", replacement)
    with pytest.raises(ValueError, match="README_COMPACT_METADATA_INVALID"):
        update_readme(bad, "v177.2", "unused", "unused")


def test_compact_readme_rejects_duplicate_metadata_and_invalid_release():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="README_COMPACT_METADATA_INVALID"):
        update_readme(readme.replace("## Release", "## Release\nCurrent release line: **v177.1**"), "v177.2", "", "")
    with pytest.raises(ValueError, match="README_INVALID_RELEASE"):
        update_readme(readme, "v177.2\nchanged", "", "")


def test_compact_readme_preserves_nonrelease_boundaries():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="README_BOUNDARY_VIOLATION"):
        _validate_boundaries(readme, readme.replace("## Quick start", "## Quick start\nMUTATED"))


@pytest.mark.parametrize("mutation", ["missing_badge", "duplicate_badge", "duplicate_header"])
def test_compact_readme_rejects_ambiguous_badge_or_section(mutation):
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    badge = next(line for line in readme.splitlines() if line.startswith("[![Latest]"))
    if mutation == "missing_badge":
        readme = readme.replace(badge, "")
    elif mutation == "duplicate_badge":
        readme = readme.replace(badge, badge + "\n" + badge)
    else:
        readme += "\n## Release\nCurrent release line: **v177.1**\n"
    with pytest.raises(ValueError, match="README_COMPACT_METADATA_INVALID"):
        update_readme(readme, "v177.2", "", "")


@pytest.mark.parametrize("release", ["v177.2", "v177.2\nINJECTED"])
def test_compact_marker_drift_does_not_fall_back_to_legacy(release):
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    bad = readme.replace("<!-- README-RELEASE:COMPACT -->", "")
    with pytest.raises(ValueError, match="README_COMPACT_METADATA_INVALID"):
        update_readme(bad, release, "", "")


def test_unknown_format_is_not_assumed_legacy():
    with pytest.raises(ValueError, match="README_FORMAT_INVALID"):
        update_readme("# Other\nCurrent release line: **v177.1**\n", "v177.2", "", "")


def test_compact_cli_needs_only_release_and_preserves_legacy_requirements(tmp_path):
    import subprocess
    import sys

    readme = tmp_path / "README.md"
    original = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    readme.write_text(original, encoding="utf-8")
    command = [sys.executable, str(REPO_ROOT / "scripts/update_readme_release_metadata.py"),
               "--readme", str(readme), "--latest-release", "v177.2"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert readme.read_text(encoding="utf-8") == update_readme(original, "v177.2", "", "")
    legacy = (REPO_ROOT / "docs/archive/README_BEFORE_V177_1_CLEANUP.md").read_text(encoding="utf-8")
    readme.write_text(legacy, encoding="utf-8")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert "legacy README requires" in result.stderr
    assert readme.read_text(encoding="utf-8") == legacy
