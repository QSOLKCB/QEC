from pathlib import Path

import pytest

from scripts.update_readme_release_metadata import _validate_boundaries, update_readme

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEST_RELEASE = "v170.3.0"
FRONTIER = "v171.0 — PanelSeparatedControlExchange"
ACTIVE_ARC = "v171.x–v176.x — Deterministic Telecommunications Switching Lineage"
COMPLETED_ARC = "v170.x — Exact Ququart Evidence / NEXUS / Strowger Routing"
REPOSITORY_STATUS = "v170.3.0 → DeterministicStrowgerSyndromeExchange"


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
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
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
    assert "stable-v170.3.0-success" in out
    assert "releases/tag/v170.3.0" in out
    assert "branch-v170.3.0%20canonical-purple" in out
    assert "Current release line: **v170.3.0**" in out
    assert "Current frontier: **v171.0 — PanelSeparatedControlExchange**" in out
    assert "Active arc: **v171.x–v176.x — Deterministic Telecommunications Switching Lineage**" in out
    assert "Completed arc: **v170.x — Exact Ququart Evidence / NEXUS / Strowger Routing**" in out
    assert (
        "Repository status is current through "
        "**v170.3.0 → DeterministicStrowgerSyndromeExchange**."
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
    assert "releases/tag/v170.3.0" in out


def test_readme_updater_rejects_no_effect_when_stale():
    stale_without_patterns = """## 📦 Release & Research
No mutable release metadata tokens are present here.
"""
    with pytest.raises(ValueError, match="README_UPDATE_NO_EFFECT"):
        _run_update(stale_without_patterns)


def test_readme_immutable_sections_unchanged():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    bad = readme.replace("## ⚡ Quickstart", "## ⚡ Quickstart\nMUTATED", 1)
    with pytest.raises(ValueError, match="README_BOUNDARY_VIOLATION"):
        _validate_boundaries(readme, bad)


def test_osf_stays_in_dois():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    out = _run_update(readme)
    dois_idx = out.index("## 📚 DOIs")
    release_idx = out.index("## 📦 Release & Research")
    osf_idx = out.index("OSF-Registration")
    quickstart_idx = out.index("## ⚡ Quickstart")
    assert release_idx < dois_idx < osf_idx < quickstart_idx


def test_capability_summary_heading_current():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    out = _run_update(readme)
    assert "## Capability Summary" in out
    assert "## v163.x → v164.x Capability Summary" not in out
