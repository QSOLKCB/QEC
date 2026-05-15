import pytest

from scripts.update_readme_release_metadata import update_readme

README = """# QSOLKCB / QEC

## 📦 Release & Research
[![Latest](https://img.shields.io/badge/stable-v164.2-success)](https://github.com/QSOLKCB/QEC/releases/tag/v164.2)
[![Branch](https://img.shields.io/badge/branch-v164.2%20canonical-purple)]()
Current release line: **v164.2**  
Current frontier: **v164.3 — x**  
Active arc: **v164.x — y**
Completed arc: **v163.x — z**
Repository status is current through **v164.2**.

## v163.x → v164.x Capability Summary
- v164.2 → CachedCanonicalKernelReceipt

## 📚 DOIs
IMMUT

## ⚡ Quickstart
IMMUT

## Testing
IMMUT

## Commands
IMMUT

## IRC Operator Surface
IMMUT

## 🧾 Attribution
IMMUT

## References
IMMUT

## Author
IMMUT
"""


def test_readme_updates_required_mutable_sections():
    out = update_readme(README, "v165.3.2", "v165.4 — Frontier", "v164.x — Arc")
    assert "stable-v165.3.2-success" in out
    assert "branch-v165.3.2%20canonical" in out
    assert "Current release line: **v165.3.2**" in out
    # Assert Active arc is updated with correct arc derivation
    assert "Active arc: **v165.3.x — Invariant-Based Heavy Dependency Optimization**" in out


def test_readme_noop_update_rejected():
    with pytest.raises(ValueError, match="README_UPDATE_NO_EFFECT"):
        update_readme("no mutable tokens", "v165.3.2", "v165.4 — Frontier", "v164.x — Arc")


def test_readme_immutable_sections_unchanged():
    out = update_readme(README, "v165.3.2", "v165.4 — Frontier", "v164.x — Arc")
    assert "## ⚡ Quickstart\nIMMUT" in out
    assert "## Commands\nIMMUT" in out


def test_readme_stable_badge_and_release_line_updated():
    out = update_readme(README, "v165.3.2", "v165.4 — Frontier", "v164.x — Arc")
    assert "stable-v165.3.2-success" in out and "Current release line: **v165.3.2**" in out
    # Assert Active arc is updated with correct arc derivation
    assert "Active arc: **v165.3.x — Invariant-Based Heavy Dependency Optimization**" in out
    # Assert badge link target is updated
    assert "releases/tag/v165.3.2" in out


def test_readme_frontier_updated():
    out = update_readme(README, "v165.3.2", "v165.4 — Frontier", "v164.x — Arc")
    assert "Current frontier: **v165.4 — Frontier**" in out


def test_readme_capability_summary_updated():
    out = update_readme(README, "v165.3.2", "v165.4 — Frontier", "v164.x — Arc")
    assert "## v163.x → v164.x Capability Summary" in out
