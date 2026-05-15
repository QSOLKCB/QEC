import pytest

from scripts.update_readme_release_metadata import ERR, update_readme


README = """# QSOLKCB / QEC\n\n## 📦 Release & Research\n[![Latest](https://img.shields.io/badge/stable-v164.2-success)]()\n[![OSF Registration](https://img.shields.io/badge/OSF-Registration-blue)](https://osf.io/sjk7b)\nCurrent release line: **v164.2**  \nCurrent frontier: **v164.3 — x**  \nCompleted arc: **v163.x — y**\nRepository status is current through **v164.2 → z**.\n\n## 📚 DOIs\nA\n\n## ⚡ Quickstart\nIMMUT\n\n## Commands\nIMMUT2\n\n## IRC Operator Surface\nIMMUT3\n\n## 🧾 Attribution\nIMMUT4\n\n## References\nIMMUT5\n\n## Author\nIMMUT6\n"""


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
