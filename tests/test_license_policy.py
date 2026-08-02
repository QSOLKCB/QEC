# SPDX-License-Identifier: MPL-2.0
"""Regression checks for the repository-wide MPL 2.0 licence policy."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SPDX_RE = re.compile(r"SPDX-License-Identifier:\s*([^\s*]+)")
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".h",
    ".hpp",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "build",
    "dist",
    "external",
    "node_modules",
    "third_party",
    "vendor",
}


def _owned_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        files.append(path)
    return sorted(files)


def test_root_license_is_mpl_2_0() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("Mozilla Public License Version 2.0\n")
    assert "3.1. Distribution of Source Form" in text
    assert "Exhibit A - Source Code Form License Notice" in text
    assert "mozilla.org/MPL/2.0/" in text


def test_package_metadata_declares_mpl_2_0() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = { text = "MPL-2.0" }' in text
    assert "Mozilla Public License 2.0 (MPL 2.0)" in text


def test_repository_policy_covers_qec_authored_material() -> None:
    text = (ROOT / "LICENSE_POLICY.md").read_text(encoding="utf-8")
    assert "SPDX license identifier:** `MPL-2.0`" in text
    assert "Historical releases" in text
    assert "Third-party material" in text


def test_explicit_qec_spdx_headers_are_mpl_2_0() -> None:
    violations: list[str] = []
    for path in _owned_text_files():
        try:
            head = "\n".join(
                path.read_text(encoding="utf-8").splitlines()[:20]
            )
        except UnicodeDecodeError:
            continue
        for match in SPDX_RE.finditer(head):
            identifier = match.group(1)
            if identifier != "MPL-2.0":
                violations.append(
                    f"{path.relative_to(ROOT)} declares {identifier}"
                )
    assert not violations, "legacy or conflicting SPDX headers:\n" + "\n".join(
        violations
    )
