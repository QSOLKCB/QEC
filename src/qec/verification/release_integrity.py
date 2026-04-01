"""Deterministic release-integrity verification for the QEC TUI installer flow.

Stdlib only.  No decoder changes.  No TUI feature changes.
Verifies that install paths, binary versions, and release tags are consistent
across the repository artifacts (Cargo.toml, README.md, install.sh).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical installer constants — single source of truth
# ---------------------------------------------------------------------------

REPO_SLUG = "QSOLKCB/QEC"
BINARY_NAME = "qec-tui"
INSTALL_DIR = "/usr/local/bin"
ASSET_NAME = "qec-tui-linux-x86_64.tar.gz"
CANONICAL_CURL_URL = (
    "https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh"
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntegrityResult:
    """Immutable verification outcome."""
    check: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class IntegrityReport:
    """Aggregated verification report."""
    results: Tuple[IntegrityResult, ...]
    all_passed: bool


def _make_report(results: List[IntegrityResult]) -> IntegrityReport:
    return IntegrityReport(
        results=tuple(sorted(results, key=lambda r: r.check)),
        all_passed=all(r.passed for r in results),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_repo_root(hint: Optional[str] = None) -> Path:
    """Resolve the repository root directory.

    If *hint* is provided it is returned as-is.  Otherwise walks upward from
    this file looking for ``tui/install.sh`` as a sentinel.

    This is the single canonical implementation — scripts and tests should
    import this rather than reimplementing the walk.
    """
    if hint is not None:
        return Path(hint)
    cursor = Path(__file__).resolve().parent
    for _ in range(10):
        if (cursor / "tui" / "install.sh").exists():
            return cursor
        cursor = cursor.parent
    raise FileNotFoundError("Cannot locate repository root")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Public verification functions
# ---------------------------------------------------------------------------

def verify_install_path_consistency(repo_root: Optional[str] = None) -> IntegrityReport:
    """Verify that install.sh, README, and Cargo.toml reference consistent paths.

    Checks:
    - install.sh REPO matches QSOLKCB/QEC
    - install.sh BINARY_NAME matches qec-tui
    - install.sh INSTALL_DIR matches /usr/local/bin
    - README curl URL points to the same install.sh path used in the repo
    """
    root = resolve_repo_root(repo_root)
    results: List[IntegrityResult] = []

    install_sh = root / "tui" / "install.sh"
    readme = root / "README.md"

    # --- install.sh checks ---
    if not install_sh.exists():
        results.append(IntegrityResult("install_sh_exists", False, "tui/install.sh not found"))
        return _make_report(results)

    results.append(IntegrityResult("install_sh_exists", True, str(install_sh)))

    sh_text = _read_text(install_sh)

    repo_match = re.search(r'^REPO="([^"]+)"', sh_text, re.MULTILINE)
    repo_val = repo_match.group(1) if repo_match else ""
    results.append(IntegrityResult(
        "install_sh_repo",
        repo_val == REPO_SLUG,
        f"REPO={repo_val!r}",
    ))

    bin_match = re.search(r'^BINARY_NAME="([^"]+)"', sh_text, re.MULTILINE)
    bin_val = bin_match.group(1) if bin_match else ""
    results.append(IntegrityResult(
        "install_sh_binary_name",
        bin_val == BINARY_NAME,
        f"BINARY_NAME={bin_val!r}",
    ))

    dir_match = re.search(r'^INSTALL_DIR="([^"]+)"', sh_text, re.MULTILINE)
    dir_val = dir_match.group(1) if dir_match else ""
    results.append(IntegrityResult(
        "install_sh_install_dir",
        dir_val == INSTALL_DIR,
        f"INSTALL_DIR={dir_val!r}",
    ))

    # --- README curl path ---
    if not readme.exists():
        results.append(IntegrityResult("readme_exists", False, "README.md not found"))
        return _make_report(results)

    results.append(IntegrityResult("readme_exists", True, str(readme)))

    readme_text = _read_text(readme)
    results.append(IntegrityResult(
        "readme_curl_url",
        CANONICAL_CURL_URL in readme_text,
        f"expected URL present: {CANONICAL_CURL_URL in readme_text}",
    ))

    return _make_report(results)


def verify_binary_version_consistency(repo_root: Optional[str] = None) -> IntegrityReport:
    """Verify that Cargo.toml declares a parseable version for qec-tui."""
    root = resolve_repo_root(repo_root)
    results: List[IntegrityResult] = []

    cargo_toml = root / "tui" / "Cargo.toml"
    if not cargo_toml.exists():
        results.append(IntegrityResult("cargo_toml_exists", False, "tui/Cargo.toml not found"))
        return _make_report(results)

    results.append(IntegrityResult("cargo_toml_exists", True, str(cargo_toml)))

    cargo_text = _read_text(cargo_toml)

    name_match = re.search(r'^name\s*=\s*"([^"]+)"', cargo_text, re.MULTILINE)
    name_val = name_match.group(1) if name_match else ""
    results.append(IntegrityResult(
        "cargo_package_name",
        name_val == BINARY_NAME,
        f"name={name_val!r}",
    ))

    ver_match = re.search(r'^version\s*=\s*"([^"]+)"', cargo_text, re.MULTILINE)
    ver_val = ver_match.group(1) if ver_match else ""
    semver_ok = bool(re.fullmatch(r"\d+\.\d+\.\d+", ver_val))
    results.append(IntegrityResult(
        "cargo_version_semver",
        semver_ok,
        f"version={ver_val!r}",
    ))

    return _make_report(results)


def verify_release_tag_alignment(
    repo_root: Optional[str] = None,
    expected_tag: Optional[str] = None,
) -> IntegrityReport:
    """Verify that install.sh resolves tags via the correct GitHub API endpoint
    and that the download URL template is well-formed.

    If *expected_tag* is given, also checks that the Cargo.toml version matches
    the numeric portion of the tag (e.g. tag ``v106.0.0`` matches version
    ``106.0.0``).
    """
    root = resolve_repo_root(repo_root)
    results: List[IntegrityResult] = []

    install_sh = root / "tui" / "install.sh"
    if not install_sh.exists():
        results.append(IntegrityResult("install_sh_exists", False, "tui/install.sh not found"))
        return _make_report(results)

    sh_text = _read_text(install_sh)

    # Check API URL pattern — install.sh uses shell variable interpolation
    # so accept both the template form and the expanded form.
    api_match = re.search(
        r'API_URL="(https://api\.github\.com/repos/[^"]+)"', sh_text,
    )
    api_val = api_match.group(1) if api_match else ""
    expected_literal = f"https://api.github.com/repos/{REPO_SLUG}/releases/latest"
    expected_template = "https://api.github.com/repos/${REPO}/releases/latest"
    api_ok = api_val in (expected_literal, expected_template)
    results.append(IntegrityResult(
        "api_url_correct",
        api_ok,
        f"API_URL={api_val!r}",
    ))

    # Check download URL template uses ${REPO} and ${TAG}
    dl_pattern = re.search(
        r'DOWNLOAD_URL="(https://github\.com/[^"]+)"', sh_text,
    )
    dl_val = dl_pattern.group(1) if dl_pattern else ""
    results.append(IntegrityResult(
        "download_url_template",
        "${REPO}" in dl_val and "${TAG}" in dl_val,
        f"DOWNLOAD_URL template well-formed: {'${REPO}' in dl_val and '${TAG}' in dl_val}",
    ))

    # Optional: tag-to-Cargo.toml alignment
    if expected_tag is not None:
        cargo_toml = root / "tui" / "Cargo.toml"
        cargo_toml_exists = cargo_toml.exists()
        results.append(IntegrityResult(
            "cargo_toml_exists_for_tag",
            cargo_toml_exists,
            f"{cargo_toml} exists for expected_tag={expected_tag!r}",
        ))

        cargo_text = _read_text(cargo_toml) if cargo_toml_exists else ""
        ver_match = re.search(r'^version\s*=\s*"([^"]+)"', cargo_text, re.MULTILINE)
        cargo_ver = ver_match.group(1) if ver_match else ""
        tag_ver = expected_tag.lstrip("v")
        results.append(IntegrityResult(
            "tag_cargo_version_match",
            cargo_ver == tag_ver,
            f"tag={expected_tag!r} -> {tag_ver!r}, Cargo.toml={cargo_ver!r}",
        ))

    return _make_report(results)
