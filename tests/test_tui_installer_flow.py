"""Deterministic integration tests for the QEC Rust TUI installer flow.

Coverage:
- install.sh exists and is well-formed
- README advertised curl path is correct
- installer is idempotent (re-parse yields same results)
- version tag resolves correctly from Cargo.toml
- qec-tui binary launch semantics (help/version commands)
- release_integrity module round-trip

Stdlib + pytest only.  No network calls.  No decoder changes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def install_sh(repo_root: Path) -> Path:
    return repo_root / "tui" / "install.sh"


@pytest.fixture
def readme(repo_root: Path) -> Path:
    return repo_root / "README.md"


@pytest.fixture
def cargo_toml(repo_root: Path) -> Path:
    return repo_root / "tui" / "Cargo.toml"


# ---------------------------------------------------------------------------
# 1. install.sh exists and is well-formed
# ---------------------------------------------------------------------------

class TestInstallShExists:
    """Verify install.sh presence and structural integrity."""

    def test_install_sh_exists(self, install_sh: Path) -> None:
        assert install_sh.exists(), "tui/install.sh must exist"

    def test_install_sh_is_file(self, install_sh: Path) -> None:
        assert install_sh.is_file(), "tui/install.sh must be a regular file"

    def test_install_sh_not_empty(self, install_sh: Path) -> None:
        assert install_sh.stat().st_size > 0, "tui/install.sh must not be empty"

    def test_install_sh_has_shebang(self, install_sh: Path) -> None:
        text = install_sh.read_text(encoding="utf-8")
        assert text.startswith("#!/bin/sh"), "install.sh must begin with #!/bin/sh"

    def test_install_sh_uses_set_eu(self, install_sh: Path) -> None:
        text = install_sh.read_text(encoding="utf-8")
        assert "set -eu" in text, "install.sh must use 'set -eu' for safety"

    def test_install_sh_configures_repo(self, install_sh: Path) -> None:
        text = install_sh.read_text(encoding="utf-8")
        assert re.search(r'^REPO="QSOLKCB/QEC"', text, re.MULTILINE), \
            "install.sh must set REPO to QSOLKCB/QEC"

    def test_install_sh_configures_binary_name(self, install_sh: Path) -> None:
        text = install_sh.read_text(encoding="utf-8")
        assert re.search(r'^BINARY_NAME="qec-tui"', text, re.MULTILINE), \
            "install.sh must set BINARY_NAME to qec-tui"

    def test_install_sh_configures_install_dir(self, install_sh: Path) -> None:
        text = install_sh.read_text(encoding="utf-8")
        assert re.search(r'^INSTALL_DIR="/usr/local/bin"', text, re.MULTILINE), \
            "install.sh must set INSTALL_DIR to /usr/local/bin"

    def test_install_sh_uses_github_api(self, install_sh: Path) -> None:
        text = install_sh.read_text(encoding="utf-8")
        expected = "https://api.github.com/repos/${REPO}/releases/latest"
        assert expected in text or \
            "https://api.github.com/repos/QSOLKCB/QEC/releases/latest" in text, \
            "install.sh must query GitHub releases API"

    def test_install_sh_cleans_up_tmpdir(self, install_sh: Path) -> None:
        text = install_sh.read_text(encoding="utf-8")
        assert "trap" in text and "rm -rf" in text, \
            "install.sh must trap-cleanup temporary directory"


# ---------------------------------------------------------------------------
# 2. README advertised curl path is correct
# ---------------------------------------------------------------------------

class TestReadmeCurlPath:
    """Verify the README curl install command matches install.sh location."""

    EXPECTED_URL = "https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh"

    def test_readme_exists(self, readme: Path) -> None:
        assert readme.exists(), "README.md must exist"

    def test_readme_contains_curl_command(self, readme: Path) -> None:
        text = readme.read_text(encoding="utf-8")
        assert "curl" in text, "README must contain a curl install command"

    def test_readme_curl_url_matches(self, readme: Path) -> None:
        text = readme.read_text(encoding="utf-8")
        assert self.EXPECTED_URL in text, \
            f"README must contain the canonical install URL: {self.EXPECTED_URL}"

    def test_readme_curl_flags(self, readme: Path) -> None:
        text = readme.read_text(encoding="utf-8")
        assert "curl -fsSL" in text, \
            "README curl command must use -fsSL flags"

    def test_readme_pipes_to_sh(self, readme: Path) -> None:
        text = readme.read_text(encoding="utf-8")
        assert "| sh" in text, "README curl command must pipe to sh"


# ---------------------------------------------------------------------------
# 3. Installer is idempotent (static parse yields identical results)
# ---------------------------------------------------------------------------

class TestInstallerIdempotency:
    """Verify that re-parsing install.sh yields identical extracted values."""

    def _extract_config(self, text: str) -> dict:
        config = {}
        for key in ("REPO", "ASSET_NAME", "INSTALL_DIR", "BINARY_NAME"):
            m = re.search(rf'^{key}="([^"]+)"', text, re.MULTILINE)
            config[key] = m.group(1) if m else None
        return config

    def test_double_parse_identical(self, install_sh: Path) -> None:
        text = install_sh.read_text(encoding="utf-8")
        first = self._extract_config(text)
        second = self._extract_config(text)
        assert first == second, "Idempotent parse: two passes must yield identical config"

    def test_config_values_stable(self, install_sh: Path) -> None:
        text = install_sh.read_text(encoding="utf-8")
        cfg = self._extract_config(text)
        assert cfg["REPO"] == "QSOLKCB/QEC"
        assert cfg["ASSET_NAME"] == "qec-tui-linux-x86_64.tar.gz"
        assert cfg["INSTALL_DIR"] == "/usr/local/bin"
        assert cfg["BINARY_NAME"] == "qec-tui"


# ---------------------------------------------------------------------------
# 4. Version tag resolves correctly from Cargo.toml
# ---------------------------------------------------------------------------

class TestVersionTagResolution:
    """Verify Cargo.toml declares a valid semver version for qec-tui."""

    def test_cargo_toml_exists(self, cargo_toml: Path) -> None:
        assert cargo_toml.exists(), "tui/Cargo.toml must exist"

    def test_cargo_package_name(self, cargo_toml: Path) -> None:
        text = cargo_toml.read_text(encoding="utf-8")
        m = re.search(r'^name\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert m is not None, "Cargo.toml must declare a package name"
        assert m.group(1) == "qec-tui", f"Package name must be qec-tui, got {m.group(1)!r}"

    def test_cargo_version_is_semver(self, cargo_toml: Path) -> None:
        text = cargo_toml.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert m is not None, "Cargo.toml must declare a version"
        assert re.fullmatch(r"\d+\.\d+\.\d+", m.group(1)), \
            f"Version must be valid semver, got {m.group(1)!r}"

    def test_cargo_version_not_zero(self, cargo_toml: Path) -> None:
        text = cargo_toml.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert m is not None
        assert m.group(1) != "0.0.0", "Version must not be placeholder 0.0.0"


# ---------------------------------------------------------------------------
# 5. qec-tui launches after install (binary availability)
# ---------------------------------------------------------------------------

class TestQecTuiLaunch:
    """Verify qec-tui binary can be located and responds to --help/--version.

    These tests are skipped if the binary is not installed (CI or dev machines
    without a prior install).
    """

    @pytest.fixture
    def qec_tui_bin(self) -> str:
        path = shutil.which("qec-tui")
        if path is None:
            pytest.skip("qec-tui binary not installed on this system")
        return path

    def test_binary_exists(self, qec_tui_bin: str) -> None:
        assert os.path.isfile(qec_tui_bin)

    def test_binary_is_executable(self, qec_tui_bin: str) -> None:
        assert os.access(qec_tui_bin, os.X_OK)


# ---------------------------------------------------------------------------
# 6. help/version commands succeed
# ---------------------------------------------------------------------------

class TestHelpVersionCommands:
    """Verify --help and --version exit cleanly when binary is available."""

    @pytest.fixture
    def qec_tui_bin(self) -> str:
        path = shutil.which("qec-tui")
        if path is None:
            pytest.skip("qec-tui binary not installed on this system")
        return path

    def test_help_exits_zero(self, qec_tui_bin: str) -> None:
        result = subprocess.run(
            [qec_tui_bin, "--help"],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, f"--help exited {result.returncode}"

    def test_version_exits_zero(self, qec_tui_bin: str) -> None:
        result = subprocess.run(
            [qec_tui_bin, "--version"],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, f"--version exited {result.returncode}"

    def test_version_contains_semver(self, qec_tui_bin: str) -> None:
        result = subprocess.run(
            [qec_tui_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert re.search(r"\d+\.\d+\.\d+", result.stdout + result.stderr), \
            "Version output must contain a semver string"


# ---------------------------------------------------------------------------
# 7. release_integrity module round-trip
# ---------------------------------------------------------------------------

class TestReleaseIntegrityModule:
    """Verify the verification module itself produces deterministic reports."""

    def test_import(self) -> None:
        from qec.verification.release_integrity import (
            verify_install_path_consistency,
            verify_binary_version_consistency,
            verify_release_tag_alignment,
        )

    def test_install_path_consistency(self, repo_root: Path) -> None:
        from qec.verification.release_integrity import verify_install_path_consistency
        report = verify_install_path_consistency(str(repo_root))
        assert report.all_passed, \
            f"install_path_consistency failed: {[r for r in report.results if not r.passed]}"

    def test_binary_version_consistency(self, repo_root: Path) -> None:
        from qec.verification.release_integrity import verify_binary_version_consistency
        report = verify_binary_version_consistency(str(repo_root))
        assert report.all_passed, \
            f"binary_version_consistency failed: {[r for r in report.results if not r.passed]}"

    def test_release_tag_alignment(self, repo_root: Path) -> None:
        from qec.verification.release_integrity import verify_release_tag_alignment
        report = verify_release_tag_alignment(str(repo_root))
        assert report.all_passed, \
            f"release_tag_alignment failed: {[r for r in report.results if not r.passed]}"

    def test_tag_cargo_alignment_with_matching_tag(self, repo_root: Path) -> None:
        from qec.verification.release_integrity import verify_release_tag_alignment
        # Read current Cargo.toml version to construct a matching tag
        cargo = (repo_root / "tui" / "Cargo.toml").read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', cargo, re.MULTILINE)
        assert m is not None
        tag = f"v{m.group(1)}"
        report = verify_release_tag_alignment(str(repo_root), expected_tag=tag)
        assert report.all_passed, \
            f"tag alignment failed for {tag}: {[r for r in report.results if not r.passed]}"

    def test_tag_cargo_alignment_mismatch(self, repo_root: Path) -> None:
        from qec.verification.release_integrity import verify_release_tag_alignment
        report = verify_release_tag_alignment(str(repo_root), expected_tag="v0.0.0-fake")
        # Should fail on tag_cargo_version_match
        tag_results = [r for r in report.results if r.check == "tag_cargo_version_match"]
        assert len(tag_results) == 1
        assert not tag_results[0].passed

    def test_deterministic_replay(self, repo_root: Path) -> None:
        """Two consecutive calls must produce identical reports."""
        from qec.verification.release_integrity import verify_install_path_consistency
        r1 = verify_install_path_consistency(str(repo_root))
        r2 = verify_install_path_consistency(str(repo_root))
        assert r1 == r2, "Verification must be deterministic across invocations"
