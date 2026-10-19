"""Validate local and legacy installer documentation without executing installers."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("command,success", [
    ("sh tui/install.sh", True),
    ("curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh", True),
    ("sh missing/install.sh", False),
    ("No installer", False),
])
def test_installer_docs_validate_exact_paths(tmp_path, command, success):
    (tmp_path / "tui").mkdir()
    for name in ("install.sh", "Cargo.toml"):
        shutil.copyfile(ROOT / "tui" / name, tmp_path / "tui" / name)
    (tmp_path / "README.md").write_text(command + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_tui_release_sync.py"),
         "--repo-root", str(tmp_path)], capture_output=True, text=True,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
