#!/usr/bin/env python3
"""Validate TUI release artifact synchronization.

Checks that Cargo.toml version, README install text, install.sh target paths,
and release tag alignment are all consistent.

Stdlib only.  Exit 0 on success, exit 1 on any mismatch.

Usage:
    python scripts/validate_tui_release_sync.py [--repo-root DIR] [--expected-tag TAG]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _repo_root_default() -> Path:
    """Walk up from script location to find repo root."""
    cursor = Path(__file__).resolve().parent
    for _ in range(10):
        if (cursor / "tui" / "install.sh").exists():
            return cursor
        cursor = cursor.parent
    print("ERROR: cannot locate repository root", file=sys.stderr)
    sys.exit(1)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=str, default=None,
                        help="Repository root directory")
    parser.add_argument("--expected-tag", type=str, default=None,
                        help="Expected release tag (e.g. v106.0.0)")
    args = parser.parse_args()

    root = Path(args.repo_root) if args.repo_root else _repo_root_default()
    errors: list[str] = []

    # --- Artifact paths ---
    cargo_toml = root / "tui" / "Cargo.toml"
    install_sh = root / "tui" / "install.sh"
    readme = root / "README.md"

    for label, path in [("Cargo.toml", cargo_toml), ("install.sh", install_sh), ("README.md", readme)]:
        if not path.exists():
            errors.append(f"MISSING: {label} at {path}")

    if errors:
        for e in sorted(errors):
            print(f"FAIL  {e}")
        sys.exit(1)

    # --- Parse Cargo.toml ---
    cargo_text = _read(cargo_toml)
    cargo_name_m = re.search(r'^name\s*=\s*"([^"]+)"', cargo_text, re.MULTILINE)
    cargo_ver_m = re.search(r'^version\s*=\s*"([^"]+)"', cargo_text, re.MULTILINE)

    cargo_name = cargo_name_m.group(1) if cargo_name_m else ""
    cargo_ver = cargo_ver_m.group(1) if cargo_ver_m else ""

    if cargo_name != "qec-tui":
        errors.append(f"Cargo.toml name={cargo_name!r}, expected 'qec-tui'")

    if not re.fullmatch(r"\d+\.\d+\.\d+", cargo_ver):
        errors.append(f"Cargo.toml version={cargo_ver!r} is not valid semver")

    # --- Parse install.sh ---
    sh_text = _read(install_sh)

    sh_checks = {
        "REPO": ("QSOLKCB/QEC", re.search(r'^REPO="([^"]+)"', sh_text, re.MULTILINE)),
        "BINARY_NAME": ("qec-tui", re.search(r'^BINARY_NAME="([^"]+)"', sh_text, re.MULTILINE)),
        "INSTALL_DIR": ("/usr/local/bin", re.search(r'^INSTALL_DIR="([^"]+)"', sh_text, re.MULTILINE)),
        "ASSET_NAME": ("qec-tui-linux-x86_64.tar.gz", re.search(r'^ASSET_NAME="([^"]+)"', sh_text, re.MULTILINE)),
    }
    for key, (expected, match) in sorted(sh_checks.items()):
        actual = match.group(1) if match else "<not found>"
        if actual != expected:
            errors.append(f"install.sh {key}={actual!r}, expected {expected!r}")

    # --- Parse README ---
    readme_text = _read(readme)
    expected_url = "https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh"
    if expected_url not in readme_text:
        errors.append(f"README.md missing canonical curl URL: {expected_url}")

    if "curl -fsSL" not in readme_text:
        errors.append("README.md missing 'curl -fsSL' install command")

    # --- Release tag alignment (optional) ---
    if args.expected_tag:
        tag_ver = args.expected_tag.lstrip("v")
        if cargo_ver and cargo_ver != tag_ver:
            errors.append(
                f"Tag version mismatch: --expected-tag={args.expected_tag!r} "
                f"({tag_ver!r}) != Cargo.toml version ({cargo_ver!r})"
            )

    # --- Report ---
    if errors:
        print(f"\n{'='*60}")
        print(f"TUI RELEASE SYNC VALIDATION: {len(errors)} error(s)")
        print(f"{'='*60}")
        for e in sorted(errors):
            print(f"  FAIL  {e}")
        print()
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("TUI RELEASE SYNC VALIDATION: ALL CHECKS PASSED")
        print(f"{'='*60}")
        print(f"  Cargo.toml  name={cargo_name!r}  version={cargo_ver!r}")
        print(f"  install.sh  REPO=QSOLKCB/QEC  BINARY=qec-tui  DIR=/usr/local/bin")
        print(f"  README.md   curl URL present")
        if args.expected_tag:
            print(f"  Tag         {args.expected_tag!r} aligned")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
