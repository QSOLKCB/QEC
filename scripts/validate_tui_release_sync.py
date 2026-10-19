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

# Reuse the canonical repo-root resolver and installer constants so that
# the script and the library module cannot drift out of sync.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qec.verification.release_integrity import (  # noqa: E402
    ASSET_NAME,
    BINARY_NAME,
    CANONICAL_CURL_URL,
    INSTALL_DIR,
    REPO_SLUG,
    resolve_repo_root,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


LOCAL_INSTALL_PROMPT = (
    "For the optional Rust TUI launcher, review the [installer](tui/install.sh) before running:"
)
LOCAL_INSTALL_BLOCK = LOCAL_INSTALL_PROMPT + "\n\n```sh\nsh tui/install.sh\n```"
COMPACT_MARKER = "<!-- README-RELEASE:COMPACT -->"


def _validate_installation_guidance(text: str) -> list[str]:
    """Check the exact supported installation block, not arbitrary Markdown/shell.

    Compact documents require the literal prose prompt and shell block above,
    followed only by whitespace until the next heading. Comments become line
    breaks so they cannot join tokens. Raw HTML and multiline inline-code spans
    are unsupported. Legacy mode requires its positive, visible release heading.
    """
    source = re.sub(r"<!--.*?(?:-->|\Z)", "\n", text.replace("\r\n", "\n"), flags=re.DOTALL)
    lines = source.splitlines()
    headings, prompts = [], []
    fence = None
    unsupported_inline = False
    for index, line in enumerate(lines):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            run, suffix = marker.groups()
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not suffix.strip():
                fence = None
            continue
        if fence is not None or line.startswith(("    ", "\t")):
            continue
        if re.match(r"^#{1,6} ", line):
            headings.append(line)
        if line == LOCAL_INSTALL_PROMPT:
            prompts.append(index)
        if "``" in line or line.count("`") % 2:
            unsupported_inline = True

    # Unsupported rendering forms reject before either format can be selected.
    if "<" in source or unsupported_inline or fence is not None:
        return ["README.md installation guidance uses unsupported rendering syntax"]
    compact = COMPACT_MARKER in text or "## Release" in headings
    legacy = "## 📦 Release & Research" in headings
    invalid = ["README.md installation guidance does not match its supported literal format"]
    if compact:
        if text.count(COMPACT_MARKER) != 1 or headings.count("## Release") != 1 or legacy:
            return invalid
        if len(prompts) != 1 or source.count("tui/install.sh") != 2:
            return invalid
        if re.search(r"\b(?:curl|wget)[ \t]", source):
            return invalid
        start = prompts[0]
        end = next((i for i in range(start + 1, len(lines)) if re.match(r"^#{1,6} ", lines[i])), len(lines))
        if "\n".join(lines[start:end]).strip() != LOCAL_INSTALL_BLOCK:
            return invalid
        return []
    if not legacy or headings.count("## 📦 Release & Research") != 1:
        return invalid
    # Supported legacy command lines are exact bare commands or exact Markdown
    # table cells. Shell suffixes cannot be accepted as part of a substring.
    command = f"curl -fsSL {CANONICAL_CURL_URL} | sh"
    table_command = command.replace("|", "\\|")
    matches = [line for line in lines if line == command or re.fullmatch(
        rf"\| `{re.escape(table_command)}` \| [^|]+ \|", line
    )]
    if len(matches) != 1 or len(re.findall(r"\b(?:curl|wget)[ \t]", source)) != 1:
        return invalid

    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=str, default=None,
                        help="Repository root directory")
    parser.add_argument("--expected-tag", type=str, default=None,
                        help="Expected release tag (e.g. v106.0.0)")
    args = parser.parse_args()

    root = resolve_repo_root(args.repo_root)
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

    if cargo_name != BINARY_NAME:
        errors.append(f"Cargo.toml name={cargo_name!r}, expected {BINARY_NAME!r}")

    if not re.fullmatch(r"\d+\.\d+\.\d+", cargo_ver):
        errors.append(f"Cargo.toml version={cargo_ver!r} is not valid semver")

    # --- Parse install.sh ---
    sh_text = _read(install_sh)

    sh_checks = {
        "REPO": (REPO_SLUG, re.search(r'^REPO="([^"]+)"', sh_text, re.MULTILINE)),
        "BINARY_NAME": (BINARY_NAME, re.search(r'^BINARY_NAME="([^"]+)"', sh_text, re.MULTILINE)),
        "INSTALL_DIR": (INSTALL_DIR, re.search(r'^INSTALL_DIR="([^"]+)"', sh_text, re.MULTILINE)),
        "ASSET_NAME": (ASSET_NAME, re.search(r'^ASSET_NAME="([^"]+)"', sh_text, re.MULTILINE)),
    }
    for key, (expected, match) in sorted(sh_checks.items()):
        actual = match.group(1) if match else "<not found>"
        if actual != expected:
            errors.append(f"install.sh {key}={actual!r}, expected {expected!r}")

    # --- Parse README using one normalized documentation contract ---
    errors.extend(_validate_installation_guidance(_read(readme)))

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
        print(f"  README.md   installer command present")
        if args.expected_tag:
            print(f"  Tag         {args.expected_tag!r} aligned")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
