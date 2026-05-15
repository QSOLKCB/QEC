#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Iterable

HEADER = "# RELEASE NOTES\n\n"
MAX_ENTRY_CHARS = 500
GENERIC_SUMMARY = (
    "Historical QEC release preserved from published tag history. "
    "Detailed metadata unavailable in local release manifest."
)
SEMVER_RE = re.compile(r"^(v?)(\d+)\.(\d+)(?:\.(\d+))?$")


class ReleaseHistoryError(ValueError):
    """Raised when release history constraints are violated."""


def _run_git_tags(repo_root: Path) -> list[str]:
    out = subprocess.check_output(["git", "tag", "--list"], cwd=repo_root, text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _version_key(tag: str) -> tuple[int, int, int, int, str]:
    m = SEMVER_RE.match(tag)
    if m:
        _prefix, major, minor, patch = m.groups()
        return (1, int(major), int(minor), int(patch or 0), tag)
    return (0, -1, -1, -1, tag)


def discover_release_tags(tags: Iterable[str]) -> list[str]:
    unique = sorted(set(t.strip() for t in tags if t and t.strip()), key=_version_key, reverse=True)
    return unique


def _validate_release_history(discovered_release_tags: list[str], generated_release_tags: list[str]) -> None:
    expected = set(discovered_release_tags)
    actual = set(generated_release_tags)

    if expected != actual:
        missing = expected - actual
        unexpected = actual - expected
        raise ReleaseHistoryError(
            f"RELEASE_HISTORY_MISMATCH missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    if len(generated_release_tags) != len(set(generated_release_tags)):
        raise ReleaseHistoryError("DUPLICATE_RELEASE_ENTRY")


def _summary_for(_tag: str) -> str:
    return GENERIC_SUMMARY[:MAX_ENTRY_CHARS]


def generate_release_notes(discovered_release_tags: list[str]) -> str:
    lines = [HEADER.rstrip(), ""]
    for tag in discovered_release_tags:
        lines.append(f"## {tag}")
        lines.append(_summary_for(tag))
        lines.append("")

    content = "\n".join(lines).rstrip() + "\n"

    generated_release_tags = [line[3:] for line in content.splitlines() if line.startswith("## ")]
    _validate_release_history(discovered_release_tags, generated_release_tags)
    return content


def build_release_notes_from_tags(tags: Iterable[str], min_release_count: int = 800) -> str:
    if min_release_count < 1:
        raise ReleaseHistoryError("INVALID_MIN_RELEASE_COUNT")
    discovered = discover_release_tags(tags)
    if not discovered:
        raise ReleaseHistoryError("NO_RELEASE_TAGS_FOUND")
    if len(discovered) < min_release_count:
        raise ReleaseHistoryError("RELEASE_HISTORY_INCOMPLETE")
    return generate_release_notes(discovered)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, default=Path("RELEASE_NOTES.md"))
    p.add_argument("--min-release-count", type=int, default=800)
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    output = (repo_root / args.output).resolve()
    tags = _run_git_tags(repo_root)
    content = build_release_notes_from_tags(tags, min_release_count=args.min_release_count)
    output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
