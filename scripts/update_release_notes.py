#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HEADER = "# RELEASE NOTES\n\n"
MAX_ENTRY_CHARS = 500

TAG_RE = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?$")


@dataclass(frozen=True, order=True)
class ReleaseTag:
    major: int
    minor: int
    patch: int
    raw: str

    @property
    def normalized(self) -> str:
        return f"v{self.major}.{self.minor}" + (f".{self.patch}" if self.patch else "")


def _run_git_tags(repo_root: Path) -> list[str]:
    out = subprocess.check_output(["git", "tag"], cwd=repo_root, text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def parse_release_tags(tags: Iterable[str]) -> list[ReleaseTag]:
    parsed: list[ReleaseTag] = []
    for tag in tags:
        m = TAG_RE.match(tag)
        if not m:
            continue
        major, minor, patch = m.groups()
        parsed.append(ReleaseTag(int(major), int(minor), int(patch or 0), tag))
    parsed.sort(reverse=True)
    return parsed


def make_summary(tag: ReleaseTag) -> str:
    summary = (
        f"Deterministic release documentation checkpoint for {tag.normalized}. "
        "Canonical tag registration preserved for replay-safe historical indexing."
    )
    return summary[:MAX_ENTRY_CHARS]


def generate_release_notes(tags: list[ReleaseTag]) -> str:
    lines = [HEADER.rstrip(), ""]
    seen: set[str] = set()
    for tag in tags:
        name = tag.normalized
        if name in seen:
            continue
        seen.add(name)
        lines.append(f"## {name}")
        lines.append(make_summary(tag))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, default=Path("RELEASE_NOTES.md"))
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    output = (repo_root / args.output).resolve()
    tags = parse_release_tags(_run_git_tags(repo_root))
    
    # Fail-fast: if no release tags are discovered, preserve existing release notes
    if not tags:
        if output.exists():
            # Preserve existing release notes when no tags found (e.g., shallow CI checkout)
            return 0
        # No tags and no existing file - write header only
        output.write_text(HEADER, encoding="utf-8")
        return 0
    
    content = generate_release_notes(tags)
    output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
