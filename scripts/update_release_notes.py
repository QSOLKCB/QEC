#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

HEADER = "# RELEASE NOTES\n\n"
MAX_ENTRY_CHARS = 500
GENERIC_SUMMARY = (
    "Historical QEC release preserved from canonical release-history manifest. "
    "Detailed metadata unavailable in the release archive."
)
SEMVER_RE = re.compile(r"^(v?)(\d+)\.(\d+)(?:\.(\d+))?$")
MIN_CANONICAL_RELEASE_COUNT = 890


class ReleaseHistoryError(ValueError):
    pass


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
    return sorted(set(t.strip() for t in tags if t and t.strip()), key=_version_key, reverse=True)


def load_release_history_tags(repo_root: Path) -> list[str]:
    p = repo_root / "release_history.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        tags = [str(item["tag"]).strip() for item in data if isinstance(item, dict) and item.get("tag")]
        if len(tags) != len(set(tags)):
            raise ReleaseHistoryError("DUPLICATE_RELEASE_ENTRY")
        if not tags:
            raise ReleaseHistoryError("NO_RELEASE_TAGS_FOUND")
        return discover_release_tags(tags)
    return []


def _validate_release_history(discovered_release_tags: list[str], generated_release_tags: list[str]) -> None:
    if set(discovered_release_tags) != set(generated_release_tags):
        raise ReleaseHistoryError("RELEASE_HISTORY_MISMATCH")
    if len(generated_release_tags) != len(set(generated_release_tags)):
        raise ReleaseHistoryError("DUPLICATE_RELEASE_ENTRY")


def _summary_for(entry: dict[str, object]) -> str:
    title = str(entry.get("title") or "").strip()
    body = str(entry.get("body") or "").strip()
    merged = " ".join(x for x in [title, body] if x).strip()
    return (merged or GENERIC_SUMMARY)[:MAX_ENTRY_CHARS]


def generate_release_notes_from_history(history: list[dict[str, object]]) -> str:
    tags = discover_release_tags([str(e.get("tag", "")) for e in history])
    entry_map = {str(e.get("tag")): e for e in history}
    lines = [HEADER.rstrip(), ""]
    for tag in tags:
        lines.append(f"## {tag}")
        lines.append(_summary_for(entry_map.get(tag, {})))
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    generated_release_tags = [line[3:] for line in content.splitlines() if line.startswith("## ")]
    _validate_release_history(tags, generated_release_tags)
    return content


def build_release_notes_from_tags(tags: Iterable[str], min_release_count: int = MIN_CANONICAL_RELEASE_COUNT) -> str:
    if min_release_count < MIN_CANONICAL_RELEASE_COUNT:
        raise ReleaseHistoryError("INVALID_MIN_RELEASE_COUNT")
    discovered = discover_release_tags(tags)
    if not discovered:
        raise ReleaseHistoryError("NO_RELEASE_TAGS_FOUND")
    if len(discovered) < min_release_count:
        raise ReleaseHistoryError("RELEASE_HISTORY_INCOMPLETE")
    history = [{"tag": t, "title": "", "body": ""} for t in discovered]
    return generate_release_notes_from_history(history)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, default=Path("RELEASE_NOTES.md"))
    p.add_argument("--min-release-count", type=int, default=MIN_CANONICAL_RELEASE_COUNT)
    args = p.parse_args()
    repo_root = args.repo_root.resolve()
    output = (repo_root / args.output).resolve()

    if args.min_release_count < MIN_CANONICAL_RELEASE_COUNT:
        raise ReleaseHistoryError("INVALID_MIN_RELEASE_COUNT")

    tags = load_release_history_tags(repo_root)
    if not tags:
        tags = discover_release_tags(_run_git_tags(repo_root))
    if not tags:
        raise ReleaseHistoryError("NO_RELEASE_TAGS_FOUND")
    if len(tags) < args.min_release_count:
        raise ReleaseHistoryError("RELEASE_HISTORY_INCOMPLETE")

    history_path = repo_root / "release_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = [{"tag": t, "title": "", "body": ""} for t in tags]

    content = generate_release_notes_from_history(history)
    output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
