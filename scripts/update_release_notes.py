#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

HEADER = "# RELEASE NOTES\n\n"
MAX_ENTRY_CHARS = 500
GENERIC_SUMMARY = (
    "Historical QEC release preserved from canonical release-history manifest. "
    "Detailed metadata unavailable in the canonical archive."
)


class ReleaseHistoryError(ValueError):
    pass


def _extract_manifest_tags(history: list[dict[str, object]]) -> list[str]:
    tags: list[str] = []
    for i, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise ReleaseHistoryError(f"INVALID_RELEASE_HISTORY_ENTRY at index {i}")
        raw_tag = entry.get("tag")
        if raw_tag is None:
            raise ReleaseHistoryError(f"MALFORMED_RELEASE_ENTRY missing_tag at index {i}")
        tag = str(raw_tag).strip()
        if not tag:
            raise ReleaseHistoryError(f"MALFORMED_RELEASE_ENTRY empty_tag at index {i}")
        tags.append(tag)
    if len(tags) != len(set(tags)):
        raise ReleaseHistoryError("DUPLICATE_RELEASE_ENTRY")
    if not tags:
        raise ReleaseHistoryError("NO_RELEASE_TAGS_FOUND")
    return tags


def load_release_history(repo_root: Path) -> list[dict[str, object]]:
    path = repo_root / "release_history.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ReleaseHistoryError(f"INVALID_RELEASE_HISTORY_JSON: {e}") from e
    if not isinstance(data, list):
        raise ReleaseHistoryError("INVALID_RELEASE_HISTORY_SCHEMA: expected list")
    _extract_manifest_tags(data)
    return data


def _validate_release_history(manifest_tags: list[str], generated_release_tags: list[str]) -> None:
    expected = set(manifest_tags)
    actual = set(generated_release_tags)
    if expected != actual:
        missing = expected - actual
        unexpected = actual - expected
        raise ReleaseHistoryError(
            f"RELEASE_HISTORY_MISMATCH missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    if len(generated_release_tags) != len(set(generated_release_tags)):
        raise ReleaseHistoryError("DUPLICATE_RELEASE_ENTRY")


def _summary_for(entry: dict[str, object]) -> str:
    title = str(entry.get("title") or "").strip()
    body = str(entry.get("body") or "").strip()
    merged = " ".join(x for x in [title, body] if x).strip()
    return (merged or GENERIC_SUMMARY)[:MAX_ENTRY_CHARS]


def generate_release_notes_from_history(history: list[dict[str, object]]) -> str:
    manifest_tags = _extract_manifest_tags(history)
    lines = [HEADER.rstrip(), ""]
    for entry in history:
        tag = str(entry["tag"]).strip()
        lines.append(f"## {tag}")
        lines.append(_summary_for(entry))
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    generated_tags = [line[3:] for line in content.splitlines() if line.startswith("## ")]
    _validate_release_history(manifest_tags, generated_tags)
    return content


def build_release_notes_from_tags(tags: Iterable[str], min_release_count: int = 1) -> str:
    discovered = [t.strip() for t in tags if t and t.strip()]
    if not discovered:
        raise ReleaseHistoryError("NO_RELEASE_TAGS_FOUND")
    if len(set(discovered)) != len(discovered):
        raise ReleaseHistoryError("DUPLICATE_RELEASE_ENTRY")
    if len(discovered) < min_release_count:
        raise ReleaseHistoryError("RELEASE_HISTORY_INCOMPLETE")
    history = [{"tag": t, "title": "", "body": ""} for t in discovered]
    return generate_release_notes_from_history(history)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, default=Path("RELEASE_NOTES.md"))
    args = p.parse_args()
    repo_root = args.repo_root.resolve()
    output = (repo_root / args.output).resolve()
    history = load_release_history(repo_root)
    output.write_text(generate_release_notes_from_history(history), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
