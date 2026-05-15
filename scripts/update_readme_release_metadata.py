#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

MUTABLE_HEADERS = {
    "# QSOLKCB / QEC",
    "## 📦 Release & Research",
    "# ✅ System Properties",
    "# 🧠 What QEC Is",
    "## Why This Matters",
    "## Capability Summary",
}
IMMUTABLE_HEADERS = {
    "## 📚 DOIs",
    "## ⚡ Quickstart",
    "### Testing",
    "## Commands",
    "## IRC Operator Surface",
    "## Proof Artifacts",
    "# 🧠 Core Law",
    "## 🧾 Attribution",
    "## References",
    "## Author",
}
ERR = "README_BOUNDARY_VIOLATION"


def _split_sections(text: str):
    parts = re.split(r"(?m)^(#\#? .+)$", text)
    sections = []
    pre = parts[0] if parts else ""
    if pre:
        sections.append(("", pre))
    for i in range(1, len(parts), 2):
        sections.append((parts[i], parts[i + 1]))
    return sections


def _replace_required(text: str, pattern: str, repl: str) -> tuple[str, bool]:
    out, n = re.subn(pattern, repl, text)
    # Compare actual text to detect real changes (pattern may match but replacement may be identical)
    return out, out != text


def update_readme(
    text: str,
    latest_release: str,
    frontier: str,
    completed_arc: str,
    active_arc: str | None = None,
    repository_status: str | None = None,
) -> str:
    orig = text
    mutable_sections_changed = 0

    if active_arc is None:
        active_arc = re.sub(r"\.\d+$", ".x", latest_release)
    if repository_status is None:
        repository_status = latest_release

    new = text
    new, changed = _replace_required(new, r"stable-v[\d.]+-success", f"stable-{latest_release}-success")
    mutable_sections_changed += int(changed)
    # Update badge link target URL in addition to shield text
    new, changed = _replace_required(new, r"releases/tag/v[\d.]+", f"releases/tag/{latest_release}")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"branch-v[\d.]+%20canonical", f"branch-{latest_release}%20canonical")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"Current release line: \*\*[^*]+\*\*", f"Current release line: **{latest_release}**")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"Current frontier: \*\*[^*]+\*\*", f"Current frontier: **{frontier}**")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"Active arc: \*\*[^*]+\*\*", f"Active arc: **{active_arc}**")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"Completed arc: \*\*[^*]+\*\*", f"Completed arc: **{completed_arc}**")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"status is current through \*\*[^*]+\*\*", f"status is current through **{repository_status}**")
    mutable_sections_changed += int(changed)

    _validate_boundaries(orig, new)
    if mutable_sections_changed == 0:
        expected_tokens = (
            f"stable-{latest_release}-success",
            f"releases/tag/{latest_release}",
            f"branch-{latest_release}%20canonical-purple",
            f"Current release line: **{latest_release}**",
            f"Current frontier: **{frontier}**",
            f"Active arc: **{active_arc}**",
            f"Completed arc: **{completed_arc}**",
            f"status is current through **{repository_status}**",
        )
        if all(token in new for token in expected_tokens):
            return new
        raise ValueError("README_UPDATE_NO_EFFECT")
    return new


def _validate_boundaries(before: str, after: str) -> None:
    b = dict(_split_sections(before))
    a = dict(_split_sections(after))
    for h in set(b) | set(a):
        if not h:
            continue
        if h in MUTABLE_HEADERS:
            continue
        # Fail-closed: raise for ANY section outside MUTABLE_HEADERS that changed
        if b.get(h, "") != a.get(h, ""):
            raise ValueError(ERR)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--latest-release", required=True)
    p.add_argument("--frontier", required=True)
    p.add_argument("--active-arc", required=True)
    p.add_argument("--completed-arc", required=True)
    p.add_argument("--repository-status", required=True)
    args = p.parse_args()
    text = args.readme.read_text(encoding="utf-8")
    updated = update_readme(
        text,
        args.latest_release,
        args.frontier,
        args.completed_arc,
        args.active_arc,
        args.repository_status,
    )
    args.readme.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
