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
    "## v163.x → v164.x Capability Summary",
}
IMMUTABLE_HEADERS = {
    "## 📚 DOIs",
    "## ⚡ Quickstart",
    "## Testing",
    "## Commands",
    "## IRC Operator Surface",
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
    return out, n > 0


def update_readme(text: str, latest_release: str, frontier: str, completed_arc: str) -> str:
    orig = text
    mutable_sections_changed = 0

    new = text
    new, changed = _replace_required(new, r"stable-v[\d.]+-success", f"stable-{latest_release}-success")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"branch-v[\d.]+%20canonical", f"branch-{latest_release}%20canonical")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"Current release line: \*\*[^*]+\*\*", f"Current release line: **{latest_release}**")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"Current frontier: \*\*[^*]+\*\*", f"Current frontier: **{frontier}**")
    mutable_sections_changed += int(changed)
    active_arc = re.sub(r"\.\d+$", ".x", latest_release)
    new, changed = _replace_required(new, r"Active arc: \*\*[^*]+\*\*", f"Active arc: **{active_arc} — Invariant-Based Heavy Dependency Optimization**")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"Completed arc: \*\*[^*]+\*\*", f"Completed arc: **{completed_arc}**")
    mutable_sections_changed += int(changed)
    new, changed = _replace_required(new, r"status is current through \*\*[^*]+\*\*", f"status is current through **{latest_release}**")
    mutable_sections_changed += int(changed)

    _validate_boundaries(orig, new)
    if mutable_sections_changed == 0:
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
        if h in IMMUTABLE_HEADERS and b.get(h, "") != a.get(h, ""):
            raise ValueError(ERR)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--latest-release", required=True)
    p.add_argument("--frontier", required=True)
    p.add_argument("--completed-arc", required=True)
    args = p.parse_args()
    text = args.readme.read_text(encoding="utf-8")
    updated = update_readme(text, args.latest_release, args.frontier, args.completed_arc)
    args.readme.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
