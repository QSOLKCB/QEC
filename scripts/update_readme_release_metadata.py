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
    "## Proof Artifacts",
    "# 🧠 Core Law",
}
IMMUTABLE_HEADERS = {
    "## 📚 DOIs",
    "## ⚡ Quickstart",
    "## Commands",
    "## IRC Operator Surface",
    "## 🧾 Attribution",
    "## References",
    "## Author",
}
OSF_LINK = "https://osf.io/sjk7b"
ERR = "README_BOUNDARY_VIOLATION"

# Regex pattern to match stable badge URL with any version
STABLE_BADGE_RE = re.compile(r"https://img\.shields\.io/badge/stable-v[\d.]+-success")
# Regex pattern to match stable badge link with any version
STABLE_BADGE_LINK_RE = re.compile(
    r'\[!\[Latest\]\(https://img\.shields\.io/badge/stable-v[\d.]+-success\)\]'
    r'\(https://github\.com/QSOLKCB/QEC/releases/tag/v[\d.]+\)'
)


def _split_sections(text: str):
    parts = re.split(r"(?m)^(#\#? .+)$", text)
    if not parts:
        return []
    sections = []
    pre = parts[0]
    if pre:
        sections.append(("", pre))
    for i in range(1, len(parts), 2):
        sections.append((parts[i], parts[i + 1]))
    return sections


def update_readme(text: str, latest_release: str, frontier: str, completed_arc: str) -> str:
    orig = text
    # Construct badge URL once for consistency
    badge_url = f"https://img.shields.io/badge/stable-{latest_release}-success"
    release_link = f"https://github.com/QSOLKCB/QEC/releases/tag/{latest_release}"
    
    # Update stable badge URL (version-agnostic match)
    text = STABLE_BADGE_RE.sub(badge_url, text)
    # Update stable badge link target alongside the shield URL
    text = STABLE_BADGE_LINK_RE.sub(
        f"[![Latest]({badge_url})]({release_link})",
        text,
    )
    text = re.sub(r"Current release line: \*\*[^*]+\*\*", f"Current release line: **{latest_release}**", text)
    text = re.sub(r"Current frontier: \*\*[^*]+\*\*", f"Current frontier: **{frontier}**", text)
    text = re.sub(r"Completed arc: \*\*[^*]+\*\*", f"Completed arc: **{completed_arc}**", text)
    text = re.sub(r"status is current through \*\*[^*]+\*\*", f"status is current through **{latest_release}**", text)

    if OSF_LINK in text:
        text = text.replace(f"[![OSF Registration](https://img.shields.io/badge/OSF-Registration-blue)]({OSF_LINK})\n", "")
        doi_header = "## 📚 DOIs\n"
        if doi_header in text:
            post = text.split(doi_header, 1)[1].split("\n## ", 1)[0]
            if OSF_LINK not in post:
                text = text.replace(doi_header, doi_header + f"[OSF Registration]({OSF_LINK})\n")

    _validate_boundaries(orig, text)
    return text


def _validate_boundaries(before: str, after: str) -> None:
    """Validate that only mutable sections changed, fail closed on unknown sections."""
    b = dict(_split_sections(before))
    a = dict(_split_sections(after))
    # Collect all headers from both before and after
    all_headers = set(b.keys()) | set(a.keys())
    for h in all_headers:
        if not h:  # Skip preamble
            continue
        # DOIs section is allowed to change for OSF migration
        if h == "## 📚 DOIs":
            continue
        # Mutable sections are allowed to change
        if h in MUTABLE_HEADERS:
            continue
        # All other sections (including IMMUTABLE_HEADERS and unknown sections) must not be changed
        if b.get(h, "") != a.get(h, ""):
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
