# SPDX-License-Identifier: MPL-2.0
"""Offline lab structure and optional dependency-free Node behavioral checks.

The Node DOM/audio stub does not establish actual browser layout or acoustic QA.
It exercises the real embedded JavaScript, including deterministic demo exports.
"""
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "docs" / "labs" / "digital" / "index.html"


class _LabParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.labels = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        if tag == "label" and "for" in values:
            self.labels.append(values["for"])
        if tag == "a" and "href" in values:
            self.links.append(values["href"])


def test_lab_structure_and_local_links():
    parser = _LabParser()
    parser.feed(LAB.read_text(encoding="utf-8"))
    assert len(parser.ids) == len(set(parser.ids))
    assert set(parser.labels) <= set(parser.ids)
    assert parser.links
    for link in parser.links:
        assert (LAB.parent / link).is_file(), link


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_lab_deterministic_demo_and_interactions():
    result = subprocess.run(
        [shutil.which("node"), str(Path(__file__).with_name("test_lab_demo.cjs"))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "7840 deterministic scenario pairs" in result.stdout
    assert "click-only audio scheduling" in result.stdout
