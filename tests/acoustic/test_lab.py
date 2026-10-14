# SPDX-License-Identifier: MPL-2.0
"""Offline inspector structure plus dependency-free DOM/audio-stub behavior.

These checks are not rendered-browser layout, audible quality, physical-acoustic
QA, or canonical evidence acceptance. Native validators establish replay.
"""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "docs/labs/acoustic-evidence.html"
SCRIPT = LAB.with_suffix(".js")


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.labels = []
        self.local_assets = []
        self.policy = ""

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        if tag == "label":
            self.labels.append(values["for"])
        if tag == "a":
            self.local_assets.append(values["href"])
        if tag == "script":
            self.local_assets.append(values["src"])
        if tag == "meta" and values.get("http-equiv") == "Content-Security-Policy":
            self.policy = values["content"]


def test_acoustic_lab_local_structure_and_claim_boundary():
    parser = _Parser()
    html = LAB.read_text(encoding="utf-8")
    parser.feed(html)
    assert len(parser.ids) == len(set(parser.ids))
    assert set(parser.labels) <= set(parser.ids)
    assert parser.local_assets
    for asset in parser.local_assets:
        assert "://" not in asset
        assert (LAB.parent / asset).is_file(), asset
    assert "connect-src 'none'" in parser.policy
    assert "default-src 'none'" in parser.policy
    assert "Canonical evidence requires Python independent replay" in html
    assert "No physical quantum protection or SPECTRAL repository/UI integration" in html
    assert "No synthetic replacement" in html
    assert "innerHTML" not in SCRIPT.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_acoustic_lab_native_artifacts_and_interactions(tmp_path):
    from qec.experiments.acoustic.channel import PCM, wav_bytes
    from qec.experiments.acoustic.evidence import (
        build_battery_source,
        build_demonstration,
        build_trial_source,
        run_comparative_battery,
        run_comparative_trial,
    )

    source = build_trial_source("broadband")
    demo = build_demonstration(source)
    trial = run_comparative_trial(build_trial_source("coherent_substitution"))
    battery_source = build_battery_source(cases=("clean", "missing_samples"), alphabets=(4,))
    fixtures = {
        "source": source,
        "demo": demo,
        "trial": trial,
        "battery_source": battery_source,
        "battery": run_comparative_battery(battery_source),
        "wav_hex": {role: wav_bytes(PCM.from_dict(demo["roles"][role])).hex()
                    for role in ("original", "observed", "filtered")},
    }
    path = tmp_path / "lab-inputs.json"
    path.write_text(json.dumps(fixtures), encoding="utf-8")
    result = subprocess.run(
        [shutil.which("node"), str(Path(__file__).with_suffix(".cjs")), str(path)],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "actual PCM/WAV, imported claim separation" in result.stdout
    assert "stale loads, reset and click-only playback" in result.stdout
