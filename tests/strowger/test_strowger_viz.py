# SPDX-License-Identifier: MPL-2.0
"""Offline Strowger lab packaging and boundary checks."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "viz" / "strowger"


def test_offline_lab_assets_exist() -> None:
    assert (LAB / "index.html").is_file()
    assert (LAB / "style.css").is_file()
    assert (LAB / "app.js").is_file()


def test_lab_has_operator_desk_and_no_network_runtime() -> None:
    html = (LAB / "index.html").read_text(encoding="utf-8")
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert "Operator Desk" in html
    assert "Manual laboratory" in html
    assert "operator_may_force_accept: false" in script
    assert "fetch(" not in script
    assert "XMLHttpRequest" not in script
    assert "WebSocket" not in script


def test_lab_supports_audio_and_wav_without_dependencies() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert "AudioContext" in script
    assert "audio/wav" in script
    assert "node_modules" not in script
