# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "viz" / "panel"

def test_panel_lab_is_offline_and_explicitly_noncanonical():
    html = (LAB / "index.html").read_text(encoding="utf-8")
    js = (LAB / "app.js").read_text(encoding="utf-8")
    assert "Demonstration evidence only" in html
    assert "qec.panel-browser-demonstration.v1" in js
    assert "browser_demo_only:true" in js
    assert "canonical_receipt:false" in js
    for marker in ("fetch(", "XMLHttpRequest", "WebSocket", "https://", "http://"):
        assert marker not in html + js

def test_panel_lab_visualises_and_sonifies_control_phases():
    js = (LAB / "app.js").read_text(encoding="utf-8")
    for marker in ("digit_register_sealed", "sender_program_sealed", "selector_move", "independent_route_verification", "AudioContext", "createOscillator"):
        assert marker in js
