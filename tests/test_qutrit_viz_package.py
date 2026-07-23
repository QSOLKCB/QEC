"""Offline packaging checks for the qutrit visual lab."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "viz"


def test_visual_lab_is_dependency_free_and_offline():
    html = (VIZ / "index.html").read_text(encoding="utf-8")
    assert 'href="style.css"' in html
    assert 'src="qutrit-core.js"' in html
    assert 'src="qutrit-draw.js"' in html
    assert 'src="qutrit-viz.js"' in html

    for path in VIZ.iterdir():
        if path.suffix not in {".html", ".css", ".js"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "https://" not in text
        assert "http://" not in text


def test_visual_lab_keeps_the_claim_boundary_visible():
    html = (VIZ / "index.html").read_text(encoding="utf-8")
    assert "Sound is a redundant classical syndrome receiver" in html
    assert "Quantum protection comes from the GF(3) stabilizer code" in html
