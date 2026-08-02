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


def test_browser_download_is_explicitly_noncanonical() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert 'schema: "qec.strowger-browser-demonstration.v1"' in script
    assert "browser_demo_only: true" in script
    assert "canonical_receipt: false" in script
    assert 'digest_algorithm: "fnv1a-derived-demo-digest-v1"' in script
    assert "pseudoSha" not in script
    assert "event_sha256" not in script
    assert "previous_event_sha256" not in script


def test_event_log_uses_text_nodes_for_untrusted_details() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert 'const summary = document.createElement("b")' in script
    assert "summary.textContent =" in script
    assert "document.createTextNode(" in script
    assert "item.innerHTML" not in script


def test_operator_target_matches_quarantined_selector() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert 'quarantine: {action: "quarantine_trunk", target: "selector-0:0"}' in script
    assert 'target: "selector-1:0"' not in script


def test_browser_validates_radix_domain_before_routing() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert "function validRadices(radices)" in script
    assert "Number.isInteger(radix)" in script
    assert "radix >= 2" in script
    assert "!validRadices(radices)" in script


def test_operator_actions_change_demo_state() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert "operatorState.quarantined.add(command.target)" in script
    assert "operatorState.seized.add(command.target)" in script
    assert 'lastReceipt.outcome = "operator_released"' in script
    assert "operatorState.manualStep = command.value % radices[0]" in script
    assert 'if (action === "replay")' in script


def test_automatic_mode_clears_pending_operator_interventions() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert 'if (mode === "automatic" && operatorEvents.length)' in script
    assert "resetOperatorState();" in script
    assert 'const activeCommands = operatorEnabled() ? [...operatorEvents] : [];' in script


def test_invalid_request_clears_prior_evidence() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert 'clearResult("INVALID REQUEST")' in script
    assert "lastReceipt = null;" in script
    assert "lastTones = null;" in script


def test_faulted_observed_tones_drive_lab_outputs() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert "observedTones.route_hz += 7" in script
    assert "lastTones = observedTones" in script
    assert "drawScope(observedTones)" in script
    assert "expected_tones: expectedTones" in script
    assert "observed_tones: observedTones" in script


def test_lab_supports_audio_and_wav_without_dependencies() -> None:
    script = (LAB / "app.js").read_text(encoding="utf-8")
    assert "AudioContext" in script
    assert "audio/wav" in script
    assert "node_modules" not in script
