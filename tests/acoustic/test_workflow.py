# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

WORKFLOW=Path(__file__).resolve().parents[2]/".github/workflows/acoustic-channel.yml"

def test_exact_head_checkout_retains_history_for_full_suite():
    text=WORKFLOW.read_text()
    checkouts=text.split("uses: actions/checkout@v4")[1:]
    assert len(checkouts)==2
    for section in checkouts:
        settings=section.split("- uses:",1)[0]
        assert "fetch-depth: 0" in settings
        assert "persist-credentials: false" in settings
        assert "github.event.pull_request.head.sha || github.sha" in settings

def test_cross_platform_and_full_gate_are_not_swallowed():
    text=WORKFLOW.read_text()
    for runner in ("ubuntu-latest","windows-latest","macos-14"):
        assert runner in text
    assert "python -m pytest -q tests/acoustic" in text
    assert "run: python -m pytest -q\n" in text
    assert "continue-on-error" not in text
    assert "|| true" not in text
