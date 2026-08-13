# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import copy

import pytest

from qec.routing.panel import PanelExchange, PanelRequest, compare_strowger_panel, demo_topology, demo_translation
from qec.routing.strowger import ExchangeConfig, RouteRequest, StageConfig, StrowgerExchange
from qec.sonify.canonical import canonical_sha256

DEST = "ququart/site-4/pauli-11"
DIGITS = (2, 3, 4, 11)


def make_strowger_receipt():
    config = ExchangeConfig(
        linefinders=2,
        selectors=(
            StageConfig("code-family", 3, trunks=3),
            StageConfig("syndrome-sector", 4, trunks=3),
        ),
        connector_vertical_radix=10,
        connector_rotary_radix=16,
    )
    request = RouteRequest("cross-era", DIGITS, 7, DEST)
    return StrowgerExchange(config).route(request).receipt


def make_panel_receipt():
    request = PanelRequest("cross-era", DIGITS, 7, DEST, b"payload")
    return PanelExchange(demo_topology(DEST), demo_translation(DIGITS, DEST)).route(request).receipt


def recompute_outer_hash(receipt: dict[str, object]) -> None:
    unsigned = dict(receipt)
    unsigned.pop("sha256", None)
    receipt["sha256"] = canonical_sha256(unsigned)


def test_cross_era_comparison_validates_source_receipts_before_equivalence():
    strowger = make_strowger_receipt()
    panel = make_panel_receipt()
    assert compare_strowger_panel(strowger, panel)["equivalent"] is True

    modified = copy.deepcopy(strowger)
    modified["outcome"] = "tone_mismatch"
    recompute_outer_hash(modified)
    with pytest.raises(ValueError, match="replay mismatch"):
        compare_strowger_panel(modified, panel)
