from decimal import Decimal

from qec.benchmark.ququart_battery.exact_channels import (
    exact_channel_fer_row,
    exact_channel_weight_enumerator,
    lane_symmetry_certificate,
)
from qec.benchmark.ququart_battery.oracle import exact_fer_row


def _projection(rows):
    return tuple(
        (
            row["weight"],
            row["patterns"],
            row["corrected"],
            row["detected_uncorrectable"],
            row["logical_failure"],
        )
        for row in rows
    )


def test_exact_restricted_channels_cover_their_full_alphabets():
    for channel in ("lane0_only", "lane1_only", "same_pauli_correlated"):
        rows = exact_channel_weight_enumerator(channel)
        assert sum(int(row["patterns"]) for row in rows) == 4 ** 5
        assert int(rows[1]["patterns"]) == 15


def test_lane_exchange_is_an_exact_invariant():
    lane0 = exact_channel_weight_enumerator("lane0_only")
    lane1 = exact_channel_weight_enumerator("lane1_only")
    assert _projection(lane0) == _projection(lane1)
    certificate = lane_symmetry_certificate()
    assert certificate["weight_enumerators_equal"] is True
    assert len(certificate["sha256"]) == 64


def test_full_channel_oracle_matches_v170_1_0_curve():
    for rate in ("0", "0.001", "0.03", "1"):
        upgraded = exact_channel_fer_row("full_packed_depolarizing", rate)
        original = exact_fer_row(rate)
        assert upgraded["frame_error_rate"] == original["frame_error_rate"]
        assert upgraded["success_rate"] == original["success_rate"]


def test_every_channel_normalizes_at_endpoints():
    for channel in (
        "full_packed_depolarizing",
        "lane0_only",
        "lane1_only",
        "same_pauli_correlated",
    ):
        at_zero = exact_channel_fer_row(channel, "0")
        at_one = exact_channel_fer_row(channel, "1")
        assert Decimal(at_zero["success_rate"]) == 1
        total = (
            Decimal(at_one["success_rate"])
            + Decimal(at_one["frame_error_rate"])
        )
        assert abs(total - Decimal(1)) < Decimal("1e-18")
