import random

from qec.benchmark.ququart_battery.channels import (
    CHANNELS,
    monte_carlo_rows,
    sample_error,
)


def test_full_channel_contains_all_15_local_nonidentity_products():
    assert len(CHANNELS["full_packed_depolarizing"]) == 15
    assert ("I", "X") in CHANNELS["full_packed_depolarizing"]
    assert ("X", "I") in CHANNELS["full_packed_depolarizing"]
    assert ("Y", "Z") in CHANNELS["full_packed_depolarizing"]


def test_per_site_sampler_respects_endpoints():
    rng = random.Random(7)
    assert sample_error(rng, width=5, error_rate=0.0).weight == 0
    assert sample_error(rng, width=5, error_rate=1.0).weight == 5


def test_monte_carlo_is_cell_deterministic():
    first = monte_carlo_rows(
        error_rates=("0.03",),
        channels=("full_packed_depolarizing",),
        trials=250,
        seed=17,
    )
    second = monte_carlo_rows(
        error_rates=("0.03",),
        channels=("full_packed_depolarizing",),
        trials=250,
        seed=17,
    )
    assert first == second
    assert first[0]["trials"] == 250
