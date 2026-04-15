from dataclasses import FrozenInstanceError
import inspect
import random

import pytest

import qec.simulation.correlated_noise_simulator as correlated_noise_simulator
from qec.simulation.correlated_noise_simulator import (
    CorrelatedNoiseConfig,
    CorrelatedNoiseSimulator,
    build_noise_receipt,
    build_topology_adjacency,
    generate_correlated_noise,
    summarize_noise_realization,
    validate_noise_config,
)


BASE_CONFIG = {
    "model": "independent_baseline",
    "topology": "ring",
    "num_sites": 9,
    "time_steps": 12,
    "event_rate": 0.21,
    "seed": 138100,
}


@pytest.mark.parametrize(
    "model",
    (
        "independent_baseline",
        "temporal_markov",
        "nearest_neighbor_spatial",
        "spatiotemporal_cluster",
    ),
)
def test_replay_stability_per_model(model: str) -> None:
    cfg = dict(BASE_CONFIG)
    cfg["model"] = model

    run_a = generate_correlated_noise(cfg)
    run_b = generate_correlated_noise(cfg)

    assert run_a.to_canonical_json() == run_b.to_canonical_json()
    assert run_a.stable_hash() == run_b.stable_hash()

    report_a = summarize_noise_realization(run_a)
    report_b = summarize_noise_realization(run_b)
    assert report_a.stable_hash() == report_b.stable_hash()

    receipt_a = build_noise_receipt(cfg, run_a, report_a)
    receipt_b = build_noise_receipt(cfg, run_b, report_b)
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()


def test_topology_validation() -> None:
    cfg = dict(BASE_CONFIG)
    cfg["topology"] = "hex"
    with pytest.raises(ValueError, match="unsupported topology"):
        validate_noise_config(cfg)


def test_seed_determinism() -> None:
    cfg = dict(BASE_CONFIG)
    first = generate_correlated_noise(cfg)
    second = generate_correlated_noise(cfg)
    assert first.stable_hash() == second.stable_hash()


def test_different_seeds_produce_different_hashes() -> None:
    base = dict(BASE_CONFIG)
    other = dict(BASE_CONFIG)
    other["seed"] = BASE_CONFIG["seed"] + 1

    first = generate_correlated_noise(base)
    second = generate_correlated_noise(other)
    assert first.stable_hash() != second.stable_hash()


def test_shuffled_config_mapping_same_hash() -> None:
    shuffled = {
        "seed": BASE_CONFIG["seed"],
        "event_rate": BASE_CONFIG["event_rate"],
        "time_steps": BASE_CONFIG["time_steps"],
        "num_sites": BASE_CONFIG["num_sites"],
        "topology": BASE_CONFIG["topology"],
        "model": BASE_CONFIG["model"],
    }
    run_a = generate_correlated_noise(BASE_CONFIG)
    run_b = generate_correlated_noise(shuffled)
    assert run_a.stable_hash() == run_b.stable_hash()


def test_immutability() -> None:
    cfg = validate_noise_config(BASE_CONFIG)
    with pytest.raises(FrozenInstanceError):
        cfg.model = "temporal_markov"  # type: ignore[misc]


def test_decoder_untouched_guarantee() -> None:
    cfg = validate_noise_config(BASE_CONFIG)
    simulator = CorrelatedNoiseSimulator(config=cfg)
    source = inspect.getsource(generate_correlated_noise)

    assert simulator.decoder_untouched is True
    assert "qec.decoder" not in source


def test_topology_shapes_line_ring_grid() -> None:
    line = dict(BASE_CONFIG)
    line["topology"] = "line"
    ring = dict(BASE_CONFIG)
    ring["topology"] = "ring"
    grid = dict(BASE_CONFIG)
    grid["topology"] = "grid"
    grid["num_sites"] = 10

    line_adj = build_topology_adjacency(line)
    ring_adj = build_topology_adjacency(ring)
    grid_adj = build_topology_adjacency(grid)

    assert line_adj[0] == (1,)
    assert ring_adj[0] == (1, 8)
    assert len(grid_adj) == 10


def test_explicit_seed_required() -> None:
    cfg = dict(BASE_CONFIG)
    cfg.pop("seed")
    with pytest.raises(KeyError):
        validate_noise_config(cfg)


def test_mismatched_version_rejected() -> None:
    cfg = dict(BASE_CONFIG)
    cfg["version"] = "v999.0.0"
    with pytest.raises(ValueError, match="unsupported schema version"):
        validate_noise_config(cfg)


def test_no_hidden_global_rng_state() -> None:
    cfg = dict(BASE_CONFIG)
    random.seed(999999)
    a = generate_correlated_noise(cfg)
    random.seed(123456)
    b = generate_correlated_noise(cfg)
    assert a.stable_hash() == b.stable_hash()


def test_spatiotemporal_cluster_marks_future_members_active(monkeypatch: pytest.MonkeyPatch) -> None:
    class SequenceRng:
        def __init__(self, values: list[float], randrange_values: list[int]) -> None:
            self._values = iter(values)
            self._randrange_values = iter(randrange_values)

        def random(self) -> float:
            return next(self._values)

        def randrange(self, stop: int) -> int:
            candidate = next(self._randrange_values)
            assert 0 <= candidate < stop
            return candidate

    monkeypatch.setattr(
        correlated_noise_simulator.random,
        "Random",
        lambda seed: SequenceRng(
            values=[
                0.05,  # t=0 baseline event draw: pass
                0.05,  # t=0 cluster trigger draw: pass
                0.10,  # t=0 temporal frontier member draw: pass
                0.50,  # t=1 baseline event draw: would fail without scheduled activation
                0.95,  # t=1 cluster trigger draw: fail (keeps sequence bounded)
                0.50,  # t=2 baseline event draw: passes only if t=1 is active
                0.99,  # t=2 cluster trigger draw: fail (keeps sequence bounded)
            ],
            randrange_values=[1],  # target_size -> 2, so one frontier member can be added
        ),
    )

    cfg = {
        "model": "spatiotemporal_cluster",
        "topology": "line",
        "num_sites": 1,
        "time_steps": 3,
        "event_rate": 0.2,
        "temporal_alpha": 1.0,
        "spatial_beta": 0.0,
        "cluster_rate": 1.0,
        "cluster_max_size": 2,
        "seed": 138100,
    }
    realization = generate_correlated_noise(cfg)
    event_keys = {(event.time_step, event.site_index) for event in realization.events}
    assert (1, 0) in event_keys
    assert (2, 0) in event_keys
