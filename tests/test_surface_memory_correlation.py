"""Physical controls for the optional CPU surface memory benchmark."""
import numpy as np
import pytest

from bench.interop import memory_correlation as memory
from bench.interop import imports


@pytest.fixture
def backend():
    if not imports.HAS_STIM or not imports.HAS_PYMATCHING:
        pytest.skip("optional Stim/PyMatching dependencies unavailable")


def test_missing_dependencies_are_explicit(monkeypatch):
    monkeypatch.setattr(imports, "HAS_STIM", False)
    assert memory.run_experiment()["status"] == "skipped"


def test_stationary_marginals_and_lag_correlation():
    rates = np.array([0.03, 0.13])
    faults = memory.sample_faults(rates, 8, 50000, 0.7, 71)
    assert np.max(np.abs(faults.mean(axis=0) - rates)) < 0.004
    x, y = faults[:, :-1].reshape(-1, 2), faults[:, 1:].reshape(-1, 2)
    cov = (x*y).mean(axis=0) - x.mean(axis=0)*y.mean(axis=0)
    assert np.allclose(cov / (rates*(1-rates)), 0.7, atol=0.02)
    iid = memory.sample_faults(rates, 8, 50000, 0., 71)
    x, y = iid[:, :-1].reshape(-1, 2), iid[:, 1:].reshape(-1, 2)
    assert np.max(np.abs((x*y).mean(axis=0) - x.mean(axis=0)*y.mean(axis=0))) < 0.001


def test_placements_change_physical_rates_not_mean():
    for d in (3, 5):
        a = memory.physical_rates(d, .01, "stripe")
        b = memory.physical_rates(d, .01, "diagonal")
        assert not np.array_equal(a, b)
        assert a.mean() == pytest.approx(.01)
        assert b.mean() == pytest.approx(.01)
        assert sorted(a) == sorted(b)


@pytest.mark.parametrize("d", [3, 5])
def test_physical_fault_controls_and_superposition(backend, d):
    r = 3
    response = memory.fault_response(d, r)
    assert np.any(response[:, :-1])  # a fault actually changes detectors
    assert np.any(response[:, -1])  # physical propagation can flip memory-Z
    faults = memory.sample_faults(np.full(d*d, .2), r, 12, .65, 28)
    predicted = memory.propagate_faults(faults, response)
    for history, expected in zip(faults, predicted):
        circuit, _ = memory.build_memory_circuit(d, r, history)
        observed = circuit.compile_detector_sampler(seed=5).sample(4, append_observables=True)
        assert np.all(observed == expected)
    # A full left column X is a logical X: flips Z_L without any detectors.
    logical = np.zeros((1, r, d*d), dtype=np.uint8)
    logical[0, 0, ::d] = 1
    propagated = memory.propagate_faults(logical, response)[0]
    assert not propagated[:-1].any()
    assert propagated[-1] == 1
    # Repeated X on the same site at consecutive boundaries cancels final
    # logical parity but must still leave the intermediate detector record.
    repeated = np.zeros_like(logical)
    repeated[0, :2, 0] = 1
    propagated = memory.propagate_faults(repeated, response)[0]
    assert propagated[:-1].any()
    assert propagated[-1] == 0


def test_no_noise_and_relabeling(backend):
    result = memory.run_experiment(distances=(3,), rounds=(1, 3), p=0., shots=100)
    assert all(x["failures"] == 0 and x["empirical_fault_rate"] == 0 for x in result["records"])
    assert all(not any(x["detector_rates"]) for x in result["records"])
    result = memory.run_experiment(distances=(3,), rounds=(3,), shots=400, p=.05)
    for model in ("iid", "temporal_markov"):
        controls = [x for x in result["records"] if x["homogeneous_control"] and x["model"] == model]
        assert len(controls) == 2
        assert controls[0]["fault_history_sha256"] == controls[1]["fault_history_sha256"]
        assert controls[0]["outcomes_sha256"] == controls[1]["outcomes_sha256"]
        assert controls[0]["failures"] == controls[1]["failures"]
    assert len({x["decoder_dem_sha256"] for x in result["records"]}) == 1


def test_replay_receipt_and_validation(backend):
    kwargs = dict(distances=(3,), rounds=(2,), shots=100, seed=9)
    a = memory.run_experiment(**kwargs)
    assert a == memory.run_experiment(**kwargs)
    receipt = a.pop("receipt_sha256")
    assert memory._hash(a) == receipt
    assert a["source_sha256"]["memory_correlation.py"]
    for bad in ({"shots": 0}, {"rho": 1.}, {"p": float("nan")}, {"distances": (7,)}, {"rounds": (0,)}, {"rounds": (151,)}, {"rounds": (150,), "shots": 20000}, {"shots": 20001}):
        with pytest.raises(ValueError):
            memory.run_experiment(**bad)


def test_wilson_bounds():
    assert memory.wilson_interval(0, 100)[1] > 0
    assert memory.wilson_interval(100, 100)[0] < 1


def test_checked_in_receipt_is_source_bound():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / "tests/fixtures/surface_memory_correlation/cpu_v1_seed12345.json").read_text())
    digest = receipt.pop("receipt_sha256")
    assert memory._hash(receipt) == digest
    for filename, expected in receipt["source_sha256"].items():
        assert memory._hash((root / "src/bench/interop" / filename).read_bytes()) == expected
    assert len(receipt["records"]) == 48


def test_long_round_circuit_and_resource_preflight(backend, monkeypatch):
    history = np.zeros((150, 25))
    circuit, _ = memory.build_memory_circuit(5, 150, history)
    assert circuit.num_detectors == 150*24
    assert not circuit.compile_detector_sampler(seed=1).sample(4, append_observables=True).any()
    history[-1, ::5] = 1
    circuit, _ = memory.build_memory_circuit(5, 150, history)
    outcomes = circuit.compile_detector_sampler(seed=1).sample(4, append_observables=True)
    assert not outcomes[:, :-1].any()
    assert outcomes[:, -1].all()
    # Rejection occurs before any circuit/response allocation.
    def must_not_allocate(*args, **kwargs):
        raise AssertionError("preflight did not reject before allocation")
    monkeypatch.setattr(memory, "build_memory_circuit", must_not_allocate)
    with pytest.raises(ValueError, match="resource budget exceeded"):
        memory.run_experiment(distances=(5,), rounds=(150,), shots=2000)
