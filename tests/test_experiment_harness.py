from src.qec.discovery.discovery_engine import run_structure_discovery


def test_harness_reproducibility(deterministic_harness):
    def dummy():
        return 42

    run1 = deterministic_harness.run(dummy)
    run2 = deterministic_harness.run(dummy)

    assert run1["result"] == run2["result"]
    assert run1["experiment_hash"] == run2["experiment_hash"]


def test_discovery_engine_runs_under_harness(deterministic_harness):
    spec = {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }
    run = deterministic_harness.run(
        run_structure_discovery,
        spec,
        num_generations=1,
        population_size=2,
        base_seed=0,
    )

    assert "result" in run
    assert "metadata" in run
    assert "experiment_hash" in run


def test_harness_hash_depends_on_result(deterministic_harness):
    run1 = deterministic_harness.run(lambda: 1)
    run2 = deterministic_harness.run(lambda: 2)

    assert run1["metadata"] == run2["metadata"]
    assert run1["experiment_hash"] != run2["experiment_hash"]
