import json

from qec.analysis.quantum_noise_balancing_layer import (
    GENESIS_HASH,
    PerturbationLedger,
    PerturbationLedgerEntry,
    append_perturbation_ledger_entry,
    build_noise_compensation_model,
    compute_bounded_perturbation_metrics,
    compute_fidelity_stability_score,
    compute_noise_compensation,
    normalize_noise_inputs,
    run_quantum_noise_balancing_layer,
    validate_perturbation_ledger,
)


def test_deterministic_normalization():
    a = normalize_noise_inputs({"zeta": 0.1, "alpha": 0.2})
    b = normalize_noise_inputs((("zeta", 0.1),))
    assert a == (("alpha", 0.2), ("zeta", 0.1))
    assert b == (("zeta", 0.1),)


def test_bounded_perturbation_outputs():
    normalized = normalize_noise_inputs({"a": 0.0, "b": 1.0, "c": 0.7})
    metrics = compute_bounded_perturbation_metrics(normalized)
    assert all(0.0 <= x <= 1.0 for x in metrics)


def test_compensation_reproducibility():
    normalized = normalize_noise_inputs({"a": 0.2, "b": 0.8, "c": 0.5})
    model1 = build_noise_compensation_model(normalized)
    model2 = build_noise_compensation_model(normalized)
    assert model1.model_hash == model2.model_hash
    assert compute_noise_compensation(normalized, model1) == compute_noise_compensation(
        normalized,
        model2,
    )


def test_fidelity_score_boundedness():
    fidelity, _, _ = compute_fidelity_stability_score(1.2, -0.1, 2.2, -4.0)
    assert 0.0 <= fidelity <= 1.0


def test_stability_score_boundedness():
    _, stability, _ = compute_fidelity_stability_score(0.8, 0.1, 0.9, 0.3)
    assert 0.0 <= stability <= 1.0


def test_same_input_same_bytes():
    noise = {"s1": 0.22, "s2": 0.41, "s3": 0.63}
    report1, ledger1 = run_quantum_noise_balancing_layer(noise)
    report2, ledger2 = run_quantum_noise_balancing_layer(noise)
    assert report1.to_canonical_json().encode("utf-8") == report2.to_canonical_json().encode(
        "utf-8",
    )
    assert ledger1.to_canonical_json().encode("utf-8") == ledger2.to_canonical_json().encode(
        "utf-8",
    )


def test_ledger_chain_stability():
    ledger = PerturbationLedger(entries=(), head_hash=GENESIS_HASH, chain_valid=True)
    ledger = append_perturbation_ledger_entry(ledger, "h1", 0.9, 0.8)
    ledger = append_perturbation_ledger_entry(ledger, "h2", 0.7, 0.95)
    assert validate_perturbation_ledger(ledger)


def test_corruption_detection():
    entry = PerturbationLedgerEntry(
        sequence_id=0,
        perturbation_hash="h1",
        parent_hash=GENESIS_HASH,
        fidelity_score=0.9,
        stability_score=0.9,
    )
    bad = PerturbationLedger(entries=(entry,), head_hash="bad", chain_valid=True)
    assert not validate_perturbation_ledger(bad)


def test_no_decoder_imports():
    import qec.analysis.quantum_noise_balancing_layer as layer

    names = set(layer.__dict__.keys())
    assert "decoder" not in " ".join(sorted(names)).lower()


def test_insertion_order_independence():
    n1 = {"a": 0.1, "b": 0.4, "c": 0.9}
    n2 = {"c": 0.9, "a": 0.1, "b": 0.4}
    report1, ledger1 = run_quantum_noise_balancing_layer(n1)
    report2, ledger2 = run_quantum_noise_balancing_layer(n2)
    assert report1.to_canonical_json() == report2.to_canonical_json()
    assert ledger1.to_canonical_json() == ledger2.to_canonical_json()


def test_zero_noise_stable_baseline():
    report, ledger = run_quantum_noise_balancing_layer({"a": 0.0, "b": 0.0})
    assert report.fidelity_score == 1.0
    assert report.stability_score == 1.0
    assert report.balanced is True
    assert ledger.chain_valid is True


def test_max_noise_bounded_saturation():
    report, ledger = run_quantum_noise_balancing_layer({"a": 1.0, "b": 1.0, "c": 1.0})
    payload = json.loads(report.to_canonical_json())
    assert 0.0 <= payload["fidelity_score"] <= 1.0
    assert 0.0 <= payload["stability_score"] <= 1.0
    assert 0.0 <= payload["compensation_effectiveness"] <= 1.0
    assert ledger.chain_valid is True
