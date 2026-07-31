from qec.benchmark.ququart_battery.harmonic import (
    harmonic_end_to_end_rows,
    harmonic_fault_rows,
    receiver_operating_rows,
)


def test_deterministic_fault_matrix_is_fail_closed():
    rows = {row["fault_case"]: row for row in harmonic_fault_rows()}
    assert rows["clean"]["successful"] == 75
    assert rows["bounded_complex_noise"]["successful"] == 75
    for name in (
        "missing_h3",
        "h1_h3_disagreement",
        "h2_parity_flip",
        "h4_dark_distortion",
        "ambiguous_h1",
    ):
        assert rows[name]["accepted"] == 0
        assert rows[name]["false_accepts"] == 0
        assert rows[name]["receiver_false_trust"] == 0


def test_end_to_end_harmonic_cells_are_deterministic_and_layered():
    kwargs = {
        "physical_error_rates": ("0.01",),
        "noise_sigmas": ("0", "0.2"),
        "trials": 100,
        "seed": 23,
    }
    first = harmonic_end_to_end_rows(**kwargs)
    assert first == harmonic_end_to_end_rows(**kwargs)
    for row in first:
        assert row["rejected"] == (
            row["receiver_rejections"] + row["decoder_rejections"]
        )
        assert row["incorrect_trusted_syndrome"] == row["receiver_false_trust"]
        assert row["false_accepts"] == (
            row["accepted_incorrect_syndrome"]
            + row["accepted_logical_residual"]
        )

    operating = receiver_operating_rows(first)
    assert len(operating) == len(first)
    assert all("receiver_false_trust_rate" in row for row in operating)
