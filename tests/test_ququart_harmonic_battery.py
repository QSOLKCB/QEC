from qec.benchmark.ququart_battery.harmonic import (
    harmonic_end_to_end_rows,
    harmonic_fault_rows,
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


def test_end_to_end_harmonic_cells_are_deterministic():
    kwargs = {
        "physical_error_rates": ("0.01",),
        "noise_sigmas": ("0", "0.2"),
        "trials": 100,
        "seed": 23,
    }
    assert harmonic_end_to_end_rows(**kwargs) == harmonic_end_to_end_rows(**kwargs)
