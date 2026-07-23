"""Exhaustive harmonic-channel fault injection over certified errors."""

from __future__ import annotations

from functools import lru_cache

from qec.decoder.qutrit import (
    ExactDecoder,
    cyclic_five_qutrit_code,
    paulis_of_weight,
    shor_nine_qutrit_code,
    ternary_golay_qutrit_code,
)
from qec.decoder.qutrit.cycle import run_harmonic_cycle
from qec.sonify.qutrit_harmonics import encode_harmonics, phasor


@lru_cache(maxsize=1)
def harmonic_fault_rows() -> tuple[dict[str, str | int], ...]:
    rows = []
    declarations = (
        ("qutrit_cyclic_5", cyclic_five_qutrit_code, 1),
        ("qutrit_shor_9", shor_nine_qutrit_code, 1),
        ("qutrit_golay_11", ternary_golay_qutrit_code, 2),
    )
    for code_id, factory, radius in declarations:
        code = factory()
        decoder = ExactDecoder(code, max_weight=radius)
        counts = {
            "clean": {"accepted": 0, "successful": 0, "false_accepts": 0},
            "h2_disagreement": {
                "accepted": 0,
                "successful": 0,
                "false_accepts": 0,
            },
            "h3_dark_distortion": {
                "accepted": 0,
                "successful": 0,
                "false_accepts": 0,
            },
        }
        tested = 0
        for weight in range(1, radius + 1):
            for error in paulis_of_weight(code.n, weight):
                tested += 1
                syndrome = code.syndrome(error)
                clean = run_harmonic_cycle(decoder, error)

                disagreement = encode_harmonics(syndrome)
                disagreement[2] = (
                    phasor((syndrome[0] + 1) % 3, 2),
                ) + disagreement[2][1:]
                h2 = run_harmonic_cycle(
                    decoder,
                    error,
                    samples=disagreement,
                    tolerance=2.0,
                )

                distortion = encode_harmonics(syndrome)
                distortion[3] = (0j,) + distortion[3][1:]
                h3 = run_harmonic_cycle(
                    decoder,
                    error,
                    samples=distortion,
                )

                for case, result in (
                    ("clean", clean),
                    ("h2_disagreement", h2),
                    ("h3_dark_distortion", h3),
                ):
                    counts[case]["accepted"] += int(result.accepted)
                    counts[case]["successful"] += int(result.success)
                    counts[case]["false_accepts"] += int(
                        result.accepted and not result.success
                    )

        for case, result in counts.items():
            rows.append({
                "code_id": code_id,
                "certified_radius": radius,
                "fault_case": case,
                "errors_tested": tested,
                "accepted": result["accepted"],
                "successful": result["successful"],
                "false_accepts": result["false_accepts"],
                "expected": (
                    "accept_and_correct_all"
                    if case == "clean"
                    else "reject_all"
                ),
                "claim_scope": (
                    "classical_harmonic_observation_fault_injection"
                ),
            })
    return tuple(rows)
