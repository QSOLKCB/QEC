"""Deterministic command-line certificate for the qutrit decoder slice."""

from __future__ import annotations

from qec.sonify.canonical import canonical_json, canonical_sha256

from .codes import (
    cyclic_five_qutrit_code,
    shor_nine_qutrit_code,
    ternary_golay_qutrit_code,
)
from .cycle import run_harmonic_cycle
from .exact import ExactDecoder
from .stabilizer import paulis_of_weight


def _certify(code, max_weight: int) -> dict[str, object]:
    decoder = ExactDecoder(code, max_weight=max_weight)
    tested = 0
    corrected = 0
    for weight in range(1, max_weight + 1):
        for error in paulis_of_weight(code.n, weight):
            tested += 1
            corrected += run_harmonic_cycle(decoder, error).success
    return {
        "code": code.name,
        "parameters": [code.n, code.k, code.distance_hint],
        "certified_weight": max_weight,
        "errors_tested": tested,
        "errors_corrected": corrected,
        "all_corrected": corrected == tested,
    }


def certificate() -> dict[str, object]:
    """Return the canonical exact-enumeration result."""
    payload: dict[str, object] = {
        "schema": "qec.qutrit.harmonic-certificate.v1",
        "decoder": "exact-bounded-coset-leader-gf3",
        "codes": [
            _certify(cyclic_five_qutrit_code(), 1),
            _certify(shor_nine_qutrit_code(), 1),
            _certify(ternary_golay_qutrit_code(), 2),
        ],
        "harmonics": {
            "state_identifying": [1, 2],
            "state_dark": [3],
            "policy": "reject-disagreement-before-correction",
        },
        "claim_scope": (
            "Exact finite qutrit Pauli correction and classical harmonic "
            "syndrome observation; not hardware-QEC performance."
        ),
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    print(canonical_json(certificate()))


if __name__ == "__main__":
    main()
