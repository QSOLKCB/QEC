"""Deterministic command-line certificate for the four-state QEC slice."""

from __future__ import annotations

from qec.sonify.canonical import canonical_json, canonical_sha256

from .codes import packed_five_ququart_code
from .cycle import run_harmonic_cycle
from .exact import ExactDecoder
from .packed import exact_distance, paulis_of_weight


def certificate() -> dict[str, object]:
    """Return an exact finite certificate for all single-ququart Pauli errors."""
    code = packed_five_ququart_code()
    decoder = ExactDecoder(code, max_weight=1)
    errors = tuple(paulis_of_weight(code.n, 1))
    corrected = sum(run_harmonic_cycle(decoder, error).success for error in errors)
    payload: dict[str, object] = {
        "schema": "qec.ququart.harmonic-certificate.v1",
        "representation": "encoded-two-qubit-additive-ququart",
        "native_gate_layer": ["X4", "Z4", "H4"],
        "decoder": "exact-bounded-coset-leader-gf2xgf2",
        "code": {
            "name": code.name,
            "parameters": [code.n, code.k, code.distance_hint],
            "exact_distance_through_weight_3": exact_distance(code, max_weight=3),
            "stabilizer_generators": 2 * len(code.base_stabilizers),
            "codespace_dimension": 4 ** code.k,
        },
        "certified_ququart_weight": 1,
        "errors_tested": len(errors),
        "errors_corrected": corrected,
        "all_corrected": corrected == len(errors),
        "harmonics": {
            "state_identifying": [1, 3],
            "parity_only": [2],
            "state_dark": [4],
            "policy": "reject-disagreement-before-correction",
        },
        "claim_scope": (
            "Exact finite correction of the 15-operator Pauli basis on any one "
            "packed ququart, with classical harmonic syndrome observation; not "
            "hardware-QEC performance or a naive GF(4)=Z4 claim."
        ),
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    print(canonical_json(certificate()))


if __name__ == "__main__":
    main()
