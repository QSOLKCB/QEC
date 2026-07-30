"""Exact four-state QEC with native-ququart and packed-qubit semantics."""

from .codes import packed_five_ququart_code
from .exact import DecodeResult, ExactDecoder, UncorrectableSyndrome
from .gates import (
    ENCODED_X_BOTH,
    ENCODED_X_LANE0,
    ENCODED_X_LANE1,
    H4,
    INTERNAL_SWAP,
    NATIVE_X4,
    X4,
    Z4,
    apply_permutation,
    basis_to_qubits,
    permutation_matrix,
    qubits_to_basis,
)
from .packed import (
    PackedPauli,
    PackedQuquartCode,
    exact_distance,
    identity,
    local_pauli,
    paulis_of_weight,
)
from .syndrome import (
    QuquartSyndrome,
    pack_binary_syndrome,
    unpack_ququart_syndrome,
)

__all__ = [
    "DecodeResult",
    "ENCODED_X_BOTH",
    "ENCODED_X_LANE0",
    "ENCODED_X_LANE1",
    "ExactDecoder",
    "H4",
    "INTERNAL_SWAP",
    "NATIVE_X4",
    "PackedPauli",
    "PackedQuquartCode",
    "QuquartSyndrome",
    "UncorrectableSyndrome",
    "X4",
    "Z4",
    "apply_permutation",
    "basis_to_qubits",
    "exact_distance",
    "identity",
    "local_pauli",
    "pack_binary_syndrome",
    "packed_five_ququart_code",
    "paulis_of_weight",
    "permutation_matrix",
    "qubits_to_basis",
    "unpack_ququart_syndrome",
]
