"""Global canonical identity contract for analysis-layer hash identities."""

from __future__ import annotations

IDENTITY_CONTRACT = """
All identity-bearing hash sequences in QEC MUST satisfy:

1. Each element is a lowercase SHA-256 hex string (64 chars, [0-9a-f])
2. The sequence is strictly sorted (lexicographic order)
3. The sequence contains no duplicates
4. The sequence is represented as an immutable tuple[str, ...]

Canonical enforcement MUST be performed using:

    canonical_hash_identity(...)

This applies to:

- input_memory_hashes
- governance_hashes
- node_proof_hashes
- proof identity tuples
- any future identity-bearing hash collections

Violation → INVALID_INPUT

No module may:

- accept non-canonical identity sequences
- silently normalize identity sequences
- bypass canonical_hash_identity

This contract is GLOBAL and MUST be enforced consistently
across all analysis-layer modules.
"""


def get_identity_contract() -> str:
    return IDENTITY_CONTRACT


__all__ = ["IDENTITY_CONTRACT", "get_identity_contract"]
