"""Formal contract linking canonical identity and hashing semantics."""

from __future__ import annotations

IDENTITY_HASHING_CONTRACT = """
QEC Identity & Hashing Surface Contract

Identity and hashing are distinct but coupled invariants.

IDENTITY:
- canonical_hash_identity(...) defines identity equivalence
- applies to all identity-bearing hash sequences
- MUST enforce:
  • lowercase SHA-256 hex
  • sorted
  • unique
  • tuple[str, ...]

HASHING:
- *_decision_hash / *_receipt_hash functions define artifact integrity
- MUST:
  • operate only on canonicalized data
  • use canonical JSON serialization
  • produce stable SHA-256 hashes
  • exclude self-referential hash fields

RELATIONSHIP:
- identity defines "what is equivalent"
- hashing defines "what is provable"

REQUIREMENTS:
- hashing MUST depend only on canonicalized identity inputs
- identity MUST be enforced before hashing
- no module may:
  • hash non-canonical identity inputs
  • redefine identity semantics
  • bypass canonical_hash_identity

SYSTEM LAW:
identity → canonicalization → hashing → proof

Violation → INVALID_INPUT
"""


def get_identity_hashing_contract() -> str:
    return IDENTITY_HASHING_CONTRACT
