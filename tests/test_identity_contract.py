from __future__ import annotations

from qec.analysis.identity_contract import get_identity_contract


def test_contract_availability() -> None:
    assert "canonical_hash_identity" in get_identity_contract()


def test_contract_consistency_mentions_required_terms() -> None:
    contract = get_identity_contract()
    assert "sorted" in contract
    assert "duplicates" in contract
    assert "SHA-256" in contract
    assert "tuple" in contract
