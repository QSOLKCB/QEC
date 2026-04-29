import re

from qec.analysis.identity_hashing_contract import get_identity_hashing_contract


def _normalize_contract_text(text: str) -> str:
    """Normalize contract text for stable documentation-style assertions."""
    return re.sub(r"\s+", " ", text).strip().lower()


def test_contract_exists() -> None:
    contract = _normalize_contract_text(get_identity_hashing_contract())
    assert "identity" in contract
    assert "hash" in contract


def test_contract_key_guarantees() -> None:
    contract = get_identity_hashing_contract()
    normalized_contract = _normalize_contract_text(contract)

    assert "canonical_hash_identity" in contract
    assert "canonical json" in normalized_contract
    assert "sha-256" in normalized_contract
    assert "exclude self-referential hash fields" in normalized_contract
