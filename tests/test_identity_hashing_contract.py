from qec.analysis.identity_hashing_contract import get_identity_hashing_contract


def test_contract_exists() -> None:
    contract = get_identity_hashing_contract().lower()
    assert "identity" in contract
    assert "hash" in contract


def test_contract_key_guarantees() -> None:
    contract = get_identity_hashing_contract()
    assert "canonical_hash_identity" in contract
    assert "canonical JSON" in contract
    assert "SHA-256" in contract
    assert "exclude self-referential hash fields" in contract
