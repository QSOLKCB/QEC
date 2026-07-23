"""Public certificate tests for the qutrit executable."""

from qec.decoder.qutrit.demo import certificate
from qec.sonify.canonical import canonical_sha256


def test_certificate_covers_the_exact_correctable_sets():
    payload = certificate()
    results = {entry["code"]: entry for entry in payload["codes"]}
    assert results["cyclic-[[5,1,3]]_3"]["errors_tested"] == 40
    assert results["shor-[[9,1,3]]_3"]["errors_tested"] == 72
    assert results["golay-[[11,1,5]]_3"]["errors_tested"] == 3608
    assert all(entry["all_corrected"] for entry in results.values())

    claimed_hash = payload.pop("sha256")
    assert claimed_hash == canonical_sha256(payload)
