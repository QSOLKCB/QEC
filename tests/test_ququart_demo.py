from qec.decoder.ququart.demo import certificate
from qec.sonify.canonical import canonical_sha256


def test_certificate_covers_all_single_ququart_paulis():
    payload = certificate()
    assert payload["code"]["parameters"] == [5, 1, 3]
    assert payload["code"]["exact_distance_through_weight_3"] == 3
    assert payload["errors_tested"] == 75
    assert payload["errors_corrected"] == 75
    assert payload["all_corrected"]
    claimed = payload.pop("sha256")
    assert claimed == canonical_sha256(payload)
