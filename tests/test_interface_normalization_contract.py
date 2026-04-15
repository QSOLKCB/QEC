from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.interface.interface_normalization_contract import InterfaceNormalizationContract


def _raw_capture(**overrides):
    base = {
        "source_id": "hw-a",
        "capture_type": "syndrome",
        "signal_payload": {"z1": 1, "x0": 0, "x1": 1, "z0": 0},
        "shape": [2, 2],
        "dtype": "np.float32",
        "timestamp_receipt_hash": "time-hash-1",
        "suppression_receipt_hash": "supp-hash-1",
        "metadata": {
            "hardware_temp_c": "12.5",
            "timing_cycle": 88,
            "operator": "unit-test",
        },
    }
    base.update(overrides)
    return base


def test_happy_path_normalization():
    contract = InterfaceNormalizationContract()
    package, report, receipt = contract.normalize(_raw_capture())

    assert package.syndrome_bits == (0, 1, 0, 1)
    assert package.shape == (2, 2)
    assert package.normalization_version == "v137.21.4"
    assert report.validation_passed is True
    assert receipt.contract_valid is True


def test_shuffled_metadata_ordering_yields_identical_hashes():
    contract = InterfaceNormalizationContract()
    raw_a = _raw_capture(metadata={"a": 1, "b": 2, "c": 3})
    raw_b = _raw_capture(metadata={"c": 3, "a": 1, "b": 2})

    package_a, report_a, receipt_a = contract.normalize(raw_a)
    package_b, report_b, receipt_b = contract.normalize(raw_b)

    assert package_a.stable_hash() == package_b.stable_hash()
    assert report_a.stable_hash() == report_b.stable_hash()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()


def test_shape_alias_normalization():
    contract = InterfaceNormalizationContract()
    package_list, _, _ = contract.normalize(_raw_capture(shape=[2, 2]))
    package_tuple, _, _ = contract.normalize(_raw_capture(shape=(2, 2)))
    package_string, _, _ = contract.normalize(_raw_capture(shape="(2,2)"))

    assert package_list.shape == (2, 2)
    assert package_list.stable_hash() == package_tuple.stable_hash() == package_string.stable_hash()


def test_dtype_alias_normalization():
    contract = InterfaceNormalizationContract()
    a, _, _ = contract.normalize(_raw_capture(dtype="float32"))
    b, _, _ = contract.normalize(_raw_capture(dtype="np.float32"))
    c, _, _ = contract.normalize(_raw_capture(dtype="FLOAT32"))

    assert a.logical_payload_hash == b.logical_payload_hash == c.logical_payload_hash


def test_malformed_shape_failure():
    contract = InterfaceNormalizationContract()
    with pytest.raises((TypeError, ValueError)):
        contract.normalize(_raw_capture(shape="2xbad"))


def test_shape_cardinality_must_match_syndrome_bit_count():
    contract = InterfaceNormalizationContract()
    with pytest.raises(ValueError, match="bit count must match shape cardinality"):
        contract.normalize(_raw_capture(shape=[2, 2], signal_payload=[0, 1, 1]))


def test_non_string_key_rejection():
    contract = InterfaceNormalizationContract()
    with pytest.raises(TypeError):
        contract.normalize(_raw_capture(metadata={1: "bad"}))


def test_timing_metadata_separation_from_logical_payload():
    contract = InterfaceNormalizationContract()
    p1, _, _ = contract.normalize(_raw_capture(metadata={"timing_cycle": 10}))
    p2, _, _ = contract.normalize(_raw_capture(metadata={"timing_cycle": 999}))

    assert p1.logical_payload_hash == p2.logical_payload_hash


def test_suppression_receipts_stay_sideband_only():
    contract = InterfaceNormalizationContract()
    p1, _, _ = contract.normalize(_raw_capture(suppression_receipt_hash="supp-A"))
    p2, _, _ = contract.normalize(_raw_capture(suppression_receipt_hash="supp-B"))

    assert p1.logical_payload_hash == p2.logical_payload_hash
    assert p1.sideband_receipt_hashes != p2.sideband_receipt_hashes


def test_stable_repeated_hash_equality():
    contract = InterfaceNormalizationContract()
    package, report, receipt = contract.normalize(_raw_capture())

    assert package.stable_hash() == package.stable_hash()
    assert report.stable_hash() == report.stable_hash()
    assert receipt.stable_hash() == receipt.stable_hash()


def test_decoder_untouched_guarantee():
    import ast
    import pathlib

    mod = __import__("qec.interface.interface_normalization_contract", fromlist=["*"])
    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("qec.decoder"):
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("qec.decoder"):
                violations.append(f"from {module} import ...")
    assert not violations, f"Unexpected qec.decoder imports: {violations}"


def test_report_and_receipt_immutability():
    contract = InterfaceNormalizationContract()
    _, report, receipt = contract.normalize(_raw_capture())

    with pytest.raises(FrozenInstanceError):
        report.contract_version = "other"
    with pytest.raises(FrozenInstanceError):
        receipt.contract_valid = False


def test_logical_payload_purity_invariant():
    contract = InterfaceNormalizationContract()
    base = _raw_capture(
        metadata={"timing_cycle": 2, "hardware_lane": "a"},
        timestamp_receipt_hash="time-A",
        suppression_receipt_hash="supp-A",
    )
    changed_sideband = _raw_capture(
        metadata={"timing_cycle": 999, "hardware_lane": "z", "extra": "x"},
        timestamp_receipt_hash="time-B",
        suppression_receipt_hash="supp-B",
    )

    p1, _, _ = contract.normalize(base)
    p2, _, _ = contract.normalize(changed_sideband)

    assert p1.syndrome_bits == p2.syndrome_bits
    assert p1.shape == p2.shape
    assert p1.logical_payload_hash == p2.logical_payload_hash
