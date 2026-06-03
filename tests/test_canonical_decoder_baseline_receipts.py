from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import canonical_decoder_baseline_receipts as cdb

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64
HEX_F = "f" * 64


def _parts():
    identity = cdb.build_canonical_decoder_identity("canonical decoder baseline")
    source_files = (
        cdb.build_canonical_decoder_source_file("src/qec/decoder/zeta.py", HEX_A),
        cdb.build_canonical_decoder_source_file("src/qec/decoder/alpha.py", HEX_B),
    )
    source_boundary = cdb.build_canonical_decoder_source_boundary(source_files)
    replay_boundary = cdb.build_canonical_decoder_replay_corpus_boundary(
        corpus_name="declared static replay corpus boundary",
        corpus_version="v166.0-corpus-declaration",
        corpus_hash=HEX_C,
        input_schema_hash=HEX_D,
        output_schema_hash=HEX_E,
    )
    equivalence_policy = cdb.build_canonical_decoder_equivalence_policy()
    immutability_boundary = cdb.build_canonical_decoder_immutability_boundary(
        ("src/qec/decoder/zeta.py", "src/qec/decoder/alpha.py")
    )
    receipt = cdb.build_canonical_decoder_baseline_receipt(
        upstream_graph_universe_claim_boundary_receipt_hash=HEX_F,
        identity=identity,
        source_boundary=source_boundary,
        replay_corpus_boundary=replay_boundary,
        equivalence_policy=equivalence_policy,
        immutability_boundary=immutability_boundary,
    )
    return (
        identity,
        source_boundary,
        replay_boundary,
        equivalence_policy,
        immutability_boundary,
        receipt,
    )


def _expect_error(expected, fn, *args, **kwargs):
    with pytest.raises(ValueError) as exc:
        fn(*args, **kwargs)
    if isinstance(exc.value, cdb.CanonicalDecoderBaselineError):
        assert exc.value.code.value == expected
    else:
        assert str(exc.value) == expected
    return exc.value


def test_happy_path_validates_hash_and_frozen_artifacts():
    (
        identity,
        source_boundary,
        replay_boundary,
        equivalence_policy,
        immutability_boundary,
        receipt,
    ) = _parts()

    cdb.validate_canonical_decoder_identity(identity)
    for source_file in source_boundary.source_files:
        cdb.validate_canonical_decoder_source_file(source_file)
    cdb.validate_canonical_decoder_source_boundary(source_boundary)
    cdb.validate_canonical_decoder_replay_corpus_boundary(replay_boundary)
    cdb.validate_canonical_decoder_equivalence_policy(equivalence_policy)
    cdb.validate_canonical_decoder_immutability_boundary(immutability_boundary)
    cdb.validate_canonical_decoder_baseline_receipt(receipt)

    assert cdb._HASH_RE.fullmatch(receipt.canonical_decoder_baseline_receipt_hash)
    assert receipt.replay_safe_canonical_decoder_baseline is True
    with pytest.raises(FrozenInstanceError):
        receipt.receipt_kind = "changed"


def test_canonical_json_and_hash_determinism_for_ordering_and_hash_seed_independence():
    ordered = (
        cdb.build_canonical_decoder_source_file("src/qec/decoder/alpha.py", HEX_A),
        cdb.build_canonical_decoder_source_file("src/qec/decoder/beta.py", HEX_B),
    )
    reversed_order = tuple(reversed(ordered))
    first = cdb.build_canonical_decoder_source_boundary(reversed_order)
    second = cdb.build_canonical_decoder_source_boundary(ordered)
    third = cdb.build_canonical_decoder_source_boundary(
        list({ordered[0].path: ordered[0], ordered[1].path: ordered[1]}.values())
    )

    assert first.source_files == ordered
    assert first.source_tree_hash == second.source_tree_hash == third.source_tree_hash
    assert (
        first.canonical_decoder_source_boundary_hash
        == second.canonical_decoder_source_boundary_hash
        == third.canonical_decoder_source_boundary_hash
    )
    assert cdb._canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'

    tree = ast.parse(Path(cdb.__file__).read_text())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hash"
        for node in ast.walk(tree)
    )


def test_self_hash_exclusion_and_stale_hash_failures():
    (
        identity,
        source_boundary,
        replay_boundary,
        equivalence_policy,
        immutability_boundary,
        receipt,
    ) = _parts()

    _expect_error(
        "HASH_MISMATCH",
        replace,
        identity,
        decoder_name="canonical decoder baseline v2",
        canonical_decoder_identity_hash=identity.canonical_decoder_identity_hash,
    )
    _expect_error(
        "HASH_MISMATCH",
        replace,
        receipt,
        upstream_graph_universe_claim_boundary_receipt_hash=HEX_A,
        canonical_decoder_baseline_receipt_hash=receipt.canonical_decoder_baseline_receipt_hash,
    )

    aggregate_payload = cdb._baseline_receipt_payload(receipt)
    assert "canonical_decoder_baseline_receipt_hash" not in aggregate_payload
    assert (
        cdb._hash_payload(aggregate_payload)
        == receipt.canonical_decoder_baseline_receipt_hash
    )
    assert "canonical_decoder_source_boundary_hash" not in cdb._source_boundary_payload(
        source_boundary
    )
    assert (
        "canonical_decoder_replay_corpus_boundary_hash"
        not in cdb._replay_corpus_boundary_payload(replay_boundary)
    )
    assert (
        "canonical_decoder_equivalence_policy_hash"
        not in cdb._equivalence_policy_payload(equivalence_policy)
    )
    assert (
        "canonical_decoder_immutability_boundary_hash"
        not in cdb._immutability_boundary_payload(immutability_boundary)
    )


def test_child_before_aggregate_validation_rejects_corrupt_child():
    (
        identity,
        source_boundary,
        replay_boundary,
        equivalence_policy,
        immutability_boundary,
        _,
    ) = _parts()
    corrupt_child = object.__new__(cdb.CanonicalDecoderSourceFile)
    object.__setattr__(corrupt_child, "path", "src/qec/decoder/alpha.py")
    object.__setattr__(corrupt_child, "sha256", "A" * 64)
    object.__setattr__(corrupt_child, "source_role", "BASELINE_DECODER_SOURCE")

    corrupt_boundary = object.__new__(cdb.CanonicalDecoderSourceBoundary)
    object.__setattr__(corrupt_boundary, "decoder_root", cdb.DECODER_ROOT)
    object.__setattr__(corrupt_boundary, "source_boundary_mode", "SOURCE_HASH_BOUND")
    object.__setattr__(corrupt_boundary, "source_files", (corrupt_child,))
    object.__setattr__(corrupt_boundary, "source_file_count", 1)
    object.__setattr__(corrupt_boundary, "source_tree_hash", HEX_A)
    object.__setattr__(corrupt_boundary, "runtime_decoder_execution_allowed", False)
    object.__setattr__(corrupt_boundary, "decoder_import_allowed", False)
    object.__setattr__(corrupt_boundary, "mutation_allowed", False)
    object.__setattr__(
        corrupt_boundary, "canonical_decoder_source_boundary_hash", HEX_B
    )

    parent = object.__new__(cdb.CanonicalDecoderBaselineReceipt)
    values = {
        "receipt_version": "v166.0",
        "receipt_kind": "CanonicalDecoderBaselineReceipt",
        "upstream_graph_universe_claim_boundary_receipt_hash": HEX_F,
        "identity": identity,
        "source_boundary": corrupt_boundary,
        "replay_corpus_boundary": replay_boundary,
        "equivalence_policy": equivalence_policy,
        "immutability_boundary": immutability_boundary,
        "replay_safe_canonical_decoder_baseline": True,
    }
    for key, value in values.items():
        object.__setattr__(parent, key, value)
    object.__setattr__(
        parent, "canonical_decoder_baseline_receipt_hash", cdb._hash_payload(values)
    )

    _expect_error(
        "INVALID_HASH", cdb.validate_canonical_decoder_baseline_receipt, parent
    )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"baseline_release": "v165.9.4"}, "INVALID_DECODER_BASELINE"),
        ({"previous_release_tag": "v165.9.3"}, "INVALID_DECODER_BASELINE"),
        ({"decoder_root": "src/qec/not_decoder/"}, "INVALID_DECODER_BASELINE"),
        ({"canonical_baseline": False}, "INVALID_DECODER_BASELINE"),
        ({"adapter_only": True}, "INVALID_DECODER_BASELINE"),
        ({"canonical_baseline": 1}, "INVALID_INPUT"),
        ({"adapter_only": 0}, "INVALID_INPUT"),
    ],
)
def test_identity_validation(kwargs, error):
    _expect_error(
        error,
        cdb.build_canonical_decoder_identity,
        "canonical decoder baseline",
        **kwargs,
    )


@pytest.mark.parametrize("bad_hash", ["A" * 64, "a" * 63, "g" * 64])
def test_source_file_rejects_invalid_hashes(bad_hash):
    _expect_error(
        "INVALID_HASH",
        cdb.build_canonical_decoder_source_file,
        "src/qec/decoder/a.py",
        bad_hash,
    )


@pytest.mark.parametrize(
    "path",
    [
        "/src/qec/decoder/a.py",
        "src/qec/decoder/../a.py",
        "src/qec/decoder/./a.py",
        "src/qec/decoder//a.py",
        "src/qec/decoder/..\\other.py",
        "src/qec/decoder/a\\b.py",
        "src/qec/other/a.py",
        "src/qec/decoder/",
    ],
)
def test_source_file_rejects_invalid_paths(path):
    error = _expect_error(
        "INVALID_INPUT", cdb.build_canonical_decoder_source_file, path, HEX_A
    )
    assert isinstance(error, cdb.CanonicalDecoderBaselineError)
    assert error.detail.startswith("decoder_path:")


def test_source_boundary_validation_failures_and_recomputations():
    sf = cdb.build_canonical_decoder_source_file("src/qec/decoder/a.py", HEX_A)
    _expect_error("INVALID_INPUT", cdb.build_canonical_decoder_source_boundary, ())
    _expect_error(
        "INVALID_INPUT", cdb.build_canonical_decoder_source_boundary, (sf, sf)
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_source_boundary,
        (sf,),
        runtime_decoder_execution_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_source_boundary,
        (sf,),
        decoder_import_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_source_boundary,
        (sf,),
        mutation_allowed=True,
    )

    boundary = cdb.build_canonical_decoder_source_boundary((sf,))
    _expect_error(
        "INVALID_INPUT",
        replace,
        boundary,
        source_file_count=2,
        canonical_decoder_source_boundary_hash=boundary.canonical_decoder_source_boundary_hash,
    )
    _expect_error("HASH_MISMATCH", replace, boundary, source_tree_hash=HEX_B)


def test_replay_corpus_boundary_validation_failures():
    for field in ("corpus_hash", "input_schema_hash", "output_schema_hash"):
        kwargs = {
            "corpus_name": "declared corpus",
            "corpus_version": "v1",
            "corpus_hash": HEX_A,
            "input_schema_hash": HEX_B,
            "output_schema_hash": HEX_C,
            field: "A" * 64,
        }
        _expect_error(
            "INVALID_HASH", cdb.build_canonical_decoder_replay_corpus_boundary, **kwargs
        )
    _expect_error(
        "INVALID_INPUT",
        cdb.build_canonical_decoder_replay_corpus_boundary,
        "declared corpus",
        "v1",
        HEX_A,
        HEX_B,
        HEX_C,
        corpus_mode="CUSTOM",
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_replay_corpus_boundary,
        "declared corpus",
        "v1",
        HEX_A,
        HEX_B,
        HEX_C,
        runtime_decoder_execution_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_replay_corpus_boundary,
        "declared corpus",
        "v1",
        HEX_A,
        HEX_B,
        HEX_C,
        candidate_replay_required_before_promotion=False,
    )
    _expect_error(
        "INVALID_INPUT",
        cdb.build_canonical_decoder_replay_corpus_boundary,
        "declared corpus",
        "v1",
        HEX_A,
        HEX_B,
        HEX_C,
        syndrome_ordering_policy="",
    )
    _expect_error(
        "INVALID_INPUT",
        cdb.build_canonical_decoder_replay_corpus_boundary,
        "declared corpus",
        "v1",
        HEX_A,
        HEX_B,
        HEX_C,
        syndrome_ordering_policy="RANDOM_ORDER",
    )


def test_equivalence_policy_validation_failures():
    cdb.validate_canonical_decoder_equivalence_policy(
        cdb.build_canonical_decoder_equivalence_policy()
    )
    _expect_error(
        "INVALID_INPUT",
        cdb.build_canonical_decoder_equivalence_policy,
        equivalence_mode="APPROXIMATE_OUTPUT_MATCH",
    )
    _expect_error(
        "INVALID_INPUT",
        cdb.build_canonical_decoder_equivalence_policy,
        equivalence_mode="PROBABILISTIC_OUTPUT_MATCH",
    )
    _expect_error(
        "INVALID_INPUT",
        cdb.build_canonical_decoder_equivalence_policy,
        precision_policy="hidden precision drift",
    )
    _expect_error(
        "INVALID_INPUT",
        cdb.build_canonical_decoder_equivalence_policy,
        approximation_policy="undeclared approximation policy",
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_equivalence_policy,
        benchmark_claims_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_equivalence_policy,
        hardware_authority_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_equivalence_policy,
        probabilistic_promotion_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_equivalence_policy,
        silent_replacement_allowed=True,
    )


def test_aggregate_rejects_missing_immutability_coverage_for_source_files():
    identity, source_boundary, replay_boundary, equivalence_policy, _, _ = _parts()
    incomplete_immutability_boundary = (
        cdb.build_canonical_decoder_immutability_boundary(("src/qec/decoder/alpha.py",))
    )

    error = _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_baseline_receipt,
        HEX_F,
        identity,
        source_boundary,
        replay_boundary,
        equivalence_policy,
        incomplete_immutability_boundary,
    )
    assert isinstance(error, cdb.CanonicalDecoderBaselineError)
    assert error.detail == "immutability_boundary:SOURCE_FILE_COVERAGE"


def test_immutability_boundary_validation_failures():
    _expect_error(
        "INVALID_INPUT", cdb.build_canonical_decoder_immutability_boundary, ()
    )
    _expect_error(
        "INVALID_INPUT",
        cdb.build_canonical_decoder_immutability_boundary,
        ("src/qec/decoder/a.py", "src/qec/decoder/a.py"),
    )
    _expect_error(
        "INVALID_INPUT",
        cdb.build_canonical_decoder_immutability_boundary,
        ("src/qec/other/a.py",),
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_immutability_boundary,
        ("src/qec/decoder/a.py",),
        mutation_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_immutability_boundary,
        ("src/qec/decoder/a.py",),
        silent_replacement_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_immutability_boundary,
        ("src/qec/decoder/a.py",),
        candidate_implementation_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_immutability_boundary,
        ("src/qec/decoder/a.py",),
        runtime_promotion_allowed=True,
    )
    _expect_error(
        "INVALID_DECODER_BASELINE",
        cdb.build_canonical_decoder_immutability_boundary,
        ("src/qec/decoder/a.py",),
        rollback_required_for_future_promotion=False,
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "silent_decoder_replacement",
        "decoder-replaced-because-faster",
        "benchmark proves correctness",
        "hardware authority",
        "QEC advantage proven",
        "hidden precision drift",
        "undeclared approximation policy",
        "silent\\n decoder___replacement",
        "decoder\nreplaced    because---faster",
    ],
)
def test_forbidden_semantic_hardening(phrase):
    _expect_error("INVALID_INPUT", cdb.build_canonical_decoder_identity, phrase)


def _imports_from(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_boundary_no_decoder_import_heavy_imports_network_benchmark_or_runtime_decoder_execution():
    module_path = Path(cdb.__file__)
    test_path = Path(__file__)
    module_imports = set(_imports_from(module_path))
    test_imports = set(_imports_from(test_path))

    assert not any(
        name == "qec.decoder" or name.startswith("qec.decoder.")
        for name in module_imports
    )
    assert not any(
        name == "qec.decoder" or name.startswith("qec.decoder.")
        for name in test_imports
    )
    assert module_imports.isdisjoint(
        {
            "numpy",
            "scipy",
            "qldpc",
            "stim",
            "pymatching",
            "qiskit",
            "requests",
            "urllib",
            "socket",
            "time",
            "datetime",
            "random",
        }
    )

    text = module_path.read_text().lower()
    assert "execute_decoder" not in text
    assert "run_decoder" not in text
    assert "decoder.decode" not in text
    assert "benchmark_loop" not in text
    assert "timeit" not in text
    assert "perf_counter" not in text


def test_hash_seed_stability_subprocesses():
    script = """
from qec.analysis import canonical_decoder_baseline_receipts as c
h='a'*64
i=c.build_canonical_decoder_identity('canonical decoder baseline')
s=c.build_canonical_decoder_source_boundary((c.build_canonical_decoder_source_file('src/qec/decoder/a.py',h),))
r=c.build_canonical_decoder_replay_corpus_boundary('declared corpus','v1','b'*64,'c'*64,'d'*64)
e=c.build_canonical_decoder_equivalence_policy()
im=c.build_canonical_decoder_immutability_boundary(('src/qec/decoder/a.py',))
print(c.build_canonical_decoder_baseline_receipt('e'*64,i,s,r,e,im).canonical_decoder_baseline_receipt_hash)
"""
    outputs = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONPATH="src", PYTHONHASHSEED=seed)
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=Path(__file__).parents[1],
                env=env,
                text=True,
            ).strip()
        )
    assert outputs[0] == outputs[1]
    assert cdb._HASH_RE.fullmatch(outputs[0])
