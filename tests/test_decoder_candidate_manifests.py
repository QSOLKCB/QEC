from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import decoder_candidate_manifests as dcm

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


def _expect_error(code: str, fn, *args, **kwargs):
    with pytest.raises(dcm.DecoderCandidateManifestError) as exc:
        fn(*args, **kwargs)
    assert exc.value.code.value == code
    assert code in str(exc.value)
    assert exc.value.detail
    return exc.value


def _parts(
    name: str = "candidate alpha", version: str = "0.1.0", upstream: str = HEX_A
):
    identity = dcm.build_decoder_candidate_identity(name, version, upstream)
    source_files = (
        dcm.build_decoder_candidate_source_file("candidate_decoders/b.py", HEX_B),
        dcm.build_decoder_candidate_source_file("candidate_decoders/a.py", HEX_A),
    )
    source_boundary = dcm.build_decoder_candidate_source_boundary(source_files)
    capability_declaration = dcm.build_decoder_candidate_capability_declaration(
        ("FAST_PATH_HYPOTHESIS", "GRAPH_CONSTRUCTION_HYPOTHESIS")
    )
    runtime_boundary = dcm.build_decoder_candidate_runtime_boundary()
    equivalence_precondition = dcm.build_decoder_candidate_equivalence_precondition(
        upstream
    )
    promotion_boundary = dcm.build_decoder_candidate_promotion_boundary()
    declaration = dcm.build_decoder_candidate_declaration(
        identity,
        source_boundary,
        capability_declaration,
        runtime_boundary,
        equivalence_precondition,
        promotion_boundary,
    )
    manifest = dcm.build_decoder_candidate_manifest(upstream, (declaration,))
    return (
        identity,
        source_boundary,
        capability_declaration,
        runtime_boundary,
        equivalence_precondition,
        promotion_boundary,
        declaration,
        manifest,
    )


def _unsafe_copy(obj, **updates):
    clone = object.__new__(type(obj))
    values = dict(obj.__dict__)
    values.update(updates)
    for key, value in values.items():
        object.__setattr__(clone, key, value)
    return clone


def _rehash_clone(obj, payload_fn_name: str, hash_field: str, **updates):
    clone = _unsafe_copy(obj, **updates)
    payload = getattr(dcm, payload_fn_name)(clone)
    object.__setattr__(clone, hash_field, dcm._hash_payload(payload))
    return clone


def test_happy_path_validates_hashes_and_frozen_artifacts():
    parts = _parts()
    validators = (
        dcm.validate_decoder_candidate_identity,
        dcm.validate_decoder_candidate_source_boundary,
        dcm.validate_decoder_candidate_capability_declaration,
        dcm.validate_decoder_candidate_runtime_boundary,
        dcm.validate_decoder_candidate_equivalence_precondition,
        dcm.validate_decoder_candidate_promotion_boundary,
        dcm.validate_decoder_candidate_declaration,
        dcm.validate_decoder_candidate_manifest,
    )
    for validator, artifact in zip(validators, parts):
        validator(artifact)
    for source_file in parts[1].source_files:
        dcm.validate_decoder_candidate_source_file(source_file)

    manifest = parts[-1]
    assert dcm._HASH_RE.fullmatch(manifest.decoder_candidate_manifest_hash)
    assert manifest.replay_safe_decoder_candidate_manifest is True
    assert manifest.all_candidates_adapter_only is True
    with pytest.raises(FrozenInstanceError):
        manifest.manifest_kind = "changed"


def test_canonical_json_hash_determinism_and_hash_seed_independence():
    f1 = dcm.build_decoder_candidate_source_file("candidate_decoders/a.py", HEX_A)
    f2 = dcm.build_decoder_candidate_source_file("candidate_decoders/b.py", HEX_B)
    first = dcm.build_decoder_candidate_source_boundary((f2, f1))
    second = dcm.build_decoder_candidate_source_boundary((f1, f2))
    third = dcm.build_decoder_candidate_source_boundary(
        list({f2.path: f2, f1.path: f1}.values())
    )
    assert first.source_files == (f1, f2)
    assert first.source_tree_hash == second.source_tree_hash == third.source_tree_hash
    assert (
        first.decoder_candidate_source_boundary_hash
        == second.decoder_candidate_source_boundary_hash
        == third.decoder_candidate_source_boundary_hash
    )

    caps_a = dcm.build_decoder_candidate_capability_declaration(
        ("FAST_PATH_HYPOTHESIS", "GRAPH_CONSTRUCTION_HYPOTHESIS")
    )
    caps_b = dcm.build_decoder_candidate_capability_declaration(
        ("GRAPH_CONSTRUCTION_HYPOTHESIS", "FAST_PATH_HYPOTHESIS")
    )
    assert caps_a.declared_capabilities == caps_b.declared_capabilities
    assert (
        caps_a.decoder_candidate_capability_declaration_hash
        == caps_b.decoder_candidate_capability_declaration_hash
    )

    decl_a = _parts("candidate beta", "0.1.0")[-2]
    decl_b = _parts("candidate alpha", "0.1.0")[-2]
    m1 = dcm.build_decoder_candidate_manifest(HEX_A, (decl_a, decl_b))
    m2 = dcm.build_decoder_candidate_manifest(HEX_A, (decl_b, decl_a))
    assert [d.identity.candidate_name for d in m1.candidate_declarations] == [
        "candidate alpha",
        "candidate beta",
    ]
    assert m1.decoder_candidate_manifest_hash == m2.decoder_candidate_manifest_hash
    assert dcm._canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'

    tree = ast.parse(Path(dcm.__file__).read_text())
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
        capability,
        runtime,
        equiv,
        promotion,
        declaration,
        manifest,
    ) = _parts()
    _expect_error(
        "HASH_MISMATCH",
        replace,
        identity,
        candidate_name="candidate changed",
        decoder_candidate_identity_hash=identity.decoder_candidate_identity_hash,
    )
    _expect_error(
        "HASH_MISMATCH", replace, declaration, decoder_candidate_declaration_hash=HEX_A
    )
    _expect_error(
        "HASH_MISMATCH", replace, manifest, decoder_candidate_manifest_hash=HEX_A
    )

    assert "decoder_candidate_manifest_hash" not in dcm._manifest_payload(manifest)
    assert (
        dcm._hash_payload(dcm._manifest_payload(manifest))
        == manifest.decoder_candidate_manifest_hash
    )
    assert "decoder_candidate_declaration_hash" not in dcm._declaration_payload(
        declaration
    )
    assert "decoder_candidate_identity_hash" not in dcm._identity_payload(identity)
    assert "decoder_candidate_source_boundary_hash" not in dcm._source_boundary_payload(
        source_boundary
    )
    assert (
        "decoder_candidate_capability_declaration_hash"
        not in dcm._capability_declaration_payload(capability)
    )
    assert (
        "decoder_candidate_runtime_boundary_hash"
        not in dcm._runtime_boundary_payload(runtime)
    )
    assert (
        "decoder_candidate_equivalence_precondition_hash"
        not in dcm._equivalence_precondition_payload(equiv)
    )
    assert (
        "decoder_candidate_promotion_boundary_hash"
        not in dcm._promotion_boundary_payload(promotion)
    )


def test_child_before_aggregate_validation_rejects_corrupt_children_even_with_forged_parent_hashes():
    identity, source_boundary, capability, runtime, equiv, promotion, declaration, _ = (
        _parts()
    )
    corrupt_runtime = _unsafe_copy(runtime, baseline_decoder_import_allowed=True)
    forged_declaration = _rehash_clone(
        declaration,
        "_declaration_payload",
        "decoder_candidate_declaration_hash",
        runtime_boundary=corrupt_runtime,
    )
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        dcm.validate_decoder_candidate_declaration,
        forged_declaration,
    )

    corrupt_declaration = _unsafe_copy(declaration, replay_safe_candidate_declaration=1)
    forged_manifest = _rehash_clone(
        dcm.build_decoder_candidate_manifest(HEX_A, (declaration,)),
        "_manifest_payload",
        "decoder_candidate_manifest_hash",
        candidate_declarations=(corrupt_declaration,),
    )
    _expect_error(
        "INVALID_INPUT", dcm.validate_decoder_candidate_manifest, forged_manifest
    )


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("candidate_release", "v166.2", "INVALID_DECODER_CANDIDATE"),
        ("previous_release_tag", "v165.9", "INVALID_DECODER_CANDIDATE"),
        ("previous_release_url", "https://example.invalid", "INVALID_INPUT"),
        ("upstream_canonical_decoder_baseline_receipt_hash", "A" * 64, "INVALID_HASH"),
        ("candidate_status", "PROMOTED", "INVALID_DECODER_CANDIDATE"),
        ("adapter_only", False, "INVALID_DECODER_CANDIDATE"),
        ("promoted", True, "INVALID_DECODER_CANDIDATE"),
        ("canonical_baseline_replacement", True, "INVALID_DECODER_CANDIDATE"),
        ("runtime_authority_allowed", True, "INVALID_DECODER_CANDIDATE"),
        ("adapter_only", 1, "INVALID_INPUT"),
        ("candidate_kind", "UNKNOWN", "INVALID_INPUT"),
    ],
)
def test_identity_validation_rejections(field, value, code):
    kwargs = {
        "candidate_name": "candidate",
        "candidate_version": "0",
        "upstream_canonical_decoder_baseline_receipt_hash": HEX_A,
    }
    kwargs[field] = value
    _expect_error(code, dcm.build_decoder_candidate_identity, **kwargs)


@pytest.mark.parametrize(
    "bad_path",
    [
        "/candidate_decoders/a.py",
        "candidate_decoders/./a.py",
        "candidate_decoders/a/../b.py",
        "candidate_decoders//a.py",
        "\\candidate_decoders\\a.py",
        "src/qec/decoder/a.py",
        "src/qec/decoder/./a.py",
        "src/qec/decoder//a.py",
        "src/qec/decoder/..\\other.py",
    ],
)
def test_source_file_rejects_invalid_paths(bad_path):
    _expect_error(
        "INVALID_INPUT", dcm.build_decoder_candidate_source_file, bad_path, HEX_A
    )


def test_source_boundary_validation_rejections_and_recomputed_fields():
    f1 = dcm.build_decoder_candidate_source_file("candidate_decoders/a.py", HEX_A)
    f2 = dcm.build_decoder_candidate_source_file("candidate_decoders/b.py", HEX_B)
    _expect_error("INVALID_INPUT", dcm.build_decoder_candidate_source_boundary, ())
    _expect_error(
        "INVALID_INPUT", dcm.build_decoder_candidate_source_boundary, (f1, f1)
    )
    _expect_error(
        "INVALID_INPUT",
        dcm.build_decoder_candidate_source_boundary,
        (f1,),
        candidate_source_root="external/decoder_candidates/",
    )
    for sha in ("A" * 64, "a" * 63, "g" * 64):
        _expect_error(
            "INVALID_HASH",
            dcm.build_decoder_candidate_source_file,
            "candidate_decoders/x.py",
            sha,
        )
    boundary = dcm.build_decoder_candidate_source_boundary((f2, f1))
    _expect_error(
        "INVALID_INPUT",
        replace,
        boundary,
        source_file_count=3,
        decoder_candidate_source_boundary_hash=boundary.decoder_candidate_source_boundary_hash,
    )
    _expect_error(
        "HASH_MISMATCH",
        replace,
        boundary,
        source_tree_hash=HEX_C,
        decoder_candidate_source_boundary_hash=boundary.decoder_candidate_source_boundary_hash,
    )
    for flag in (
        "candidate_import_allowed",
        "candidate_runtime_execution_allowed",
        "baseline_decoder_mutation_allowed",
        "filesystem_mutation_allowed",
    ):
        _expect_error(
            "INVALID_DECODER_CANDIDATE",
            dcm.build_decoder_candidate_source_boundary,
            (f1,),
            **{flag: True},
        )


@pytest.mark.parametrize(
    "flag",
    [
        "performance_claim_allowed",
        "correctness_claim_allowed",
        "benchmark_claim_allowed",
        "qec_advantage_claim_allowed",
        "hardware_authority_allowed",
        "exact_equivalence_claimed",
    ],
)
def test_capability_declaration_validation_rejections(flag):
    _expect_error(
        "INVALID_INPUT", dcm.build_decoder_candidate_capability_declaration, ()
    )
    _expect_error(
        "INVALID_INPUT",
        dcm.build_decoder_candidate_capability_declaration,
        ("FAST_PATH_HYPOTHESIS", "FAST_PATH_HYPOTHESIS"),
    )
    _expect_error(
        "INVALID_INPUT",
        dcm.build_decoder_candidate_capability_declaration,
        ("CUSTOM_CAPABILITY",),
    )
    _expect_error(
        "INVALID_INPUT",
        dcm.build_decoder_candidate_capability_declaration,
        ("speed proves correctness",),
    )
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        dcm.build_decoder_candidate_capability_declaration,
        ("FAST_PATH_HYPOTHESIS",),
        **{flag: True},
    )
    cap = dcm.build_decoder_candidate_capability_declaration(("FAST_PATH_HYPOTHESIS",))
    _expect_error(
        "INVALID_INPUT",
        replace,
        cap,
        capability_count=2,
        decoder_candidate_capability_declaration_hash=cap.decoder_candidate_capability_declaration_hash,
    )


@pytest.mark.parametrize(
    "field",
    [
        "baseline_decoder_import_allowed",
        "candidate_import_allowed",
        "decoder_workload_execution_allowed",
        "benchmark_execution_allowed",
        "network_allowed",
        "heavy_backend_import_allowed",
        "hardware_sdk_allowed",
    ],
)
def test_runtime_boundary_validation_rejections(field):
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        dcm.build_decoder_candidate_runtime_boundary,
        **{field: True},
    )
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        dcm.build_decoder_candidate_runtime_boundary,
        runtime_boundary_mode="RUN_DECODER",
    )


@pytest.mark.parametrize(
    "kwargs,code",
    [
        (
            {"upstream_canonical_decoder_baseline_receipt_hash": "A" * 64},
            "INVALID_HASH",
        ),
        ({"required_future_receipt_kind": "OtherReceipt"}, "INVALID_DECODER_CANDIDATE"),
        ({"required_future_release": "v166.1"}, "INVALID_DECODER_CANDIDATE"),
        ({"equivalence_required_before_promotion": False}, "INVALID_DECODER_CANDIDATE"),
        ({"equivalence_mode": "APPROXIMATE_MATCH"}, "INVALID_DECODER_CANDIDATE"),
        ({"equivalence_mode": "PROBABILISTIC_MATCH"}, "INVALID_DECODER_CANDIDATE"),
        ({"equivalence_mode": "CUSTOM"}, "INVALID_DECODER_CANDIDATE"),
        ({"replay_corpus_required": False}, "INVALID_DECODER_CANDIDATE"),
        ({"output_schema_match_required": False}, "INVALID_DECODER_CANDIDATE"),
        ({"precision_policy": "hidden precision drift"}, "INVALID_INPUT"),
        ({"approximation_policy": "undeclared approximation policy"}, "INVALID_INPUT"),
        ({"equivalence_proven": True}, "INVALID_DECODER_CANDIDATE"),
        ({"candidate_output_authority_allowed": True}, "INVALID_DECODER_CANDIDATE"),
        (
            {"candidate_status_until_equivalence": "PROMOTED"},
            "INVALID_DECODER_CANDIDATE",
        ),
    ],
)
def test_equivalence_precondition_validation(kwargs, code):
    params = {"upstream_canonical_decoder_baseline_receipt_hash": HEX_A}
    params.update(kwargs)
    _expect_error(code, dcm.build_decoder_candidate_equivalence_precondition, **params)
    dcm.validate_decoder_candidate_equivalence_precondition(
        dcm.build_decoder_candidate_equivalence_precondition(HEX_A)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"promotion_status": "PROMOTED"},
        {"promotion_allowed_in_this_release": True},
        {"runtime_authority_allowed": True},
        {"silent_replacement_allowed": True},
        {"probabilistic_promotion_allowed": True},
        {"ml_decoder_authority_allowed": True},
        {"benchmark_marketing_allowed": True},
        {"rollback_receipt_required_before_promotion": False},
        {"benchmark_ladder_required_before_performance_claims": False},
        {"baseline_mutation_allowed": True},
    ],
)
def test_promotion_boundary_validation_rejections(kwargs):
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        dcm.build_decoder_candidate_promotion_boundary,
        **kwargs,
    )


def test_candidate_declaration_validation_rejections():
    identity, source_boundary, capability, runtime, equiv, promotion, declaration, _ = (
        _parts()
    )
    equiv_b = dcm.build_decoder_candidate_equivalence_precondition(HEX_B)
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        dcm.build_decoder_candidate_declaration,
        identity,
        source_boundary,
        capability,
        runtime,
        equiv_b,
        promotion,
    )
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        replace,
        declaration,
        replay_safe_candidate_declaration=False,
        decoder_candidate_declaration_hash=declaration.decoder_candidate_declaration_hash,
    )
    unsafe_runtime = _unsafe_copy(runtime, decoder_workload_execution_allowed=True)
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        dcm.build_decoder_candidate_declaration,
        identity,
        source_boundary,
        capability,
        unsafe_runtime,
        equiv,
        promotion,
    )
    unsafe_promotion = _unsafe_copy(promotion, promotion_allowed_in_this_release=True)
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        dcm.build_decoder_candidate_declaration,
        identity,
        source_boundary,
        capability,
        runtime,
        equiv,
        unsafe_promotion,
    )


def test_manifest_validation_rejections_and_ordering():
    decl_a = _parts("candidate alpha", "0.1.0")[-2]
    decl_b = _parts("candidate beta", "0.1.0")[-2]
    _expect_error("INVALID_INPUT", dcm.build_decoder_candidate_manifest, HEX_A, ())
    _expect_error(
        "INVALID_INPUT", dcm.build_decoder_candidate_manifest, HEX_A, (decl_a, decl_a)
    )
    manifest = dcm.build_decoder_candidate_manifest(HEX_A, (decl_b, decl_a))
    assert manifest.candidate_count == 2
    assert [d.identity.candidate_name for d in manifest.candidate_declarations] == [
        "candidate alpha",
        "candidate beta",
    ]
    _expect_error(
        "INVALID_INPUT",
        replace,
        manifest,
        candidate_count=3,
        decoder_candidate_manifest_hash=manifest.decoder_candidate_manifest_hash,
    )
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        replace,
        manifest,
        all_candidates_adapter_only=False,
        decoder_candidate_manifest_hash=manifest.decoder_candidate_manifest_hash,
    )
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        replace,
        manifest,
        replay_safe_decoder_candidate_manifest=False,
        decoder_candidate_manifest_hash=manifest.decoder_candidate_manifest_hash,
    )
    decl_other_hash = _parts("candidate gamma", "0.1.0", HEX_B)[-2]
    _expect_error(
        "INVALID_DECODER_CANDIDATE",
        dcm.build_decoder_candidate_manifest,
        HEX_A,
        (decl_other_hash,),
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "silent_decoder_replacement",
        "candidate-replaces-baseline",
        "decoder replaced because faster",
        "speed proves correctness",
        "benchmark proves correctness",
        "benchmark marketing",
        "runtime promotion",
        "candidate decoder promoted",
        "probabilistic decoder authority",
        "ML decoder authority",
        "hardware authority",
        "QEC advantage proven",
        "hidden precision drift",
        "undeclared approximation policy",
        "equivalence already proven",
        "output accepted as canonical",
        "speed\\nproves___correctness",
        "candidate   replaces\n baseline",
    ],
)
def test_forbidden_semantic_hardening_blocks_phrase_variants(phrase):
    _expect_error(
        "INVALID_INPUT", dcm.build_decoder_candidate_identity, phrase, "0", HEX_A
    )


def test_boundary_static_inspection_and_decoder_immutability_contracts():
    module_path = Path(dcm.__file__)
    module_text = module_path.read_text()
    tree = ast.parse(module_text)
    forbidden_imports = {
        "qec.decoder",
        "numpy",
        "scipy",
        "qldpc",
        "stim",
        "pymatching",
        "qiskit",
        "requests",
        "urllib.request",
        "socket",
        "random",
        "time",
        "datetime",
    }
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.lower())
    assert not any(
        name == "qec.decoder" or name.startswith("qec.decoder.") for name in imports
    )
    assert not (imports & forbidden_imports)
    assert '"src/qec/decoder/"' not in module_text
    assert "src/qec/decoder/" not in dcm._ALLOWED_CANDIDATE_SOURCE_ROOTS
    _expect_error(
        "INVALID_INPUT",
        dcm.build_decoder_candidate_source_file,
        "src/qec/decoder/a.py",
        HEX_A,
    )

    test_tree = ast.parse(Path(__file__).read_text())
    assert not any(
        (
            isinstance(node, ast.Import)
            and any(
                alias.name == "qec.decoder" or alias.name.startswith("qec.decoder.")
                for alias in node.names
            )
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "qec.decoder" or node.module.startswith("qec.decoder."))
        )
        for node in ast.walk(test_tree)
    )
    assert not Path("candidate_decoders").exists()
    assert not Path("src/qec/analysis/decoder_candidates").exists()


def test_hash_seed_stability_subprocesses():
    script = """
from qec.analysis import decoder_candidate_manifests as d
h='a'*64
f1=d.build_decoder_candidate_source_file('candidate_decoders/b.py','b'*64)
f2=d.build_decoder_candidate_source_file('candidate_decoders/a.py','a'*64)
i=d.build_decoder_candidate_identity('candidate alpha','0.1.0',h)
s=d.build_decoder_candidate_source_boundary((f1,f2))
c=d.build_decoder_candidate_capability_declaration(('GRAPH_CONSTRUCTION_HYPOTHESIS','FAST_PATH_HYPOTHESIS'))
r=d.build_decoder_candidate_runtime_boundary()
e=d.build_decoder_candidate_equivalence_precondition(h)
p=d.build_decoder_candidate_promotion_boundary()
decl=d.build_decoder_candidate_declaration(i,s,c,r,e,p)
print(d.build_decoder_candidate_manifest(h,(decl,)).decoder_candidate_manifest_hash)
"""
    outputs = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONPATH="src", PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).parents[1],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]
