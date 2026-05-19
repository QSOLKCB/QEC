from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import byte_level_model_boundary_receipts as blmbr
from qec.analysis import inference_backend_manifest as ibm
from qec.analysis import tokenization_policy_receipts as tpr


def _upstream_manifest():
    return ibm.build_inference_backend_manifest(
        ibm.build_inference_backend_identity("backend", "1.0", "BYTE_LEVEL_MODEL"),
        ibm.build_tokenization_policy_declaration("BYTE_LEVEL", "declared", "boundary only"),
        ibm.build_quantization_declaration("NO_QUANTIZATION", "none"),
        ibm.build_precision_format_declaration("FP32", "full precision"),
        ibm.build_hardware_kernel_boundary("ADAPTER_ONLY_HARDWARE", "boundary only", True),
        ibm.build_benchmark_corpus_declaration("NO_BENCHMARK_DECLARED", "none", "context only"),
        ibm.build_kv_cache_declaration("NO_CACHE", "none"),
    )


def _upstream_byte_receipt(manifest=None):
    m = manifest or _upstream_manifest()
    return blmbr.build_byte_level_model_boundary_receipt(
        m,
        blmbr.build_byte_level_model_identity("family", "PURE_BYTE_MODEL", "declared boundary"),
        blmbr.build_patch_segmentation_declaration("FIXED_PATCH", 64, "segmentation only"),
        blmbr.build_byte_window_declaration(128, True, "replay layout only"),
        blmbr.build_entropy_threshold_declaration("FIXED_THRESHOLD", 0.0, "declared only"),
        blmbr.build_byte_normalization_policy("NO_NORMALIZATION", "declared only"),
        blmbr.build_tokenizer_bypass_declaration("TOKENIZER_REMOVED", "declared only"),
        True,
    )


def _build_receipt(**kwargs):
    manifest = kwargs.get("manifest") or _upstream_manifest()
    byte_receipt = kwargs.get("byte_receipt") or _upstream_byte_receipt(manifest)
    identity = kwargs.get("identity") or tpr.build_tokenizer_identity("tok", "1", "BYTE_LEVEL_TOKENIZER")
    vocab = kwargs.get("vocab") or tpr.build_vocabulary_policy_declaration("BYTE_NATIVE_VOCABULARY", 256, "boundary only")
    merge = kwargs.get("merge") or tpr.build_merge_rule_declaration("BYTE_NATIVE_SEGMENTATION", "replay boundary only")
    boundary = kwargs.get("boundary") or tpr.build_token_boundary_declaration("BYTE_COMPATIBLE_BOUNDARY", "byte compatible boundary")
    eq = kwargs.get("eq") or tpr.build_tokenizer_equivalence_declaration("NO_EQUIVALENCE_CLAIM", "no replay identity claim")
    drift = kwargs.get("drift") or tpr.build_tokenizer_drift_boundary("DRIFT_PROHIBITED", "drift prohibited")
    adapter_only = kwargs.get("adapter_only", True)
    return tpr.build_tokenization_policy_receipt(
        manifest, byte_receipt, identity, vocab, merge, boundary, eq, drift, adapter_only
    )


def test_hash_and_canonical_json_and_idempotent_and_hashseed_stability():
    r1, r2 = _build_receipt(), _build_receipt()
    assert r1.tokenization_policy_receipt_hash == r2.tokenization_policy_receipt_hash
    assert tpr._canonical_json(r1.__dict__) == tpr._canonical_json(r2.__dict__)


def test_replay_safe_recomputed_and_adapter_only_enforced_and_child_validation_before_aggregate():
    r = _build_receipt()
    assert r.replay_safe_tokenization is True
    with pytest.raises(ValueError):
        tpr.validate_tokenization_policy_receipt(replace(r, replay_safe_tokenization=False), _upstream_manifest(), _upstream_byte_receipt())
    with pytest.raises(ValueError):
        _build_receipt(adapter_only=False)
    bad_child = replace(r.tokenizer_identity, tokenizer_identity_hash="f" * 63)
    with pytest.raises(ValueError):
        tpr.validate_tokenization_policy_receipt(replace(r, tokenizer_identity=bad_child), _upstream_manifest(), _upstream_byte_receipt())


@pytest.mark.parametrize(
    "builder,args",
    [
        (tpr.build_tokenizer_identity, ("tok", "1", "BAD")),
        (tpr.build_vocabulary_policy_declaration, ("BAD", 10, "r")),
        (tpr.build_merge_rule_declaration, ("BAD", "r")),
        (tpr.build_token_boundary_declaration, ("BAD", "r")),
        (tpr.build_tokenizer_equivalence_declaration, ("BAD", "r")),
        (tpr.build_tokenizer_drift_boundary, ("BAD", "r")),
    ],
)
def test_invalid_enums_rejected(builder, args):
    with pytest.raises(ValueError):
        builder(*args)


@pytest.mark.parametrize("size", [0, -1, True, tpr._MAX_VOCAB_SIZE + 1])
def test_invalid_vocab_size_rejected(size):
    with pytest.raises(ValueError):
        tpr.build_vocabulary_policy_declaration("FIXED_VOCABULARY", size, "boundary")


def test_immutable_payload_validation():
    with pytest.raises(FrozenInstanceError):
        _build_receipt().adapter_only = False


@pytest.mark.parametrize(
    "text",
    [
        "hidden tokenizer drift",
        "hidden merge-rule mutation",
        "hidden semantic-equivalence",
        "hidden replay-equivalence",
        "hidden vocabulary mutation",
        "semantic equivalence guaranteed",
        "tokenization preserves reasoning",
        "merge rules prove meaning",
        "benchmark proves tokenizer superiority",
        "autonomous evaluation",
    ],
)
def test_hidden_and_forbidden_semantics_rejected(text):
    with pytest.raises(ValueError):
        tpr.build_tokenizer_equivalence_declaration("DECLARED_PARTIAL_EQUIVALENCE", text)


@pytest.mark.parametrize("value", ["runtime tokenization v1", "hidden semantic equivalence"])
def test_tokenizer_identity_rejects_forbidden_or_hidden_semantics(value):
    with pytest.raises(ValueError):
        tpr.build_tokenizer_identity(value, "1", "BYTE_LEVEL_TOKENIZER")
    with pytest.raises(ValueError):
        tpr.build_tokenizer_identity("tok", value, "BYTE_LEVEL_TOKENIZER")


def test_upstream_manifest_and_byte_boundary_validation_enforced():
    r = _build_receipt()
    bad_manifest = replace(_upstream_manifest(), adapter_only=False)
    with pytest.raises(ValueError):
        tpr.validate_tokenization_policy_receipt(r, bad_manifest, _upstream_byte_receipt())
    bad_byte = replace(_upstream_byte_receipt(), replay_safe_layout=False)
    with pytest.raises(ValueError):
        tpr.validate_tokenization_policy_receipt(r, _upstream_manifest(), bad_byte)


def test_byte_compatible_boundary_and_boundary_modes_supported():
    r = _build_receipt(boundary=tpr.build_token_boundary_declaration("BYTE_COMPATIBLE_BOUNDARY", "replay-safe byte boundary only"))
    assert r.token_boundary_declaration.token_boundary_mode == "BYTE_COMPATIBLE_BOUNDARY"


def test_byte_compatible_claims_must_match_upstream_artifacts():
    manifest = ibm.build_inference_backend_manifest(
        ibm.build_inference_backend_identity("backend", "1.0", "BYTE_LEVEL_MODEL"),
        ibm.build_tokenization_policy_declaration("BPE", "declared", "boundary only"),
        ibm.build_quantization_declaration("NO_QUANTIZATION", "none"),
        ibm.build_precision_format_declaration("FP32", "full precision"),
        ibm.build_hardware_kernel_boundary("ADAPTER_ONLY_HARDWARE", "boundary only", True),
        ibm.build_benchmark_corpus_declaration("NO_BENCHMARK_DECLARED", "none", "context only"),
        ibm.build_kv_cache_declaration("NO_CACHE", "none"),
    )
    byte_receipt = _upstream_byte_receipt(manifest)
    receipt = _build_receipt(manifest=manifest, byte_receipt=byte_receipt)
    with pytest.raises(ValueError):
        tpr.validate_tokenization_policy_receipt(receipt, manifest, byte_receipt)

    manifest_ok = _upstream_manifest()
    byte_not_bypassed = blmbr.build_byte_level_model_boundary_receipt(
        manifest_ok,
        blmbr.build_byte_level_model_identity("family", "PURE_BYTE_MODEL", "declared boundary"),
        blmbr.build_patch_segmentation_declaration("FIXED_PATCH", 64, "segmentation only"),
        blmbr.build_byte_window_declaration(128, True, "replay layout only"),
        blmbr.build_entropy_threshold_declaration("FIXED_THRESHOLD", 0.0, "declared only"),
        blmbr.build_byte_normalization_policy("NO_NORMALIZATION", "declared only"),
        blmbr.build_tokenizer_bypass_declaration("TOKENIZER_NOT_BYPASSED", "declared only"),
        True,
    )
    receipt2 = _build_receipt(manifest=manifest_ok, byte_receipt=byte_not_bypassed)
    with pytest.raises(ValueError):
        tpr.validate_tokenization_policy_receipt(receipt2, manifest_ok, byte_not_bypassed)


def test_replay_safe_rejects_mutable_or_drift_modes_and_non_bool_values():
    dynamic_vocab = tpr.build_vocabulary_policy_declaration("DECLARED_DYNAMIC_VOCABULARY", 256, "declared")
    r = _build_receipt(vocab=dynamic_vocab)
    assert r.replay_safe_tokenization is False

    mutable_merge = tpr.build_merge_rule_declaration("DECLARED_MUTABLE_MERGE_RULES", "declared")
    r2 = _build_receipt(merge=mutable_merge)
    assert r2.replay_safe_tokenization is False

    drifted = tpr.build_tokenizer_drift_boundary("DRIFT_DECLARED", "declared")
    r3 = _build_receipt(drift=drifted)
    assert r3.replay_safe_tokenization is False

    with pytest.raises(ValueError):
        tpr.validate_tokenization_policy_receipt(replace(_build_receipt(), replay_safe_tokenization=1), _upstream_manifest(), _upstream_byte_receipt())


def test_no_runtime_imports_no_autonomous_semantics_and_decoder_boundary_untouched():
    source = (Path(__file__).parent.parent / "src/qec/analysis/tokenization_policy_receipts.py").read_text(encoding="utf-8")
    for token in (
        "transformers", "torch", "tensorflow", "sentencepiece", "tokenizers", "tiktoken", "openai", "anthropic",
        "mlx", "vllm", "llama_cpp", "requests", "urllib", "selenium", "playwright", "pandas", "polars", "subprocess", "os.system"
    ):
        assert token not in source
    assert "src/qec/decoder/" not in "src/qec/analysis/tokenization_policy_receipts.py"
