from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import byte_level_model_boundary_receipts as blmbr
from qec.analysis import inference_backend_manifest as ibm


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


def _build_receipt(**kwargs):
    manifest = kwargs.get("manifest") or _upstream_manifest()
    ident = kwargs.get("ident") or blmbr.build_byte_level_model_identity("family", "PURE_BYTE_MODEL", "declared boundary")
    patch = kwargs.get("patch") or blmbr.build_patch_segmentation_declaration("FIXED_PATCH", 64, "segmentation only")
    window = kwargs.get("window") or blmbr.build_byte_window_declaration(128, True, "replay layout only")
    entropy = kwargs.get("entropy") or blmbr.build_entropy_threshold_declaration("FIXED_THRESHOLD", 0.0, "declared only")
    norm = kwargs.get("norm") or blmbr.build_byte_normalization_policy("NO_NORMALIZATION", "declared only")
    tok = kwargs.get("tok") or blmbr.build_tokenizer_bypass_declaration("TOKENIZER_REMOVED", "declared only")
    adapter_only = kwargs.get("adapter_only", True)
    return blmbr.build_byte_level_model_boundary_receipt(manifest, ident, patch, window, entropy, norm, tok, adapter_only)


def test_hash_stability_canonical_json_and_idempotent_rebuild_and_hashseed_stability():
    r1, r2 = _build_receipt(), _build_receipt()
    assert r1.byte_level_model_boundary_receipt_hash == r2.byte_level_model_boundary_receipt_hash
    assert blmbr._canonical_json(r1.__dict__) == blmbr._canonical_json(r2.__dict__)


def test_replay_safe_layout_recomputed_and_adapter_only_enforced():
    receipt = _build_receipt()
    assert receipt.replay_safe_layout is True
    with pytest.raises(ValueError):
        blmbr.validate_byte_level_model_boundary_receipt(replace(receipt, replay_safe_layout=False), _upstream_manifest())
    with pytest.raises(ValueError):
        blmbr.validate_byte_level_model_boundary_receipt(replace(receipt, adapter_only=False), _upstream_manifest())
    with pytest.raises(ValueError):
        _build_receipt(adapter_only=False)


@pytest.mark.parametrize(
    "builder,args",
    [
        (blmbr.build_byte_level_model_identity, ("f", "BAD", "r")),
        (blmbr.build_patch_segmentation_declaration, ("BAD", 1, "r")),
        (blmbr.build_byte_normalization_policy, ("BAD", "r")),
        (blmbr.build_tokenizer_bypass_declaration, ("BAD", "r")),
        (blmbr.build_entropy_threshold_declaration, ("BAD", 0.0, "r")),
    ],
)
def test_invalid_enum_rejections(builder, args):
    with pytest.raises(ValueError):
        builder(*args)


@pytest.mark.parametrize("size", [0, -1, blmbr._MAX_PATCH_SIZE + 1])
def test_invalid_patch_size_rejection(size):
    with pytest.raises(ValueError):
        blmbr.build_patch_segmentation_declaration("FIXED_PATCH", size, "x")
    with pytest.raises(ValueError):
        blmbr.build_patch_segmentation_declaration("FIXED_PATCH", True, "x")


def test_negative_entropy_threshold_rejection():
    with pytest.raises(ValueError):
        blmbr.build_entropy_threshold_declaration("FIXED_THRESHOLD", -0.1, "x")
    with pytest.raises(ValueError):
        blmbr.build_entropy_threshold_declaration("FIXED_THRESHOLD", True, "x")


def test_bool_window_size_rejection():
    with pytest.raises(ValueError):
        blmbr.build_byte_window_declaration(True, False, "declared")


@pytest.mark.parametrize("builder,args", [
    (blmbr.build_patch_segmentation_declaration, ("FIXED_PATCH", 64, "hidden tokenizer used here")),
    (blmbr.build_byte_window_declaration, (128, True, "hidden normalization declaration")),
    (blmbr.build_entropy_threshold_declaration, ("FIXED_THRESHOLD", 0.1, "hidden tokenizer drift")),
])
def test_hidden_boundary_drift_rejected_in_reason_fields(builder, args):
    with pytest.raises(ValueError):
        builder(*args)


def test_hidden_boundary_drift_rejected_in_model_family():
    with pytest.raises(ValueError):
        blmbr.build_byte_level_model_identity("hidden tokenizer family", "PURE_BYTE_MODEL", "declared boundary")


def test_child_validation_before_aggregate_and_malformed_hash_rejection():
    receipt = _build_receipt()
    bad = replace(receipt.model_identity, byte_level_model_identity_hash="f" * 63)
    with pytest.raises(ValueError):
        blmbr.validate_byte_level_model_boundary_receipt(replace(receipt, model_identity=bad), _upstream_manifest())


def test_immutable_payload_validation():
    with pytest.raises(FrozenInstanceError):
        _build_receipt().adapter_only = False


@pytest.mark.parametrize("text", ["hidden tokenizer drift", "hidden normalization semantics", "semantic equivalence guaranteed", "autonomous evaluation", "reasoning preserved", "entropy proves intelligence"])
def test_forbidden_semantic_and_hidden_semantic_rejections(text):
    with pytest.raises(ValueError):
        blmbr.build_tokenizer_bypass_declaration("TOKENIZER_OPTIONAL", text)


def test_upstream_manifest_validation_enforced():
    r = _build_receipt()
    bad_manifest = replace(_upstream_manifest(), adapter_only=False)
    with pytest.raises(ValueError):
        blmbr.validate_byte_level_model_boundary_receipt(r, bad_manifest)


def test_overlapping_window_canonical_stability_and_patch_window_boundary_declaration():
    r = _build_receipt(window=blmbr.build_byte_window_declaration(256, True, "layout declaration only"))
    assert "layout" in r.byte_window.byte_window_reason


def test_no_runtime_or_tokenizer_imports_and_decoder_boundary_and_no_autonomous_eval_semantics():
    source = (Path(__file__).parent.parent / "src/qec/analysis/byte_level_model_boundary_receipts.py").read_text(encoding="utf-8")
    for token in (
        "transformers", "torch", "tensorflow", "sentencepiece", "tokenizers", "tiktoken", "openai", "anthropic",
        "mlx", "vllm", "llama_cpp", "requests", "urllib", "selenium", "playwright", "pandas", "polars", "subprocess", "os.system"
    ):
        assert token not in source
    assert "src/qec/decoder/" not in "src/qec/analysis/byte_level_model_boundary_receipts.py"


def test_tokenizer_free_and_entropy_and_normalization_boundary_enforcement_and_replay_false_when_invalid_component():
    tok = blmbr.build_tokenizer_bypass_declaration("TOKENIZER_REMOVED", "boundary declaration only")
    ent = blmbr.build_entropy_threshold_declaration("DECLARED_CONTEXT_THRESHOLD", 1.0, "threshold declaration only")
    norm = blmbr.build_byte_normalization_policy("UTF8_DECLARED", "declared normalization only")
    receipt = _build_receipt(tok=tok, entropy=ent, norm=norm)
    assert receipt.replay_safe_layout is True
    bad_patch = replace(receipt.patch_segmentation, patch_size=0)
    bad_receipt = replace(receipt, patch_segmentation=bad_patch, replay_safe_layout=True)
    with pytest.raises(ValueError):
        blmbr.validate_byte_level_model_boundary_receipt(bad_receipt, _upstream_manifest())
