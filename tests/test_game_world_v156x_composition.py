from dataclasses import replace
from pathlib import Path
import zipfile

import pytest

from qec.analysis.game_world_intake_contract import build_game_world_corpus_manifest, build_game_world_intake_receipt
from qec.analysis.game_world_adapter_contract import build_world_adapter_contract_receipt
from qec.analysis.game_world_observation_snapshot import (
    build_observation_channel_spec,
    build_observation_snapshot_receipt,
    validate_observation_snapshot_receipt_with_adapter,
)
from qec.analysis.game_world_episode_trace import (
    build_episode_step,
    build_episode_trace,
    build_episode_trace_receipt,
    validate_episode_trace_receipt_with_adapter,
)
from qec.analysis.game_world_strategy_probe import (
    build_strategy_probe_receipt,
    validate_strategy_probe_receipt_with_adapter,
    validate_strategy_probe_result_with_adapter,
)


def _mkzip(tmp_path: Path, name: str) -> str:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("main.py", "print(1)")
    return str(p)


def _build_chain(tmp_path: Path, zip_name: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = build_game_world_corpus_manifest([_mkzip(tmp_path, zip_name)])
    intake = build_game_world_intake_receipt(manifest, "a" * 64)
    contract = build_world_adapter_contract_receipt(manifest, intake)
    spec = contract.adapter_specs[0]

    text = build_observation_snapshot_receipt(contract, spec, build_observation_channel_spec("TEXT_EVENT"), 0, "evt")
    term = build_observation_snapshot_receipt(contract, spec, build_observation_channel_spec("TERMINAL_FLAG"), 1, True)
    mask_payload = [a.action_code for a in spec.action_alphabet.actions]
    mask = build_observation_snapshot_receipt(contract, spec, build_observation_channel_spec("ACTION_MASK"), 2, mask_payload)

    s0 = build_episode_step(contract, spec, 0, text, spec.action_alphabet.actions[0], False)
    s1 = build_episode_step(contract, spec, 1, term, spec.action_alphabet.actions[min(1, len(spec.action_alphabet.actions) - 1)], True)
    trace = build_episode_trace(contract, spec, [s0, s1])
    trace_receipt = build_episode_trace_receipt(contract, spec, trace)

    probes = {
        "NO_OP_BASELINE": build_strategy_probe_receipt(contract, spec, trace_receipt, "NO_OP_BASELINE", "NO_OP_BASELINE"),
        "LEGAL_ACTION_SCAN": build_strategy_probe_receipt(contract, spec, trace_receipt, "LEGAL_ACTION_SCAN", "LEGAL_ACTION_SCAN"),
        "TERMINAL_STATE_SCAN": build_strategy_probe_receipt(contract, spec, trace_receipt, "TERMINAL_STATE_SCAN", "TERMINAL_STATE_SCAN"),
        "UNKNOWN_ACTION_REJECTION": build_strategy_probe_receipt(contract, spec, trace_receipt, "UNKNOWN_ACTION_REJECTION", "UNKNOWN_ACTION_REJECTION", {"action_code": spec.action_alphabet.actions[0].action_code}),
    }

    return {
        "manifest": manifest,
        "intake": intake,
        "contract": contract,
        "spec": spec,
        "observations": (text, term, mask),
        "trace_receipt": trace_receipt,
        "probes": probes,
    }


def test_v156x_full_chain_deterministic_build(tmp_path):
    a = _build_chain(tmp_path / "a", "doom.zip")
    b = _build_chain(tmp_path / "a", "doom.zip")

    assert a["intake"].receipt_hash == b["intake"].receipt_hash
    assert a["contract"].adapter_contract_receipt_hash == b["contract"].adapter_contract_receipt_hash
    assert a["observations"][0].observation_snapshot_receipt_hash == b["observations"][0].observation_snapshot_receipt_hash
    assert a["trace_receipt"].episode_trace_receipt_hash == b["trace_receipt"].episode_trace_receipt_hash
    assert a["probes"]["NO_OP_BASELINE"].strategy_probe_receipt_hash == b["probes"]["NO_OP_BASELINE"].strategy_probe_receipt_hash

    assert a["trace_receipt"].to_canonical_json() == b["trace_receipt"].to_canonical_json()
    assert a["trace_receipt"].to_canonical_bytes() == b["trace_receipt"].to_canonical_bytes()


def test_v156x_cross_layer_adapter_binding(tmp_path):
    ok = _build_chain(tmp_path / "ok", "doom.zip")
    other = _build_chain(tmp_path / "other", "atari.zip")

    for obs in ok["observations"]:
        assert validate_observation_snapshot_receipt_with_adapter(obs, ok["contract"], ok["spec"])
    assert validate_episode_trace_receipt_with_adapter(ok["trace_receipt"], ok["contract"], ok["spec"])
    for probe in ok["probes"].values():
        assert validate_strategy_probe_receipt_with_adapter(probe, ok["contract"], ok["spec"], ok["trace_receipt"])

    with pytest.raises(ValueError, match="MISMATCH|ADAPTER"):
        validate_episode_trace_receipt_with_adapter(ok["trace_receipt"], other["contract"], other["spec"])
    with pytest.raises(ValueError, match="MISMATCH|ADAPTER"):
        validate_strategy_probe_receipt_with_adapter(ok["probes"]["NO_OP_BASELINE"], other["contract"], other["spec"], ok["trace_receipt"])


def test_v156x_probe_result_rederivation_rejects_tamper(tmp_path):
    chain = _build_chain(tmp_path / "x", "doom.zip")
    probe = chain["probes"]["NO_OP_BASELINE"]

    tampered_payload = replace(probe.strategy_probe_result)
    object.__setattr__(tampered_payload, "canonical_result_payload", '{"tampered":true}')
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_strategy_probe_result_with_adapter(
            tampered_payload,
            probe.strategy_probe_spec,
            chain["contract"],
            chain["spec"],
            chain["trace_receipt"],
        )

    tampered_spec_hash = replace(probe.strategy_probe_result)
    object.__setattr__(tampered_spec_hash, "strategy_probe_spec_hash", "a" * 64)
    with pytest.raises(ValueError, match="MISMATCH|HASH"):
        validate_strategy_probe_result_with_adapter(
            tampered_spec_hash,
            probe.strategy_probe_spec,
            chain["contract"],
            chain["spec"],
            chain["trace_receipt"],
        )
