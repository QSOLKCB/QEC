from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from qec.analysis.phase_transition_forecast_engine import (
    ForecastLedger,
    ForecastLedgerEntry,
    MAX_HORIZON_STEPS,
    append_forecast_ledger_entry,
    compute_transition_probability_score,
    detect_bifurcation_early_warning,
    empty_forecast_ledger,
    forecast_attractor_boundary,
    normalize_forecast_inputs,
    run_phase_transition_forecast_engine,
    validate_forecast_ledger,
)


def _base_input() -> tuple[str, float, float, float, float]:
    return ("macro_A", 0.8, 0.3, 0.4, 0.7)


def test_normalize_forecast_inputs_is_deterministic() -> None:
    state_id, prior_stability, drift, pressure, boundary_distance = _base_input()
    a = normalize_forecast_inputs(state_id, prior_stability, drift, pressure, boundary_distance)
    b = normalize_forecast_inputs(state_id, prior_stability, drift, pressure, boundary_distance)
    assert a.to_canonical_json() == b.to_canonical_json()


def test_probability_score_is_bounded() -> None:
    inp = normalize_forecast_inputs("bounded", 1.5, 3.2, 2.9, 0.0)
    forecast = compute_transition_probability_score(inp)
    assert 0.0 <= forecast.probability_score <= 1.0


def test_probability_score_rejects_nan_inf() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="must be finite"):
            normalize_forecast_inputs("bad", 0.5, bad, 0.3, 0.2)
        with pytest.raises(ValueError, match="must be finite"):
            normalize_forecast_inputs("bad", bad, 0.1, 0.3, 0.2)


def test_bifurcation_warning_thresholds_are_stable() -> None:
    # Inputs chosen so that the computed probability spans all four escalation levels.
    levels = []
    for score in (0.1, 0.3, 0.7, 0.9):
        inp = normalize_forecast_inputs("threshold", 0.5, score, score, 1.0 - score)
        report = detect_bifurcation_early_warning(compute_transition_probability_score(inp))
        levels.append(report.escalation_level)
    assert levels[0] == "none"
    assert levels[1] == "observe"
    assert levels[2] == "warning"
    assert levels[3] == "critical"


def test_boundary_forecast_is_deterministic() -> None:
    inp = normalize_forecast_inputs("boundary", 0.6, 0.2, 0.3, 0.5)
    a = forecast_attractor_boundary(inp)
    b = forecast_attractor_boundary(inp)
    assert a.to_canonical_json() == b.to_canonical_json()


def test_horizon_is_stable() -> None:
    inp = normalize_forecast_inputs("stable_state", 0.95, 0.05, 0.05, 0.9)
    _, _, _, horizon, _, _ = run_phase_transition_forecast_engine(inp)
    assert horizon.stable
    assert horizon.forecast_state == "stable_state"


def test_forecast_ledger_chain_is_stable() -> None:
    inp = normalize_forecast_inputs("ledger", 0.8, 0.4, 0.4, 0.6)
    *_, ledger_a = run_phase_transition_forecast_engine(inp)
    *_, ledger_b = run_phase_transition_forecast_engine(inp)
    assert ledger_a.to_canonical_json() == ledger_b.to_canonical_json()
    assert validate_forecast_ledger(ledger_a)


def test_forecast_ledger_detects_corruption() -> None:
    inp = normalize_forecast_inputs("corrupt", 0.8, 0.4, 0.4, 0.6)
    *_, ledger = run_phase_transition_forecast_engine(inp)
    bad_entry = ForecastLedgerEntry(
        sequence_id=0,
        forecast_hash=ledger.entries[0].forecast_hash,
        parent_hash="f" * 64,
        warning_score=ledger.entries[0].warning_score,
        horizon_score=ledger.entries[0].horizon_score,
        entry_hash=ledger.entries[0].entry_hash,
    )
    corrupted = ForecastLedger(entries=(bad_entry,), head_hash=ledger.head_hash, chain_valid=True)
    assert not validate_forecast_ledger(corrupted)


def test_append_rejects_malformed_forecast_ledger() -> None:
    malformed = ForecastLedger(entries=(), head_hash="1" * 64, chain_valid=False)
    with pytest.raises(ValueError, match="malformed forecast ledger"):
        append_forecast_ledger_entry(malformed, "a" * 64, 0.1, 0.1)


def test_same_input_same_bytes() -> None:
    inp = normalize_forecast_inputs("bytes", 0.7, 0.2, 0.3, 0.8)
    out1 = run_phase_transition_forecast_engine(inp)
    out2 = run_phase_transition_forecast_engine(inp)
    blob1 = "|".join(part.to_canonical_json() for part in out1)
    blob2 = "|".join(part.to_canonical_json() for part in out2)
    assert blob1 == blob2


def test_no_decoder_imports() -> None:
    source = Path("src/qec/analysis/phase_transition_forecast_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
            continue
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
            imports.extend(f"{node.module}.{alias.name}" for alias in node.names)
    assert not any(name.startswith("qec.decoder") for name in imports)


def test_zero_drift_is_stable() -> None:
    inp = normalize_forecast_inputs("zero_drift", 1.0, 0.0, 0.0, 1.0)
    probability, warning, boundary, horizon, report, _ = run_phase_transition_forecast_engine(inp)
    assert probability.probability_score == 0.0
    assert warning.escalation_level == "none"
    assert not boundary.projected_boundary_crossing
    assert horizon.stable
    assert not report.transition_detected


def test_boundary_crossing_detected() -> None:
    inp = normalize_forecast_inputs("cross", 0.2, 1.0, 1.0, 0.1)
    _, _, boundary, _, report, _ = run_phase_transition_forecast_engine(inp)
    assert boundary.projected_boundary_crossing
    assert report.transition_detected
    assert boundary.horizon_steps == 1


def test_critical_warning_path() -> None:
    inp = normalize_forecast_inputs("critical", 0.1, 1.0, 1.0, 0.0)
    _, warning, _, horizon, report, _ = run_phase_transition_forecast_engine(inp)
    assert warning.escalation_level == "critical"
    assert warning.warning_detected
    assert not horizon.stable
    assert report.transition_detected


def test_empty_forecast_ledger_baseline() -> None:
    ledger = empty_forecast_ledger()
    assert ledger.entries == ()
    assert ledger.chain_valid
    assert validate_forecast_ledger(ledger)


def test_boundary_zero_closing_rate_max_horizon_steps() -> None:
    inp = normalize_forecast_inputs("horizon_cap", 0.8, 0.0, 0.0, 0.9)
    boundary = forecast_attractor_boundary(inp)
    assert boundary.horizon_steps == MAX_HORIZON_STEPS


def _make_valid_entry(sequence_id: int = 0, parent_hash: str = "0" * 64) -> ForecastLedgerEntry:
    """Build a correctly-hashed ledger entry for use in corruption tests."""
    import hashlib as _hashlib
    import json as _json

    forecast_hash = "a" * 64
    body = {
        "sequence_id": sequence_id,
        "forecast_hash": forecast_hash,
        "parent_hash": parent_hash,
        "warning_score": 0.5,
        "horizon_score": 0.5,
    }
    entry_hash = _hashlib.sha256(
        _json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ForecastLedgerEntry(
        sequence_id=sequence_id,
        forecast_hash=forecast_hash,
        parent_hash=parent_hash,
        warning_score=0.5,
        horizon_score=0.5,
        entry_hash=entry_hash,
    )


def _head_hash_for(parent_hash: str, entry_hash: str) -> str:
    import hashlib as _hashlib
    import json as _json

    payload = {"parent_hash": parent_hash, "entry_hash": entry_hash}
    return _hashlib.sha256(
        _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_validate_ledger_rejects_short_forecast_hash() -> None:
    """A forecast_hash shorter than 64 chars must invalidate the ledger even if entry_hash matches."""
    short_forecast = "a" * 32  # only 32 chars — not a valid sha256
    parent_hash = "0" * 64
    import hashlib as _hashlib
    import json as _json

    body = {
        "sequence_id": 0,
        "forecast_hash": short_forecast,
        "parent_hash": parent_hash,
        "warning_score": 0.5,
        "horizon_score": 0.5,
    }
    entry_hash = _hashlib.sha256(
        _json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    head_hash = _head_hash_for(parent_hash, entry_hash)
    entry = ForecastLedgerEntry(
        sequence_id=0,
        forecast_hash=short_forecast,
        parent_hash=parent_hash,
        warning_score=0.5,
        horizon_score=0.5,
        entry_hash=entry_hash,
    )
    ledger = ForecastLedger(entries=(entry,), head_hash=head_hash, chain_valid=True)
    assert not validate_forecast_ledger(ledger)


def test_validate_ledger_rejects_short_entry_hash() -> None:
    """A entry_hash shorter than 64 chars must invalidate the ledger."""
    entry = _make_valid_entry()
    bad_entry = ForecastLedgerEntry(
        sequence_id=entry.sequence_id,
        forecast_hash=entry.forecast_hash,
        parent_hash=entry.parent_hash,
        warning_score=entry.warning_score,
        horizon_score=entry.horizon_score,
        entry_hash="b" * 32,  # not 64 chars
    )
    head_hash = _head_hash_for(entry.parent_hash, bad_entry.entry_hash)
    ledger = ForecastLedger(entries=(bad_entry,), head_hash=head_hash, chain_valid=True)
    assert not validate_forecast_ledger(ledger)


def test_validate_ledger_rejects_nan_warning_score() -> None:
    """NaN warning_score must invalidate the ledger (and not raise an exception)."""
    entry = _make_valid_entry()
    nan_entry = ForecastLedgerEntry(
        sequence_id=entry.sequence_id,
        forecast_hash=entry.forecast_hash,
        parent_hash=entry.parent_hash,
        warning_score=math.nan,
        horizon_score=entry.horizon_score,
        entry_hash=entry.entry_hash,
    )
    head_hash = _head_hash_for(entry.parent_hash, nan_entry.entry_hash)
    ledger = ForecastLedger(entries=(nan_entry,), head_hash=head_hash, chain_valid=True)
    assert not validate_forecast_ledger(ledger)


def test_validate_ledger_rejects_inf_horizon_score() -> None:
    """Inf horizon_score must invalidate the ledger (and not raise an exception)."""
    entry = _make_valid_entry()
    inf_entry = ForecastLedgerEntry(
        sequence_id=entry.sequence_id,
        forecast_hash=entry.forecast_hash,
        parent_hash=entry.parent_hash,
        warning_score=entry.warning_score,
        horizon_score=math.inf,
        entry_hash=entry.entry_hash,
    )
    head_hash = _head_hash_for(entry.parent_hash, inf_entry.entry_hash)
    ledger = ForecastLedger(entries=(inf_entry,), head_hash=head_hash, chain_valid=True)
    assert not validate_forecast_ledger(ledger)
