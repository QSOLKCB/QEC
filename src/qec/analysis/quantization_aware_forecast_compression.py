"""v137.0.7 — Quantization-Aware Forecast Compression.

Upgrades policy arbitration into compressed long-horizon
supervisory summaries:

  arbitration decision history
  -> symbolic forecast horizon
  -> quantized compression
  -> bounded summary lattice
  -> replay-safe forecast artifact

Consumes ordered sequences of TemporalAuditoryArbitrationDecision
from v137.0.6.

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from qec.analysis.temporal_auditory_policy_arbitration import (
    ARBITRATION_LOCKDOWN,
    ARBITRATION_MERGE,
    ARBITRATION_PASS_THROUGH,
    ARBITRATION_PRIORITIZE_CRITICAL,
    ARBITRATION_PRIORITIZE_STABLE,
    CONFLICT_CRITICAL,
    CONFLICT_HIGH,
    CONFLICT_LOW,
    CONFLICT_MEDIUM,
    CONFLICT_NONE,
    CONSENSUS_CRITICAL_LOCK,
    CONSENSUS_INTERVENE,
    CONSENSUS_MONITOR,
    CONSENSUS_NONE,
    CONSENSUS_STABILIZE,
    TemporalAuditoryArbitrationDecision,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

QUANTIZATION_AWARE_FORECAST_COMPRESSION_VERSION: str = "v137.0.7"

# ---------------------------------------------------------------------------
# Constants — conflict token family
# ---------------------------------------------------------------------------

CF_NONE: str = "CF_NONE"
CF_LOW: str = "CF_LOW"
CF_MED: str = "CF_MED"
CF_HIGH: str = "CF_HIGH"
CF_CRIT: str = "CF_CRIT"

# ---------------------------------------------------------------------------
# Constants — decision token family
# ---------------------------------------------------------------------------

AR_PASS: str = "AR_PASS"
AR_MERGE: str = "AR_MERGE"
AR_STABLE: str = "AR_STABLE"
AR_CRIT: str = "AR_CRIT"
AR_LOCK: str = "AR_LOCK"

# ---------------------------------------------------------------------------
# Constants — consensus token family
# ---------------------------------------------------------------------------

CS_NONE: str = "CS_NONE"
CS_MON: str = "CS_MON"
CS_STAB: str = "CS_STAB"
CS_INT: str = "CS_INT"
CS_LOCK: str = "CS_LOCK"

# ---------------------------------------------------------------------------
# Constants — stability classes
# ---------------------------------------------------------------------------

STABILITY_STABLE: str = "STABLE"
STABILITY_DRIFTING: str = "DRIFTING"
STABILITY_VOLATILE: str = "VOLATILE"
STABILITY_CRITICAL: str = "CRITICAL"

# ---------------------------------------------------------------------------
# Constants — loss budget classes
# ---------------------------------------------------------------------------

LOSS_LOSSLESS: str = "LOSSLESS"
LOSS_LOW: str = "LOW_LOSS"
LOSS_MEDIUM: str = "MEDIUM_LOSS"
LOSS_HIGH: str = "HIGH_LOSS"

# ---------------------------------------------------------------------------
# Constants — arbitration severity ordering for tie-breaking
# ---------------------------------------------------------------------------

_ARBITRATION_SEVERITY: Dict[str, int] = {
    ARBITRATION_PASS_THROUGH: 0,
    ARBITRATION_MERGE: 1,
    ARBITRATION_PRIORITIZE_STABLE: 2,
    ARBITRATION_PRIORITIZE_CRITICAL: 3,
    ARBITRATION_LOCKDOWN: 4,
}

# ---------------------------------------------------------------------------
# Token mapping tables
# ---------------------------------------------------------------------------

_CONFLICT_TOKEN_MAP: Dict[str, str] = {
    CONFLICT_NONE: CF_NONE,
    CONFLICT_LOW: CF_LOW,
    CONFLICT_MEDIUM: CF_MED,
    CONFLICT_HIGH: CF_HIGH,
    CONFLICT_CRITICAL: CF_CRIT,
}

_ARBITRATION_TOKEN_MAP: Dict[str, str] = {
    ARBITRATION_PASS_THROUGH: AR_PASS,
    ARBITRATION_MERGE: AR_MERGE,
    ARBITRATION_PRIORITIZE_STABLE: AR_STABLE,
    ARBITRATION_PRIORITIZE_CRITICAL: AR_CRIT,
    ARBITRATION_LOCKDOWN: AR_LOCK,
}

_CONSENSUS_TOKEN_MAP: Dict[str, str] = {
    CONSENSUS_NONE: CS_NONE,
    CONSENSUS_MONITOR: CS_MON,
    CONSENSUS_STABILIZE: CS_STAB,
    CONSENSUS_INTERVENE: CS_INT,
    CONSENSUS_CRITICAL_LOCK: CS_LOCK,
}

# ---------------------------------------------------------------------------
# Float precision for deterministic hashing
# ---------------------------------------------------------------------------

FLOAT_PRECISION: int = 12

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastCompressionDecision:
    """Immutable forecast compression decision."""

    horizon_length: int
    compressed_forecast_tokens: Tuple[str, ...]
    compression_ratio: float
    entropy_proxy: float
    dominant_arbitration_mode: str
    forecast_stability_class: str
    loss_budget_class: str
    forecast_symbolic_trace: str
    stable_hash: str
    version: str = QUANTIZATION_AWARE_FORECAST_COMPRESSION_VERSION


@dataclass(frozen=True)
class ForecastCompressionLedger:
    """Immutable ledger of forecast compression decisions."""

    decisions: Tuple[ForecastCompressionDecision, ...]
    decision_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers — canonical JSON & hashing
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)


def _decision_to_canonical_dict(
    decision: ForecastCompressionDecision,
) -> Dict[str, Any]:
    """Convert forecast compression decision to a canonical dict."""
    return {
        "compressed_forecast_tokens": list(decision.compressed_forecast_tokens),
        "compression_ratio": decision.compression_ratio,
        "dominant_arbitration_mode": decision.dominant_arbitration_mode,
        "entropy_proxy": decision.entropy_proxy,
        "forecast_stability_class": decision.forecast_stability_class,
        "forecast_symbolic_trace": decision.forecast_symbolic_trace,
        "horizon_length": decision.horizon_length,
        "loss_budget_class": decision.loss_budget_class,
        "version": decision.version,
    }


def _compute_decision_hash(
    decision: ForecastCompressionDecision,
) -> str:
    """SHA-256 of canonical JSON of a forecast compression decision."""
    payload = _decision_to_canonical_dict(decision)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(
    decisions: Tuple[ForecastCompressionDecision, ...],
) -> str:
    """SHA-256 of ordered decision hashes."""
    hashes = tuple(d.stable_hash for d in decisions)
    canonical = _canonical_json(list(hashes))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _tokenize_decision(
    decision: TemporalAuditoryArbitrationDecision,
) -> str:
    """Convert a single arbitration decision into a composite symbolic token.

    Format: CF_xxx|AR_xxx|CS_xxx
    """
    cf = _CONFLICT_TOKEN_MAP.get(decision.conflict_level, CF_NONE)
    ar = _ARBITRATION_TOKEN_MAP.get(decision.arbitration_decision, AR_PASS)
    cs = _CONSENSUS_TOKEN_MAP.get(decision.consensus_hint, CS_NONE)
    return f"{cf}|{ar}|{cs}"


# ---------------------------------------------------------------------------
# Run-length compression
# ---------------------------------------------------------------------------


def _run_length_compress(
    tokens: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Collapse repeated adjacent identical tokens.

    Example: A A A B B C -> ("A×3", "B×2", "C×1")
    """
    if len(tokens) == 0:
        return ()

    runs = []
    current = tokens[0]
    count = 1

    for t in tokens[1:]:
        if t == current:
            count += 1
        else:
            runs.append(f"{current} ×{count}")
            current = t
            count = 1
    runs.append(f"{current} ×{count}")

    return tuple(runs)


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------


def _compute_compression_ratio(
    compressed_length: int,
    original_length: int,
) -> float:
    """Compute compression ratio bounded (0, 1].

    compressed_length / original_length
    """
    return _round(compressed_length / original_length)


# ---------------------------------------------------------------------------
# Entropy proxy
# ---------------------------------------------------------------------------


def _compute_entropy_proxy(
    tokens: Tuple[str, ...],
) -> float:
    """Compute entropy proxy: unique_token_count / original_length.

    Bounded (0, 1].
    """
    unique_count = len(set(tokens))
    return _round(unique_count / len(tokens))


# ---------------------------------------------------------------------------
# Dominant arbitration mode
# ---------------------------------------------------------------------------


def _detect_dominant_arbitration_mode(
    decisions: Tuple[TemporalAuditoryArbitrationDecision, ...],
) -> str:
    """Detect most frequent arbitration_decision.

    Tie-break by severity ordering:
    LOCKDOWN > PRIORITIZE_CRITICAL > PRIORITIZE_STABLE > MERGE > PASS_THROUGH
    """
    counts: Dict[str, int] = {}
    for d in decisions:
        ad = d.arbitration_decision
        counts[ad] = counts.get(ad, 0) + 1

    max_count = max(counts.values())
    candidates = [ad for ad, c in counts.items() if c == max_count]

    if len(candidates) == 1:
        return candidates[0]

    candidates.sort(
        key=lambda ad: _ARBITRATION_SEVERITY.get(ad, -1),
        reverse=True,
    )
    return candidates[0]


# ---------------------------------------------------------------------------
# Forecast stability class
# ---------------------------------------------------------------------------


def _classify_forecast_stability(
    tokens: Tuple[str, ...],
    compressed_tokens: Tuple[str, ...],
    dominant_mode: str,
) -> str:
    """Classify forecast stability.

    Rules:
    - Single unique token (all repeated) -> STABLE
    - Any repeated LOCKDOWN or PRIORITIZE_CRITICAL dominant -> CRITICAL
    - Low diversity + monotone severity shift -> DRIFTING
    - Mixed tokens / high diversity -> VOLATILE
    """
    unique_tokens = set(tokens)

    # STABLE: single unique token always wins first — even if severe.
    # A uniform horizon is stable by definition regardless of severity.
    if len(unique_tokens) == 1:
        return STABILITY_STABLE

    # CRITICAL: dominant is severe AND repeated severe tokens (with diversity)
    if dominant_mode in (ARBITRATION_LOCKDOWN, ARBITRATION_PRIORITIZE_CRITICAL):
        severe_count = sum(1 for t in tokens if _is_severe_token(t))
        if severe_count >= 2:
            return STABILITY_CRITICAL

    # Diversity metric
    diversity = len(unique_tokens) / len(tokens)

    # DRIFTING: low diversity (≤ 0.5) and no severe tokens dominate
    if diversity <= 0.5:
        return STABILITY_DRIFTING

    # VOLATILE: high diversity / mixed tokens
    return STABILITY_VOLATILE


def _is_severe_token(token: str) -> bool:
    """Check if a token contains critical or lockdown indicators."""
    return "AR_CRIT" in token or "AR_LOCK" in token


# ---------------------------------------------------------------------------
# Loss budget class
# ---------------------------------------------------------------------------


def _classify_loss_budget(
    compression_ratio: float,
    entropy_proxy: float,
) -> str:
    """Classify loss budget from compression ratio and entropy proxy.

    Rules:
    - compression_ratio == 1.0 -> LOSSLESS
    - compression_ratio > 0.7 and entropy_proxy <= 0.3 -> LOW_LOSS
    - compression_ratio > 0.4 -> MEDIUM_LOSS
    - else -> HIGH_LOSS
    """
    if compression_ratio >= 1.0:
        return LOSS_LOSSLESS
    if compression_ratio > 0.7 and entropy_proxy <= 0.3:
        return LOSS_LOW
    if compression_ratio > 0.4:
        return LOSS_MEDIUM
    return LOSS_HIGH


# ---------------------------------------------------------------------------
# Symbolic trace
# ---------------------------------------------------------------------------


def _build_forecast_symbolic_trace(
    compressed_tokens: Tuple[str, ...],
) -> str:
    """Build forecast symbolic trace from compressed tokens.

    Example: CF_LOW|AR_MERGE|CS_STAB ×3 -> CF_HIGH|AR_CRIT|CS_INT ×2
    """
    return " -> ".join(compressed_tokens)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def compress_forecast_horizon(
    decisions: Sequence[TemporalAuditoryArbitrationDecision],
) -> ForecastCompressionDecision:
    """Compress an ordered sequence of arbitration decisions into a forecast.

    Parameters
    ----------
    decisions : Sequence[TemporalAuditoryArbitrationDecision]
        Ordered arbitration decisions from v137.0.6.
        Normalized to tuple internally.

    Returns
    -------
    ForecastCompressionDecision
        Frozen, hash-stable forecast compression decision.

    Raises
    ------
    TypeError
        If decisions is not iterable or contains wrong types.
    ValueError
        If decisions is empty.
    """
    if isinstance(decisions, (str, bytes, dict)):
        raise TypeError(
            f"decisions must be iterable of TemporalAuditoryArbitrationDecision, "
            f"got {type(decisions).__name__}"
        )
    if not hasattr(decisions, "__iter__"):
        raise TypeError(
            f"decisions must be iterable, got {type(decisions).__name__}"
        )
    decisions = tuple(decisions)

    for i, d in enumerate(decisions):
        if not isinstance(d, TemporalAuditoryArbitrationDecision):
            raise TypeError(
                f"decisions[{i}] must be TemporalAuditoryArbitrationDecision, "
                f"got {type(d).__name__}"
            )
    if len(decisions) == 0:
        raise ValueError("decisions must not be empty")

    # 1) Tokenization
    raw_tokens = tuple(_tokenize_decision(d) for d in decisions)

    # 2) Run-length compression
    compressed_tokens = _run_length_compress(raw_tokens)

    # 3) Compression ratio
    compression_ratio = _compute_compression_ratio(
        len(compressed_tokens), len(raw_tokens),
    )

    # 4) Entropy proxy
    entropy_proxy = _compute_entropy_proxy(raw_tokens)

    # 5) Dominant arbitration mode
    dominant_mode = _detect_dominant_arbitration_mode(decisions)

    # 6) Forecast stability class
    stability_class = _classify_forecast_stability(
        raw_tokens, compressed_tokens, dominant_mode,
    )

    # 7) Loss budget class
    loss_budget = _classify_loss_budget(compression_ratio, entropy_proxy)

    # 8) Symbolic trace
    symbolic_trace = _build_forecast_symbolic_trace(compressed_tokens)

    proto = ForecastCompressionDecision(
        horizon_length=len(decisions),
        compressed_forecast_tokens=compressed_tokens,
        compression_ratio=compression_ratio,
        entropy_proxy=entropy_proxy,
        dominant_arbitration_mode=dominant_mode,
        forecast_stability_class=stability_class,
        loss_budget_class=loss_budget,
        forecast_symbolic_trace=symbolic_trace,
        stable_hash="",
    )
    stable_hash = _compute_decision_hash(proto)

    return ForecastCompressionDecision(
        horizon_length=len(decisions),
        compressed_forecast_tokens=compressed_tokens,
        compression_ratio=compression_ratio,
        entropy_proxy=entropy_proxy,
        dominant_arbitration_mode=dominant_mode,
        forecast_stability_class=stability_class,
        loss_budget_class=loss_budget,
        forecast_symbolic_trace=symbolic_trace,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def build_forecast_compression_ledger(
    decisions: Any,
) -> ForecastCompressionLedger:
    """Build an immutable forecast compression ledger.

    Parameters
    ----------
    decisions : iterable of ForecastCompressionDecision
        Decisions to collect. Normalized to a tuple internally.

    Returns
    -------
    ForecastCompressionLedger
    """
    decisions = tuple(decisions)
    for i, d in enumerate(decisions):
        if not isinstance(d, ForecastCompressionDecision):
            raise TypeError(
                f"decisions[{i}] must be ForecastCompressionDecision, "
                f"got {type(d).__name__}"
            )
    ledger_hash = _compute_ledger_hash(decisions)
    return ForecastCompressionLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=ledger_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_forecast_compression_bundle(
    decision: ForecastCompressionDecision,
) -> Dict[str, Any]:
    """Export a single forecast compression decision as a canonical JSON-safe dict.

    Deterministic: same decision always produces byte-identical export.
    """
    base = _decision_to_canonical_dict(decision)
    base["layer"] = "quantization_aware_forecast_compression"
    base["stable_hash"] = decision.stable_hash
    return base


def export_forecast_compression_ledger(
    ledger: ForecastCompressionLedger,
) -> Dict[str, Any]:
    """Export a ledger as a canonical JSON-safe dict.

    Deterministic: same ledger always produces byte-identical export.
    """
    return {
        "decision_count": ledger.decision_count,
        "decisions": [
            export_forecast_compression_bundle(d)
            for d in ledger.decisions
        ],
        "layer": "quantization_aware_forecast_compression",
        "stable_hash": ledger.stable_hash,
        "version": QUANTIZATION_AWARE_FORECAST_COMPRESSION_VERSION,
    }
