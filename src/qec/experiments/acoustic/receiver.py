# SPDX-License-Identifier: MPL-2.0
"""Integer matched-filter bank with explicit harmonic and repetition checks.

Only the noisy samples, public channel configuration, length, and fixed receiver
policy enter this candidate. No transmitted symbol, source hash, error, or
reference syndrome is an input. Acceptance is fallible classical observation,
not an authenticated syndrome; coherent symbol substitutions can pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ._common import CLAIM_BOUNDARY, exact, integer, read, seal
from .channel import ChannelConfig, MAX_SYMBOLS, PCM, SINE48, symbol_block


@dataclass(frozen=True)
class ReceiverPolicy:
    fec: str = "majority"
    tolerance_per_mille: int = 350
    max_residual_per_mille: int = 600
    minimum_margin_per_mille: int = 100
    estimator: str = "matched_filter"

    def __post_init__(self):
        if type(self.fec) is not str or self.fec not in ("majority", "unanimous", "none"):
            raise ValueError("INVALID_INPUT")
        integer(self.tolerance_per_mille, 900, 1)
        integer(self.max_residual_per_mille, 1000, 1)
        integer(self.minimum_margin_per_mille, 1000, 1)
        if type(self.estimator) is not str or self.estimator not in ("matched_filter", "median_period"):
            raise ValueError("INVALID_INPUT")

    def as_dict(self):
        return _seal("receiver-policy", fec=self.fec, tolerance_per_mille=self.tolerance_per_mille,
                    max_residual_per_mille=self.max_residual_per_mille,
                    minimum_margin_per_mille=self.minimum_margin_per_mille,
                    estimator=self.estimator, algorithm="integer_matched_filter_bank.v1", timing="known_fixed_block_grid",
                    fec_none_rule="inspect_first_repetition_only_others_still_transmitted")

    @classmethod
    def from_dict(cls, value):
        return read(cls, value, ("fec", "tolerance_per_mille", "max_residual_per_mille", "minimum_margin_per_mille", "estimator"))


def _seal(name, **fields):
    return seal(name, contract_version="177.2", **fields)


def _features(samples, harmonic, estimator="matched_filter"):
    if estimator == "median_period":
        periods = [_features(samples[start:start+48], harmonic) for start in range(0, len(samples), 48)]
        def median(values):
            values = sorted(values)
            n = len(values)
            return (values[(n-1)//2] + values[n//2]) // 2
        return tuple(median([v[j] for v in periods]) * len(periods) for j in (0, 1))
    return (sum(x * SINE48[(harmonic * n) % 48] for n, x in enumerate(samples)),
            sum(x * SINE48[(harmonic * n + 12) % 48] for n, x in enumerate(samples)))


def _block(samples, config, policy, templates, features):
    rows = []
    candidates = []
    reasons = []
    for h in config.harmonics:
        observed = _features(samples, h, policy.estimator)
        expected = features[h]
        scores = [sum((a - b) ** 2 for a, b in zip(observed, feature)) for feature in expected]
        minimum = min(scores)
        winners = [s for s, score in enumerate(scores) if score == minimum]
        scale = max(sum(x * x for x in feature) for feature in expected)
        close = minimum * 1000000 <= scale * policy.tolerance_per_mille ** 2
        kind = "informative" if math.gcd(h, config.alphabet_size) == 1 else ("dark" if h % config.alphabet_size == 0 else "partial")
        margin_ok = True
        if kind == "informative":
            ordered = sorted(scores)
            margin_ok = ((ordered[1] - ordered[0]) * 1000 >= scale * policy.minimum_margin_per_mille)
            if not close or len(winners) != 1 or not margin_ok:
                reasons.append("ambiguous_or_weak_informative_order")
            else:
                candidates.append(winners[0])
        elif kind == "dark" and not close:
            reasons.append("dark_reference_violation")
        rows.append({"order": h, "kind": kind, "observed_projection": list(observed),
                     "squared_distances": scores, "nearest_symbols": winners,
                     "within_tolerance": close, "margin_passed": margin_ok})
    candidate = candidates[0] if candidates and len(set(candidates)) == 1 else None
    if candidate is None:
        reasons.append("discordant_or_missing_informative_orders")
    else:
        for row in rows:
            if row["kind"] == "partial" and (not row["within_tolerance"] or candidate not in row["nearest_symbols"]):
                reasons.append("partial_order_disagreement")
    residual = None
    energy = None
    if candidate is not None:
        template = templates[candidate]
        residual = sum((x - y) ** 2 for x, y in zip(samples, template))
        energy = sum(y * y for y in template)
        if residual * 1000000 > energy * policy.max_residual_per_mille ** 2:
            reasons.append("excessive_residual")
    accepted = not reasons and candidate is not None
    return {"accepted": accepted, "candidate_symbol": candidate, "estimated_symbol": candidate if accepted else None,
            "reasons": sorted(set(reasons)), "orders": rows, "residual_energy": residual, "template_energy": energy}


def receive_pcm(pcm: PCM, config: ChannelConfig, symbol_count: int,
                policy: ReceiverPolicy = ReceiverPolicy()) -> dict:
    if type(pcm) is not PCM or type(config) is not ChannelConfig or type(policy) is not ReceiverPolicy:
        raise ValueError("INVALID_INPUT")
    integer(symbol_count, MAX_SYMBOLS, 1)
    pcm = PCM.from_dict(pcm.as_dict())
    config = ChannelConfig.from_dict(config.as_dict())
    policy = ReceiverPolicy.from_dict(policy.as_dict())
    reasons = []
    if pcm.sample_rate != config.sample_rate:
        reasons.append("sample_clock_mismatch")
    if len(pcm.samples) != symbol_count * config.repetitions * config.block_samples:
        reasons.append("sample_count_mismatch")
    residues = {h % config.alphabet_size for h in config.harmonics}
    if residues != set(range(config.alphabet_size)):
        reasons.append("incomplete_harmonic_classes")
    per_symbol = []
    estimated = []
    if not reasons:
        templates = [symbol_block(s, config) for s in range(config.alphabet_size)]
        features = {h: [_features(t, h, policy.estimator) for t in templates] for h in config.harmonics}
        for index in range(symbol_count):
            blocks = []
            for repetition in range(config.repetitions):
                start = (index * config.repetitions + repetition) * config.block_samples
                block = pcm.samples[start:start + config.block_samples]
                blocks.append(_block(block, config, policy, templates, features))
            votes = [sum(row["accepted"] and row["estimated_symbol"] == s for row in blocks)
                     for s in range(config.alphabet_size)]
            if policy.fec == "none":
                candidate = blocks[0]["estimated_symbol"]
            elif policy.fec == "unanimous":
                candidate = next((s for s, count in enumerate(votes) if count == config.repetitions), None)
            else:
                candidate = next((s for s, count in enumerate(votes) if 2 * count > config.repetitions), None)
            per_symbol.append({"index": index, "blocks": blocks, "votes": votes, "estimated_symbol": candidate,
                               "accepted": candidate is not None})
            estimated.append(candidate)
        if any(s is None for s in estimated):
            reasons.append("unrecoverable_repetition_word")
    accepted = not reasons
    return _seal("harmonic-observation", config=config.as_dict(), policy=policy.as_dict(), symbol_count=symbol_count,
                received_waveform_sha256=pcm.as_dict()["sha256"], received_pcm_sha256=pcm.as_dict()["pcm_sha256"],
                accepted=accepted, estimated_syndrome=estimated if accepted else None,
                per_symbol=per_symbol, reasons=reasons, receiver_false_accept_possible=True,
                claim_boundary=dict(CLAIM_BOUNDARY))


def validate_observation(receipt, *, expected_pcm: PCM, config: ChannelConfig, symbol_count: int,
                         policy: ReceiverPolicy = ReceiverPolicy()) -> dict:
    expected = receive_pcm(expected_pcm, config, symbol_count, policy)
    exact(receipt, expected)
    return _seal("observation-validation", observation_sha256=expected["sha256"], all_passed=True,
                receiver_replayed=True, syndrome_independently_verified=False,
                physical_provenance_authenticated=False)
