# SPDX-License-Identifier: MPL-2.0
"""Bounded integer-PCM faults and explicitly permitted reference subtraction.

These are synthetic classical channel experiments. All operations are applied
in declared order to actual samples, using fixed integer arithmetic, the channel
LUT, and the specified LCG32 algorithm. Cancellation receives no clean target,
symbols, decoder, or noise-generation plan. A reference is an explicit separate
input whose independence and permission are caller declarations, not something
an embedded hash can establish.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._common import CLAIM_BOUNDARY, digest, exact, integer, read, seal, sequence
from .channel import MAX_SAMPLES, PCM, SINE48

MAX_FAULTS = 16
RNG_ALGORITHM = "lcg32_a1664525_c1013904223_high16_v1"
# Required parameters, with inclusive integer bounds. No implicit defaults or
# ignored extra keys are accepted. Order is canonical lexicographic order.
_PARAMETERS = {
    "broadband": {"amplitude": (0, 16384)},
    "colored": {"amplitude": (0, 16384), "memory": (1, 255)},
    "correlated": {"amplitude": (0, 16384), "period": (1, 4096)},
    "speech_like": {"amplitude": (0, 16384), "envelope_period": (48, 4096),
                    "fundamental_period": (24, 960)},
    "impulses": {"amplitude": (0, 32767), "interval": (1, 4096), "offset": (0, 4095)},
    "clipping": {"limit": (1, 32767)},
    "timing_slip": {"at_sample": (0, MAX_SAMPLES), "slip_samples": (-4096, 4096)},
    "missing_samples": {"count": (1, 4096), "start": (0, MAX_SAMPLES - 1)},
    "phase_drift": {"period": (1, 4096), "sample_step": (-24, 24)},
    "nonlinear": {"gain_denominator": (1, 4096), "gain_numerator": (0, 32768)},
    "echo": {"delay": (1, 4096), "gain_denominator": (1, 4096), "gain_numerator": (0, 16384)},
    "aliasing": {"factor": (2, 16)},
}
ADDITIVE_FAULTS = ("broadband", "colored", "correlated", "speech_like", "impulses")


def _seal(name: str, **fields) -> dict:
    return seal(name, contract_version="177.2", **fields)


@dataclass(frozen=True)
class NoiseFault:
    kind: str
    parameters: tuple[tuple[str, int], ...]

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in _PARAMETERS:
            raise ValueError("INVALID_INPUT")
        pairs = sequence(self.parameters, 3)
        normalized = []
        for pair in pairs:
            pair = sequence(pair, 2)
            if len(pair) != 2 or type(pair[0]) is not str:
                raise ValueError("INVALID_INPUT")
            normalized.append(pair)
        if tuple(pair[0] for pair in normalized) != tuple(sorted(_PARAMETERS[self.kind])):
            raise ValueError("INVALID_INPUT")
        for key, value in normalized:
            low, high = _PARAMETERS[self.kind][key]
            integer(value, high, low)
        object.__setattr__(self, "parameters", tuple(normalized))
        p = dict(self.parameters)
        if self.kind == "impulses" and p["offset"] >= p["interval"]:
            raise ValueError("INVALID_INPUT")
        if self.kind in ("nonlinear", "echo"):
            maximum_gain = 8 if self.kind == "nonlinear" else 4
            if p["gain_numerator"] > maximum_gain * p["gain_denominator"]:
                raise ValueError("INVALID_INPUT")

    @classmethod
    def make(cls, kind: str, **parameters):
        """Convenience constructor; the stored representation remains immutable."""
        return cls(kind, tuple(sorted(parameters.items())))

    def as_dict(self):
        return _seal("noise-fault", kind=self.kind, parameters=dict(self.parameters))

    @classmethod
    def from_dict(cls, value):
        def pairs(parameters):
            if type(parameters) is not dict or any(type(key) is not str for key in parameters):
                raise ValueError("INVALID_INPUT")
            return tuple(sorted(parameters.items()))
        return read(cls, value, ("kind", "parameters"), {"parameters": pairs})


@dataclass(frozen=True)
class NoisePlan:
    faults: tuple[NoiseFault, ...] = ()
    seed: int = 1

    def __post_init__(self):
        faults = sequence(self.faults, MAX_FAULTS)
        if any(type(fault) is not NoiseFault for fault in faults):
            raise ValueError("INVALID_INPUT")
        integer(self.seed, 0xFFFFFFFF)
        object.__setattr__(self, "faults", faults)

    def as_dict(self):
        return _seal("noise-plan", faults=[fault.as_dict() for fault in self.faults], seed=self.seed,
                     rng_algorithm=RNG_ALGORITHM, order="listed_sequential_sample_transforms",
                     arithmetic="unbounded_integer_floor_division_then_saturate_int16",
                     missing_samples="delete_without_resampling", aliasing="unfiltered_decimate_then_zero_order_hold",
                     phase_drift="integer_time_warp_zero_outside_input",
                     speech_like="synthetic_harmonic_envelope_proxy_not_actual_speech")

    @classmethod
    def from_dict(cls, value):
        return read(cls, value, ("faults", "seed"),
                    {"faults": lambda values: tuple(NoiseFault.from_dict(v) for v in sequence(values, MAX_FAULTS))})


def _saturate(sample: int) -> int:
    return max(-32768, min(32767, sample))


class _LCG32:
    def __init__(self, seed: int):
        self.state = seed

    def signed(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) - 32768


def _transform(samples: tuple[int, ...], fault: NoiseFault, rng: _LCG32):
    p, kind, size = dict(fault.parameters), fault.kind, len(samples)
    raw = []
    if kind == "broadband":
        raw = [sample + p["amplitude"] * rng.signed() // 32768 for sample in samples]
    elif kind == "colored":
        state = 0
        for sample in samples:
            state = (p["memory"] * state + rng.signed()) // (p["memory"] + 1)
            raw.append(sample + p["amplitude"] * state // 32768)
    elif kind == "correlated":
        period = tuple(p["amplitude"] * rng.signed() // 32768 for _ in range(p["period"]))
        raw = [sample + period[n % len(period)] for n, sample in enumerate(samples)]
    elif kind == "speech_like":
        # Three fixed harmonics, triangular syllable-like amplitude envelope,
        # and a small seeded unvoiced component. This is not a speech model.
        for n, sample in enumerate(samples):
            phase = (48 * n // p["fundamental_period"]) % 48
            voiced = (4 * SINE48[phase] + 2 * SINE48[(2 * phase) % 48] + SINE48[(3 * phase) % 48]) // 7
            carrier = (7 * voiced + rng.signed()) // 8
            envelope = p["envelope_period"] - abs(2 * (n % p["envelope_period"]) - p["envelope_period"])
            raw.append(sample + p["amplitude"] * carrier * envelope // (32768 * p["envelope_period"]))
    elif kind == "impulses":
        raw = [sample + (p["amplitude"] * (1 if ((n - p["offset"]) // p["interval"]) % 2 == 0 else -1)
                         if n >= p["offset"] and (n - p["offset"]) % p["interval"] == 0 else 0)
               for n, sample in enumerate(samples)]
    elif kind == "clipping":
        raw = [max(-p["limit"], min(p["limit"], sample)) for sample in samples]
    elif kind == "timing_slip":
        start, slip = p["at_sample"], p["slip_samples"]
        if start > size or (slip < 0 and start - slip > size):
            raise ValueError("INVALID_INPUT")
        if slip >= 0:
            raw = list(samples[:start]) + [0] * slip + list(samples[start:])
            raw = raw[:size]
        else:
            raw = list(samples[:start]) + list(samples[start - slip:]) + [0] * (-slip)
    elif kind == "missing_samples":
        start, count = p["start"], p["count"]
        if start + count > size:
            raise ValueError("INVALID_INPUT")
        raw = list(samples[:start] + samples[start + count:])
    elif kind == "phase_drift":
        for n in range(size):
            index = n + n * p["sample_step"] // p["period"]
            raw.append(samples[index] if 0 <= index < size else 0)
    elif kind == "nonlinear":
        raw = [sample + (p["gain_numerator"] * sample ** 3) // (p["gain_denominator"] * 32768 ** 2)
               for sample in samples]
    elif kind == "echo":
        raw = [sample + (p["gain_numerator"] * samples[n - p["delay"]] // p["gain_denominator"]
                         if n >= p["delay"] else 0) for n, sample in enumerate(samples)]
    elif kind == "aliasing":
        raw = [samples[(n // p["factor"]) * p["factor"]] for n in range(size)]
    else:  # Defensive even though the immutable fault constructor validates it.
        raise ValueError("INVALID_INPUT")
    saturated = sum(sample < -32768 or sample > 32767 for sample in raw)
    return tuple(_saturate(sample) for sample in raw), saturated


def _replay(original: PCM, plan: NoisePlan):
    samples, rng, stages = original.samples, _LCG32(plan.seed), []
    for index, fault in enumerate(plan.faults):
        before = PCM(samples, original.sample_rate)
        samples, saturated = _transform(samples, fault, rng)
        after = PCM(samples, original.sample_rate)
        stages.append({"index": index, "fault_sha256": fault.as_dict()["sha256"],
                       "input_pcm_sha256": before.as_dict()["sha256"],
                       "output_pcm_sha256": after.as_dict()["sha256"],
                       "input_sample_count": len(before.samples), "output_sample_count": len(samples),
                       "changed_aligned_samples": sum(a != b for a, b in zip(before.samples, samples)),
                       "saturated_sample_count": saturated, "rng_state_after": rng.state})
    return PCM(samples, original.sample_rate), stages


def apply_noise(original: PCM, plan: NoisePlan, *, source_sha256: str) -> dict:
    """Retain original and transformed PCM plus exact ordered replay evidence."""
    if type(original) is not PCM or type(plan) is not NoisePlan:
        raise ValueError("INVALID_INPUT")
    digest(source_sha256)
    observed, stages = _replay(original, plan)
    return _seal("noise-observation", source_sha256=source_sha256,
                 original=original.as_dict(), plan=plan.as_dict(), observed=observed.as_dict(), stages=stages,
                 cancellation_applied=False, claim_boundary=dict(CLAIM_BOUNDARY))


def validate_noise_observation(value, *, expected_source_sha256: str,
                               expected_original_sha256: str, expected_plan_sha256: str) -> dict:
    """Pins must come from outside this artifact; self-hash validity is insufficient."""
    for pin in (expected_source_sha256, expected_original_sha256, expected_plan_sha256):
        digest(pin)
    try:
        if type(value) is not dict:
            raise ValueError("INVALID_INPUT")
        original, plan = PCM.from_dict(value["original"]), NoisePlan.from_dict(value["plan"])
        if (value["source_sha256"] != expected_source_sha256 or
                original.as_dict()["sha256"] != expected_original_sha256 or
                plan.as_dict()["sha256"] != expected_plan_sha256):
            raise ValueError("INVALID_INPUT")
        expected = apply_noise(original, plan, source_sha256=expected_source_sha256)
        exact(value, expected)
        return _seal("noise-validation", observation_sha256=expected["sha256"], all_passed=True,
                     full_sample_replay=True, independently_pinned_inputs=True,
                     physical_provenance_authenticated=False)
    except (KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def independent_noise_reference(sample_count: int, plan: NoisePlan, *, sample_rate: int = 48000) -> PCM:
    """Privileged synthetic baseline generated without reading any target samples.

    Only signal-independent additive families are eligible. The reference models
    their pre-saturation sum; it is not the observed-minus-clean residual. Plans
    that would saturate the reference are rejected because that would conceal
    the unavailable unsaturated noise. Cancellation may still fail if mixing
    the reference with a target clipped, or if alignment/gain/phase are wrong.
    """
    integer(sample_count, MAX_SAMPLES)
    if type(plan) is not NoisePlan or any(f.kind not in ADDITIVE_FAULTS for f in plan.faults):
        raise ValueError("INVALID_INPUT")
    reference, stages = _replay(PCM((0,) * sample_count, sample_rate), plan)
    if any(stage["saturated_sample_count"] for stage in stages):
        raise ValueError("INVALID_INPUT")
    return reference


@dataclass(frozen=True)
class CancellationConfig:
    reference_source_sha256: str
    reference_role: str
    permission_purpose: str
    permission_granted: bool
    independent_reference: bool
    delay_samples: int = 0
    gain_numerator: int = 1
    gain_denominator: int = 1
    phase_degrees: int = 0

    def __post_init__(self):
        digest(self.reference_source_sha256)
        if (type(self.reference_role) is not str or self.reference_role not in
                ("privileged_synthetic_reference", "independent_noise_measurement") or
                type(self.permission_purpose) is not str or self.permission_purpose != "noise_cancellation" or
                self.permission_granted is not True or self.independent_reference is not True):
            raise ValueError("INVALID_INPUT")
        integer(self.delay_samples, 4096, -4096)
        integer(self.gain_numerator, 32768)
        integer(self.gain_denominator, 4096, 1)
        if self.gain_numerator > 8 * self.gain_denominator:
            raise ValueError("INVALID_INPUT")
        if type(self.phase_degrees) is not int or self.phase_degrees not in (0, 180):
            raise ValueError("INVALID_INPUT")

    def as_dict(self):
        return _seal("cancellation-config", reference_source_sha256=self.reference_source_sha256,
                     reference_role=self.reference_role, permission_purpose=self.permission_purpose,
                     permission_granted=self.permission_granted, independent_reference=self.independent_reference,
                     delay_samples=self.delay_samples, gain_numerator=self.gain_numerator,
                     gain_denominator=self.gain_denominator, phase_degrees=self.phase_degrees,
                     phase_model="declared_zero_or_180_degree_polarity_only",
                     alignment="reference_index_equals_observed_index_minus_delay_zero_outside",
                     parameter_selection="caller_declared_no_target_fitting")

    @classmethod
    def from_dict(cls, value):
        return read(cls, value, ("reference_source_sha256", "reference_role", "permission_purpose",
                                "permission_granted", "independent_reference", "delay_samples",
                                "gain_numerator", "gain_denominator", "phase_degrees"))


def cancel_reference(observed: PCM, reference: PCM, config: CancellationConfig,
                     *, observed_source_sha256: str) -> dict:
    """Subtract an explicit reference with declared alignment, gain and polarity.

    No inferred alignment, adaptive fitting, ideal symbol, or clean target is
    available here. An unrelated voice/reference has no guaranteed benefit.
    """
    if type(observed) is not PCM or type(reference) is not PCM or type(config) is not CancellationConfig:
        raise ValueError("INVALID_INPUT")
    digest(observed_source_sha256)
    if reference.sample_rate != observed.sample_rate or not reference.samples:
        raise ValueError("INVALID_INPUT")
    polarity = 1 if config.phase_degrees == 0 else -1
    raw, used = [], 0
    for n, sample in enumerate(observed.samples):
        index = n - config.delay_samples
        if 0 <= index < len(reference.samples):
            component = polarity * config.gain_numerator * reference.samples[index] // config.gain_denominator
            used += 1
        else:
            component = 0
        raw.append(sample - component)
    filtered = PCM(tuple(_saturate(sample) for sample in raw), observed.sample_rate)
    return _seal("cancellation-receipt", observed_source_sha256=observed_source_sha256,
                 reference_source_sha256=config.reference_source_sha256, config=config.as_dict(),
                 observed=observed.as_dict(), reference=reference.as_dict(), filtered=filtered.as_dict(),
                 reference_used_samples=used, saturated_sample_count=sum(s != _saturate(s) for s in raw),
                 changed_samples=sum(a != b for a, b in zip(observed.samples, filtered.samples)),
                 clean_target_access=False, ideal_symbol_access=False, automatic_reference_inference=False,
                 independence_authority="caller_declared_not_authenticated_by_hash",
                 improvement_claimed=False, claim_boundary=dict(CLAIM_BOUNDARY))


def validate_cancellation_receipt(value, *, expected_observed_source_sha256: str,
                                  expected_reference_source_sha256: str, expected_observed_sha256: str,
                                  expected_reference_sha256: str, expected_config_sha256: str) -> dict:
    for pin in (expected_observed_source_sha256, expected_reference_source_sha256, expected_observed_sha256,
                expected_reference_sha256, expected_config_sha256):
        digest(pin)
    try:
        if type(value) is not dict:
            raise ValueError("INVALID_INPUT")
        observed, reference = PCM.from_dict(value["observed"]), PCM.from_dict(value["reference"])
        config = CancellationConfig.from_dict(value["config"])
        if (value["observed_source_sha256"] != expected_observed_source_sha256 or
                config.reference_source_sha256 != expected_reference_source_sha256 or
                observed.as_dict()["sha256"] != expected_observed_sha256 or
                reference.as_dict()["sha256"] != expected_reference_sha256 or
                config.as_dict()["sha256"] != expected_config_sha256):
            raise ValueError("INVALID_INPUT")
        expected = cancel_reference(observed, reference, config, observed_source_sha256=expected_observed_source_sha256)
        exact(value, expected)
        return _seal("cancellation-validation", cancellation_sha256=expected["sha256"], all_passed=True,
                     full_sample_replay=True, independently_pinned_inputs=True,
                     independent_reference_permission_recomputed=True, physical_provenance_authenticated=False)
    except (KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc
