# v177.2 — Deterministic PCM noise and reference cancellation

This opt-in experiment implements the v177.2 noise/cancellation boundary from
[the acoustic roadmap](https://github.com/QSOLKCB/QEC/blob/docs/acoustic-quantum-protection-roadmap-v177/ROADMAP.md).
It transforms actual signed 16-bit mono PCM samples. It neither changes existing
decoders or switching contracts nor imports a receiver or decoder. A successful
replay establishes a declared classical computation, not a physical quantum
benefit, microphone result, real-speech separation result, or provenance proof.

## Contracts and inputs

`src/qec/experiments/acoustic/noise.py` provides:

- `NoiseFault`: immutable operation name and canonical key/value parameter tuple.
- `NoisePlan`: immutable ordered fault tuple and explicit unsigned 32-bit seed.
- `apply_noise`: separate original and observed PCM with an ordered stage trace.
- `independent_noise_reference`: a privileged synthetic additive-noise baseline.
- `CancellationConfig`: explicit reference source, role, permission, delay, gain,
  and supported phase declaration.
- `cancel_reference`: separate observed, reference, and filtered PCM receipt.
- Strict, independently pinned validators for both receipt types.

`NoiseFault.make(kind, **parameters)` canonicalizes parameter names for
convenience. Direct tuple construction requires lexicographic ordering, no
repeated keys, and exactly the required keys. Unknown parameters and operations
are errors, not ignored configuration. Inputs are bounded to 16 sequential
faults and the channel's 73,728 samples. Booleans are not integer parameters.
The sample rate is inherited unchanged from the PCM contract (24 or 48 kHz).

## Normative arithmetic

Each step consumes the preceding step's actual PCM, in listed order. All
products, sums, indices, state transitions, and reductions use integers. Division
uses Python floor division, including for negative values. Intermediate integers
are unbounded; every completed step saturates to `[-32768, 32767]`. Consequently,
changing operation order can change the result. Clipping cannot be undone by
subtraction after the information is lost.

The fixed `SINE48` channel lookup table is the only trigonometric source. There
is no runtime libm trigonometry, NumPy/SciPy generator, ambient random state,
wall clock, adaptive parameter fitting, or floating-point DSP.

The sole pseudo-random algorithm is versioned as
`lcg32_a1664525_c1013904223_high16_v1`:

```
state = (1664525 * state + 1013904223) mod 2^32
draw = (state >> 16) - 32768
```

One generator starts from the plan seed and is shared by its sequential steps.
Broadband and colored noise consume one draw per current sample. Correlated
noise consumes exactly `period` draws, even for a shorter/empty input, then
repeats that fixed sequence. The speech-like proxy consumes one draw per current
sample. Other operations consume none. Each stage records the resulting state.
These finite seeded sequences are synthetic disturbances, not a statistical
claim about environmental noise.

A seed-1, amplitude-1000 broadband step applied to twelve zero samples has this
normative little-endian PCM hexadecimal output:

```
f0fdfafe080099017dfcfbfe2502700038fc16010cfe64ff
```

The focused tests pin its plan and complete observation hashes, and check replay
in separate processes with different `PYTHONHASHSEED` values. Integer arithmetic
provides a portable byte contract; those local process checks are not a claim
that a multi-operating-system CI matrix has already run.

## Implemented sample transforms

Required parameters and inclusive bounds are part of the versioned contract.

| Kind | Parameters | Exact interpretation |
| --- | --- | --- |
| `broadband` | `amplitude=0..16384` | Add `amplitude * draw // 32768` per sample; bounded uniform, not Gaussian noise. |
| `colored` | `amplitude=0..16384`, `memory=1..255` | Begin state at zero; update `(memory * state + draw) // (memory + 1)` and scale as broadband. This changes temporal correlation and power; it does not normalize RMS to the broadband case. |
| `correlated` | `amplitude=0..16384`, `period=1..4096` | Generate one seeded scaled-noise period and repeat it. Correlation is temporal within the interference, with no clean-target access. |
| `speech_like` | `amplitude=0..16384`, `envelope_period=48..4096`, `fundamental_period=24..960` | Add a fixed three-harmonic carrier with 4:2:1 weights, 7:1 voiced/noise mix, and a triangular envelope. The lookup phase is `48*n // fundamental_period`. This is clearly a synthetic overlap proxy, not actual speech or a validated vocal-tract model. |
| `impulses` | `amplitude=0..32767`, `interval=1..4096`, `offset=0..interval-1` | At offset and every interval thereafter, add alternating positive/negative impulses. |
| `clipping` | `limit=1..32767` | Hard clip each sample to `[-limit, limit]`. |
| `timing_slip` | `at_sample=0..73728`, `slip_samples=-4096..4096` | Positive slip inserts zeros at the event and truncates the tail to the original length; negative slip deletes that many samples and pads the tail with zeros. The event/deletion must fit the current input. |
| `missing_samples` | `start=0..73727`, `count=1..4096` | Delete an exact contiguous range, reducing PCM length. The full range must exist. There is no hidden replacement or resampling. |
| `phase_drift` | `period=1..4096`, `sample_step=-24..24` | Output index `n` reads input index `n + n*sample_step // period`, or zero outside the input. This is a quantized time-warp proxy for accumulating carrier phase drift, not an analog oscillator model. |
| `nonlinear` | `gain_numerator=0..32768`, `gain_denominator=1..4096`, ratio at most 8 | For input `x`, emit `x + gain_numerator*x^3 // (gain_denominator*32768^2)` before saturation. |
| `echo` | `delay=1..4096`, `gain_numerator=0..16384`, `gain_denominator=1..4096`, ratio at most 4 | Add one gain-scaled delayed tap from the input to this step. No recursive feedback or future samples. |
| `aliasing` | `factor=2..16` | Unfiltered decimation followed by zero-order hold: sample `n` reads input `(n // factor)*factor`. This is a sampled-signal aliasing/distortion test, not a calibrated analog frontend. |

A stage records the fault identity, input/output PCM identities and lengths,
aligned samples that changed, int16 saturation count, and final generator state.
The count of aligned changed samples compares only positions present in both
arrays; the separately retained lengths expose deletion. The saturation count
counts int16 overflow, not samples changed by the optional lower clipping limit.

## Reference permissions and authority

The cancellation function accepts only `(observed, reference, config)` and the
observed-source commitment. It cannot inspect clean samples, ideal symbols,
syndromes, noise plans, or a decoder. It never estimates alignment, gain, or
phase from the target. A second voice is not automatically a noise reference.

The caller must explicitly provide:

- A separately supplied reference PCM and its declared source SHA-256.
- A role of `independent_noise_measurement` or `privileged_synthetic_reference`.
- Exact purpose `noise_cancellation`, `permission_granted=True`, and
  `independent_reference=True`.
- Integer delay in `[-4096, 4096]`, nonnegative rational gain at most 8, and
  phase polarity of exactly 0 or 180 degrees.

The available phase operation is deliberately restricted to polarity. An
arbitrary complex phase rotation of a nonperiodic real waveform is not silently
invented. Other angles are rejected. Positive delay uses
`reference[n - delay]`; out-of-range reference samples are zero. The gain and
phase operation is `polarity * numerator * sample // denominator`, after which
it is subtracted from the observed sample and saturated. Reference and observed
sample rates must agree; the reference must contain at least one sample.

Reference independence, source identity, and permission are caller declarations.
A hash binds those declarations and bytes; it cannot authenticate that a
measurement is independent, physically sourced, or permissioned. Validators
therefore require external commitments and explicitly decline physical
provenance claims.

`independent_noise_reference` accepts only sample count, sample rate, and a
plan composed of signal-independent additive families (broadband, colored,
correlated, synthetic speech-like, impulses). It evaluates those generators
against zeros without accessing a target. It is a privileged synthetic baseline,
not available noise knowledge for a general receiver. Its reference stream is
rejected if intermediate mixing would saturate; it is never computed as
`observed - clean`. Mixing with the actual target can still saturate and make
exact reconstruction impossible. Reference acquisition cost and privileged
knowledge must be reported by any comparative harness.

## Usage

```python
from qec.experiments.acoustic.channel import PCM, encode_symbols
from qec.experiments.acoustic.noise import (
    NoiseFault, NoisePlan, CancellationConfig, apply_noise,
    independent_noise_reference, cancel_reference,
)

original = encode_symbols((0, 1, 2))
plan = NoisePlan((NoiseFault.make("broadband", amplitude=700),), seed=23)
# Supply actual caller-trusted commitments, not hashes extracted from the
# received artifact and then treated as independent evidence.
observation = apply_noise(original, plan, source_sha256=source_commitment)
observed = PCM.from_dict(observation["observed"])
reference = independent_noise_reference(len(original.samples), plan,
                                        sample_rate=original.sample_rate)
config = CancellationConfig(
    reference_source_sha256=reference_commitment,
    reference_role="privileged_synthetic_reference",
    permission_purpose="noise_cancellation",
    permission_granted=True,
    independent_reference=True,
    delay_samples=0,
    gain_numerator=1,
    gain_denominator=1,
    phase_degrees=0,
)
cancellation = cancel_reference(observed, reference, config,
                                observed_source_sha256=observation["sha256"])
filtered = PCM.from_dict(cancellation["filtered"])
```

Original PCM remains in the noise observation. Observed and reference PCM remain
in the separate cancellation receipt alongside filtered PCM. Cancellation
receipts make no improvement, symbol-trust, or decoder-success claim. Such
comparisons belong to independently validated trial evidence; the cancellation
algorithm is not allowed to read clean truth to improve its answer.

## Validation and negative controls

`validate_noise_observation` requires all three keyword pins:
`expected_source_sha256`, `expected_original_sha256`, `expected_plan_sha256`.
The latter two commit to the full canonical PCM/plan artifacts, not only raw PCM
bytes. It parses the original and plan strictly and recomputes every transformed
sample, stage, claim, schema, and hash before accepting.

`validate_cancellation_receipt` requires all five keyword pins:
`expected_observed_source_sha256`, `expected_reference_source_sha256`,
`expected_observed_sha256`, `expected_reference_sha256`, and
`expected_config_sha256`. It recomputes subtraction and the complete receipt.
These pins must be independently held upstream commitments. Unknown fields,
wrong types, omitted keys, stale hashes, changed pinned inputs, and rehashed
result/claim/counter tampering fail with `ValueError("INVALID_INPUT")`. Missing
required keyword arguments fail at the Python API boundary.

`tests/acoustic/test_noise.py` verifies all twelve real-sample transforms,
known-answer bytes, immutable/bounded plans, exact timing/deletion/echo/aliasing,
saturation, source pins, complete replay, and rehashed adversarial artifacts.
An aligned known reference recovers an unclipped fixture exactly. Wrong-phase,
wrong-delay, and unrelated seeded references strictly worsen the fixture's
squared sample error relative to leaving it unfiltered. That metric is used by
the independent test evaluator, never by the canceller. Negative results and
irrecoverable clipping are retained. This module does not claim that reference
subtraction beats matched filters, FEC, or robust spectral estimators; those are
separate receiver/trial comparisons.

Focused command from a configured checkout:

```
PYTHONPATH=src python -m pytest -q tests/acoustic/test_noise.py
```

Any release still requires the aggregate suite and existing protected-contract
regressions. No release, merge, tag, or physical acceptance follows merely from
these focused tests passing.
