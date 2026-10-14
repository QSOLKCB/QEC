# Trusted PCM-to-syndrome bridge (v177.3)

This additive experiment connects actual sampled classical waveforms to the
existing exact five-site qutrit and packed-ququart software decoders. It does
not change `src/qec/decoder/`, call the ideal-sample fallback in
`run_harmonic_cycle`, evolve a quantum state, or demonstrate physical quantum
protection. Its claim level remains `classical_channel_verified`; an exact
Pauli/stabilizer fixture outcome is not a quantum-state simulation.

## Authority and data boundaries

`run_pcm_bridge` takes a received `PCM`, `ChannelConfig`, receiver observation,
`MeasuredSyndromeReference`, and a mandatory separately supplied
`expected_reference_sha256`. The receiver observation is replayed from the
actual PCM and explicit receiver/FEC policy. Every observation field is checked,
so writing `accepted: true` or recomputing an artifact hash cannot authorize a
correction.

The independent reference has exactly four measured symbols, alphabet size 3
or 4, a caller-declared fixture authority, and a source identifier. Symbols are
strict integers in the declared alphabet; booleans, floats and wrong widths are
rejected. The digest pin binds all these fields. A reference with a different
source, authority, or syndrome fails before any decoder call, even when its
own hash is internally consistent. On replay, the expected pin must be retained
outside the artifact being validated. Copying the artifact's own embedded pin
back as authority defeats this boundary.

The reference is caller-declared **test authority**, not authenticated physical
provenance. A consistently false reference plus a caller-approved matching pin
cannot be detected by hashes. This limitation is executable: a faithfully
recovered but incorrect measured syndrome can pass reference verification and
still fail the later exact-model logical-outcome test.

The stages remain separate:

- `ideal_syndrome`: computed only by the post-decode fixture evaluator from its
  separately supplied hidden Pauli error.
- `measured_syndrome`: the independent reference's declared measurement. It may
  differ from the ideal syndrome, including in explicit negative tests.
- `encoded_sound`: waveform produced by the source encoder, retained by the
  source/channel fixture. The bridge does not manufacture or inspect it.
- `received_waveform`: the actual PCM supplied to the receiver and bridge,
  including sample-level interference or distortion.
- `reference_noise`: an optional separately authorized cancellation input in
  the channel/cancellation experiment. The bridge neither infers it from hidden
  ideal data nor equates it with the received waveform.
- `estimated_syndrome`: raw PCM receiver recovery, retained even when the
  independent reference vetoes correction.
- `decoder_decision`: attempted/accepted flags, exact decoder identity, bounded
  error weight, actual decoder syndrome, correction, and reason.
- `logical_outcome`: later Pauli/stabilizer fixture evaluation, never fed back
  into candidate recovery or decoder selection.

## Fail-closed decision sequence

1. Validate the independent measured-reference type, values, alphabet, and
   externally supplied reference identity pin.
2. Replay the receiver observation from received PCM, channel configuration,
   and receiver/FEC policy. Reject malformed, changed, or rehashed evidence.
3. Retain raw receiver acceptance and recovered symbols. If receiver acceptance
   fails, do not call the decoder.
4. Compare the recovered symbols with the independently supplied measured
   reference. On mismatch, retain `candidate_false_accept: true` and veto the
   decoder. Independent verification does not retroactively turn this into a
   receiver rejection or hide the false-accept rate.
5. For verified recovery only, invoke the unchanged exact bounded-weight
   decoder. A verified syndrome outside its bounded table produces an attempted
   but rejected decision with no correction.
6. Optionally evaluate the exact-model residual in the separate fixture
   evaluator. Success requires an accepted correction **and stabilizer
   membership**, not only a zero residual syndrome.

`validate_bridge_receipt` repeats the complete calculation against caller-held
PCM, configuration, receiver policy, observation, independent reference, and
pin. It checks the complete result, including decoder/code identity and exact
correction. It does not use a receipt's self-hash as proof of source truth.

`validate_logical_outcome` also replays the complete fixture outcome against a
separately supplied bridge receipt, hidden error, and the same external bridge
inputs. It rejects rehashed changes to ideal/measured syndromes, correction,
residual, success/stabilizer/status fields, error identity, or claim boundary.
The bridge's code name is semantic identity; an evidence manifest must also
pin the source revision and decoder implementation bytes for reproducibility.

## Algebra and bounded scope

The qutrit bridge uses the existing cyclic `[[5,1,3]]_3` code and its GF(3)
Pauli/stabilizer algebra. The acoustic four-symbol syndrome is passed directly
to the existing exact decoder with maximum error weight one.

The ququart bridge uses the existing packed `[[5,1,3]]_4` code, which consists of
two binary Pauli lanes per physical ququart. Four acoustic digits label paired
checks: `s = 2*b0 + b1`. `unpack_ququart_syndrome` produces four lane-0 bits
followed by four lane-1 bits, and those eight binary checks reach the existing
exact decoder. These are **packed GF(2) × GF(2)** operations, not mod-4 Pauli
exponents. The correction artifact explicitly names its algebra.

The five-site bounded sets are exhaustively covered:

- Qutrit: identity plus `5 * (3^2 - 1)` single-site errors = 41 cases.
- Packed ququart: identity plus `5 * (4^2 - 1)` single-site errors = 76 cases.

Each case is encoded, perturbed at the PCM sample level, received without hidden
error access, independently checked, decoded, and tested for a stabilizer
residual. This establishes only the declared bounded software fixture result;
it does not establish logical-error improvements under arbitrary noise.

## Executed negative controls

Tests include:

- No-tone PCM rejected without a decoder call.
- Coherent per-block phase shifts by 16 samples for qutrits or 12 samples for
  ququarts in the 48-step alphabet. These preserve waveform consistency while
  changing the recovered valid symbol. They are raw receiver false accepts,
  independently vetoed before decoding.
- Forged/rehashed receiver observations; missing, serialized-as-trust, foreign
  source/authority, wrong-alphabet, and changed-syndrome references.
- PCM, configuration, and FEC-policy substitution against existing observations.
- Rehashed bridge-result mutations, including correction or reference changes.
- Legitimately recovered syndromes outside the bounded decoder table.
- A self-consistent false caller authority, which recovers faithfully yet fails
  the exact Pauli/stabilizer model test.
- Zero-syndrome non-stabilizer logical operators, which are never reported as
  successful merely because their syndrome vanishes.
- Strict reference validation, canonical deterministic replay, and explicit
  receiver/bridge signatures excluding hidden error inputs.

Run focused tests with the project's Python environment:

```sh
PYTHONPATH=src python -m pytest -q tests/acoustic/test_bridge.py
```

Canonical receipt construction triggers the repository's full-suite validation
law; focused tests alone are not the release gate. Broader channel batteries,
transport replay, release validation, and matched-resource comparisons remain
separate obligations.

## Minimal usage

The authority and expected pin below are assumed to have been independently
retained by the caller. They must not be imported solely from the observation
or bridge artifact under test.

```python
from qec.experiments.acoustic.bridge import run_pcm_bridge
from qec.experiments.acoustic.receiver import receive_pcm

observation = receive_pcm(received_waveform, config, 4, policy=policy)
result = run_pcm_bridge(
    received_waveform,
    config,
    observation,
    independently_retained_measured_reference,
    expected_reference_sha256=independently_retained_reference_pin,
    policy=policy,
)
```

The model-only `evaluate_logical_outcome` additionally accepts a hidden Pauli
error and the same external replay inputs. Its implementation validates the
bridge first, then computes the ideal syndrome and residual; none of those
hidden values reach the receiver or bridge.
