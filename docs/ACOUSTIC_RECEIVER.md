<!-- SPDX-License-Identifier: MPL-2.0 -->
# v177.2 PCM receiver and classical repetition baseline

`qec.experiments.acoustic.receiver` consumes only noisy `PCM`, public
`ChannelConfig`, declared symbol count and frozen `ReceiverPolicy`. It receives
no ideal/measurement symbols, hidden error, source hash, noise plan or desired
answer. Length, sample clock and the complete harmonic residue classes must
match; no automatic timing recovery or oracle alignment occurs.

## Exact algorithm

Each fixed block projects onto the normative integer sine/cosine pair for each
harmonic. Every state template is generated from the public alphabet, never
from transmitted data. Squared projection distances determine the unique
nearest informative symbol, with separately declared distance and margin
gates. All informative orders must agree; partial orders constrain their
permitted state subset; state-dark orders must remain near their invariant.
An independent full-waveform residual-energy guard rejects excessive unexplained
energy. Ambiguous, weak, incomplete, discordant and state-dark-only observations
remain explicit rejections.

The optional `median_period` estimator takes a componentwise integer median of
per-period projections (the middle pair is floor-averaged for even counts),
then rescales by period count. The same residual guard is retained. This is a
bounded robust alternative, not a claim of state-of-the-art performance.

`fec='majority'` accepts strictly more than half of the transmitted repetition
word; `unanimous` requires all repetitions; `none` reads only the first. All
modes still transmit identical samples, energy, clock and latency, so discarding
redundancy does not silently receive a smaller channel budget. Repetition is a
conventional classical code with bounded correction capability. It corrects a
single wrong/erased block of three, but two coherent wrong blocks can be falsely
accepted. The default bank is harmonic redundancy plus repetition, not a
quantum code.

## Evidence and trust

An observation retains every per-order projection, distance, candidate,
confidence/rejection result, per-block decision and repetition vote. Complete
validation reconstructs it from independently supplied PCM/config/policy; an
outer self-hash alone is insufficient. `accepted` is raw receiver acceptance,
not an independent assertion that the original measured syndrome was recovered.
A coherent 16-sample qutrit or 12-sample ququart phase displacement can map a
valid state into another valid state without violating any harmonic check.
Those false accepts are retained before a separately trusted reference veto.

Reference subtraction is defined separately in
[the noise/cancellation contract](ACOUSTIC_NOISE_AND_CANCELLATION.md). The
later exact-decoder bridge must verify independently
before correction; it never inherits authority from a channel confidence score.

Run `python -m pytest -q tests/acoustic/test_receiver.py`. Coverage includes all
states/clocks/FEC policies, absence, erased/wrong repetitions, dark/parity-only
banks, order disagreement, coherent substitutions, robust-estimator negative
controls, strict policies, foreign input and rehashed forgery.
