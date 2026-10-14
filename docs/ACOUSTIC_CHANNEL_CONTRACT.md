<!-- SPDX-License-Identifier: MPL-2.0 -->
# v177.0–v177.1 source-bound acoustic channel

This additive, opt-in experiment implements a classical sampled-signal
contract. It does not modify v137/v166/v167/v170–v176 decoders, symbolic runtime,
transport schemas, package version or historical fixtures. It is neither a
quantum-state simulator nor evidence of physical quantum protection.

## Declared boundary

`qec.experiments.acoustic.channel` provides frozen `ChannelConfig`, `PCM` and
`AcousticSource` values, canonical JSON receipts, exact PCM/WAV byte bindings,
strict deserialization, and source-pinned complete replay. Every serialized
field is reconstructed; changing a field and recomputing the outer hash does
not make it valid. The caller's expected source-receipt hash must be obtained
independently. A hash identifies declared input; it does not authenticate a
microphone, experimenter, device, or measurement.

`AcousticSource.measured_symbols` is the **declared synthetic measurement** to
transmit. It is not a hidden-error oracle available to a candidate receiver.
There is no quantum logical outcome in this channel contract. Later bridge
contracts must keep ideal syndrome, measured syndrome, encoded sound, received
waveform, reference noise, estimated syndrome, decoder decision and logical
outcome separate.

## Alphabet, clock and rounding

- Mono signed PCM16 little-endian, at 24,000 or 48,000 samples/second.
- The normative 48-entry `SINE48` table is included and hash-bound. No runtime
  trigonometry, floating-point reduction or PRNG enters clean PCM generation.
- A symbol `s` of dimension `d` has phase `2*pi*s/d`; harmonic `h` reads phase
  `2*pi*h*s/d`. Sample `n` is the sum of
  `amplitude * SINE48[(h*(n + s*(48//d))) % 48] // 32767`.
  Each product uses integer floor division **before** summation. Maximum clean
  amplitude is bounded below int16 saturation; no silent clipping occurs.
- Each block has 1–8 complete periods; phase resets at each block. The public
  configuration declares 1–3 repeated blocks per symbol. Repetition is the
  classical redundancy; dark tones do not add state information.
- Carrier frequency is 500 or 1,000 Hz. These are explicit half-frequency
  carrier alternatives, not evidence that engineered subharmonics protect a
  quantum state. Orders 1–8 stay strictly below the PCM Nyquist boundary.
- At most 32 symbols are encoded. Empty source symbols, float/bool integers,
  duplicate or unsorted harmonics, unsupported rates and malformed PCM fail
  with `ValueError("INVALID_INPUT")`.

For qutrits, orders not divisible by three identify the symbol; multiples of
three are state-dark. For packed ququarts, odd orders identify, order 2 modulo
4 reveals only parity, and multiples of 4 are state-dark. This phase alphabet
is a classical four-label encoding; it does not replace the existing packed
GF(2) × GF(2) quantum Pauli/syndrome algebra with arithmetic modulo four.

The default `(1,2,3,4)` bank retains every residue class. Other banks are
accepted as experimental encoders, including deliberately uninformative
negative controls. Acceptance by an encoder is not receiver trust.

## Clean known answers and independent checks

`tests/acoustic/fixtures/clean_q3.{json,wav}` and `clean_q4.{json,wav}` freeze
all states at the default clock, amplitude and repetition budget. JSON retains
all PCM bytes; WAV hashes bind the exact 44-byte-header canonical WAV format.
The strict reader rejects extra chunks and alternate layouts rather than
silently interpreting them. Python's separate WAV reader checks the export;
an independent Fourier projection checks each state/harmonic phase. Production
replay is integer-only; this independent test's floating-point phase tolerance
is not part of canonical identity.

Run `python -m pytest -q tests/acoustic/test_channel.py` and the full
`python -m pytest -q` gate. Fresh-process `PYTHONHASHSEED` checks cover ordering.
Multi-OS execution must be evidenced separately by CI; a local Linux run is
not a cross-platform result.

## Scientific limits

A coherent phase shift can change one valid symbol to another while preserving
harmonic and dark checks. Repetition can repeat the same wrong state. These
checks therefore cannot authenticate a syndrome. A future receiver must retain
raw false acceptance before an independent reference veto, and only an
independently validated recovery may reach an exact decoder. No inference of
physical logical-qubit benefit follows from synthetic sound or canonical WAV
bytes. v178–v182 remain separately gated research proposals.
