# QEC v170.1.1 — Replication Receipts and Report-Claim Validation

## Purpose

v170.1.1 hardens the evidence layer introduced by v170.1.0. It does **not**
change the packed `[[5,1,3]]_4` construction, the bounded exact decoder, or the
v170.0.0 harmonic receiver. It upgrades how benchmark evidence is compared,
reported, and allowed to make scientific claims.

The release adds:

- exact finite FER oracles for all four declared physical-noise channels;
- an exact lane-exchange symmetry certificate;
- canonical independent-replication receipts;
- fail-closed machine validation of report claims;
- controlled finite-code vocabulary that does not misuse threshold language;
- separated receiver, decoder, and residual-failure telemetry;
- a machine-readable hardware-claim declaration contract;
- an expanded dependency-free GitHub Pages evidence view.

## Exact channel oracles

v170.1.0 provided the complete `16^5 = 1,048,576`-pattern oracle for the full
packed-depolarizing channel. v170.1.1 additionally enumerates each restricted
three-operator channel exactly:

- `lane0_only`: `XI`, `YI`, `ZI`;
- `lane1_only`: `IX`, `IY`, `IZ`;
- `same_pauli_correlated`: `XX`, `YY`, `ZZ`.

Each restricted alphabet contains identity plus three nonidentity operators,
so each exact restricted space contains `4^5 = 1,024` patterns. For a channel
with `m` nonidentity local operators,

\[
\mathrm{FER}_{\mathcal C}(p)
=
\sum_{w=0}^{5}
N_{\mathrm{fail},\mathcal C}(w)
\left(\frac{p}{m}\right)^w
(1-p)^{5-w}.
\]

The lane-0 and lane-1 exact weight enumerators must be identical. The generated
`lane_symmetry_certificate.json` hashes that invariant and fails generation if
the two lanes diverge.

## Replication receipts

A replication receipt records:

- target release, commit, and canonical release-manifest SHA-256;
- execution environment and declared parameters;
- source-document and source-manifest identities;
- artifact-by-artifact verification status;
- whether sampled artifacts are expected to differ because parameters differ.

Verification states are deliberately strict:

- `full_hash_match`: complete observed and release SHA-256 values are equal;
- `prefix_consistent`: a reported hash prefix agrees, but no full match is
  claimed;
- `parameter_variant_expected`: sampled artifacts use different declared trial
  counts and should not share the release hash;
- `unverified`: evidence is insufficient;
- `mismatch`: receipt generation fails.

The supplied qBraid report used seed `1701001`, 1,000 Monte Carlo trials per
cell, and 500 harmonic trials per cell. The canonical release run used 10,000
and 4,000 respectively. Its exact deterministic artifact prefixes agree with
the release artifacts, while parameter-bound sampled artifacts differ as
expected. Because only eight-character artifact prefixes were included in the
Markdown report, the receipt says `prefix_consistent`, not `full_hash_match`.

See:

```text
docs/replications/QBRAID_V170_1_0_REPLICATION.md
docs/replications/qbraid_v170_1_0_receipt.json
```

## Report-claim validation

Generated evidence now produces:

```text
report_claims.json
claim_validation.json
```

The validator derives counts from the evidence tables and rejects claims that
disagree with them. Enforced rules include:

- declared end-to-end cell count must equal the table row count;
- deterministic acceptance and rejection populations must be separated;
- zero-false-trust claims require zero receiver false-trust events;
- lane-symmetry claims require the exact symmetry certificate;
- artifact equality claims require complete 64-character SHA-256 values;
- an `all tests passed` claim requires a separate passed test receipt;
- threshold claims are forbidden for this single finite-code study;
- hardware claims require a complete hardware declaration contract.

External claims can be checked with:

```bash
qec-ququart-validate-report \
  --claims /path/to/report_claims.json \
  --evidence /path/to/generated/evidence
```

The accepted finite-code curve vocabulary is:

- `low_error_quadratic_regime`;
- `intermediate_error_regime`;
- `high_error_regime`;
- `finite_code_crossover`.

## Harmonic receiver telemetry

v170.1.1 separates layers that were previously easy to conflate:

- **receiver rejection** — harmonic observation is not trusted;
- **decoder rejection** — observation is trusted, but the syndrome is outside
  the bounded decoder table;
- **trusted correct syndrome** — receiver trusts the exact syndrome;
- **receiver false trust** — receiver trusts a syndrome different from the
  exact syndrome;
- **accepted logical residual** — the receiver supplied the correct syndrome,
  correction was attempted, but the physical error lies outside the guaranteed
  correction radius and leaves a logical residual.

An accepted logical residual is therefore not automatically called a harmonic
false accept. The generated `receiver_operating_curve.csv` reports these rates
separately across physical error probability and harmonic-noise sigma.

## Hardware-claim contract

A hardware claim is fail-closed unless all of the following are declared:

1. device specification;
2. circuit model;
3. leakage model;
4. timing model;
5. readout model.

The v170.1.1 methodology sets `hardware_claim` to `false`. This remains exact
finite code-capacity Pauli analysis plus classical harmonic-readout simulation.

## Commands

Generate the complete evidence bundle:

```bash
qec-ququart-bench \
  --output /tmp/ququart-fer \
  --trials 10000 \
  --harmonic-trials 4000 \
  --seed 1701001
```

Default output now includes:

```text
benchmark_manifest.json
methodology.json
report.js
exact_weight_enumerator.csv
exact_fer_curve.csv
exact_channel_weight_enumerator.csv
exact_channel_fer.csv
monte_carlo_fer.csv
harmonic_fault_matrix.csv
harmonic_end_to_end.csv
receiver_operating_curve.csv
lane_symmetry_certificate.json
report_claims.json
claim_validation.json
qbraid_replication_receipt.json
```

All generated identities use canonical JSON and SHA-256. Sampled artifacts are
bound to their seed and trial parameters. Exact artifacts are independently
enumerated and require no Monte Carlo interpretation.

## GitHub Pages

The report remains available under:

```text
https://qsolkcb.github.io/QEC/ququart-fer/
```

The browser now displays the exact curve for the selected channel, rather than
always overlaying the full packed curve on restricted-channel Monte Carlo data.
It also exposes lane symmetry, claim-validation state, replication status, and
disjoint receiver operating telemetry.

## Claim boundary

This release establishes exact finite behaviour under declared Pauli channels,
deterministic software replication receipts, and validated reporting rules. It
does not establish a hardware threshold, circuit-level fault-tolerance
threshold, break-even experiment, pulse fidelity, leakage or SPAM performance,
or universal quantum advantage.
