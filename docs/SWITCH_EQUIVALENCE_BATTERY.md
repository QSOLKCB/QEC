<!-- SPDX-License-Identifier: MPL-2.0 -->
# QEC v172.5 — Strowger/Panel/Crossbar Equivalence Battery

## Contract and scope

v172.5 completes the planned v172.x implementation with a bounded shared corpus
and explicit adapters into the unchanged Strowger, Panel and Crossbar models.
`switch_equivalence_matrix.json` retains every native receipt and compares four
fields: requested destination, routing outcome, selected lane and reached
destination. Event traces and native commit semantics are deliberately separate.

The contract is a **four-destination, two-lane admission comparison**, not a
universal switching equivalence theorem. Every case starts from its declared
initial lane states in fresh model instances. There is no persistent contention,
reservation lifecycle comparison, live adapter, network access or decoder call.
The broader cross-era migration contracts remain assigned to v176.x.

## Explicit adapters

The adapter manifest seals the complete fixed Strowger configuration, Panel
topology and translation table, Crossbar fabric, destination mapping, lane and
state mappings, comparison fields, native outcome projections and claim boundary.

| Shared construct | Strowger v170.3.0 | Panel v171.x | Crossbar v172.2 / v172.4 |
|---|---|---|---|
| Destination d, 0–3 | Digits `(0, d//2, d%2)`, connector `(d//2, d%2)` | Exact digit translation to `correction/destination-d` | Ingress H000 to egress V00d |
| Lane 0 or 1 | Trunk 0 or 1 of the single `lane` selector | Bank 0 or 1, one path per destination | Wire `lane-0` or `lane-1` between two matrices |
| Idle lane | Free trunk | Available bank and paths | Idle wire |
| Busy lane | Busy trunk | Busy bank | Busy wire |
| Quarantined lane | Quarantined trunk | All four lane paths unavailable | Quarantined wire |
| Successful decision | Native `committed` | Native `committed` | Selected path with independent continuity witness |
| Capacity rejection | `all_trunks_busy` | `capacity_exhausted` | `no_admissible_complete_path` |

Panel has no native quarantine state. Its unavailable-path representation only
models the same initial admission exclusion. It does not claim equivalent
quarantine transitions or ownership. Crossbar success remains a verified plan;
the battery neither invents nor performs a Crossbar connection commit.

Both lane alternatives have the same destination. Successful native results must
select the first available lane. Achieved destinations are read from Strowger
connector coordinates, the selected Panel path and the final Crossbar coordinate,
not copied from the requested label. Complete native input records are checked
against the shared case and retained for replay.

## Corpus and deterministic bounds

`EquivalenceCase(case_id, destination, lane_states, payload=b"",
decoder_output_sha256="a"*64, fault="none")` owns immutable inputs. The synthetic
default decoder reference is caller-declared, not a decoder execution result.
`EquivalenceCorpus(cases)` requires nonempty, unique case IDs already sorted
lexicographically. Noncanonical input order is rejected, never silently repaired.

| Input | Bound |
|---|---|
| Cases | 1–64 |
| Case ID | Nonempty UTF-8 text, at most 128 code points |
| Destination | Exact integer 0–3; booleans/floats rejected |
| Lane states | Exactly two values from `idle`, `busy`, `quarantined` |
| Payload | Exact bytes, at most 4,096 per case and 65,536 per corpus |
| Decoder reference | Lowercase 64-character SHA-256 |
| Native fault | One fixed control below, or `none` |

Native topology sizes and digit radices are fixed. Crossbar uses two stages,
10 intersections and two wires; its complete search costs 11 evaluations. Normal
cases use the published 131,072 evaluation cap; the budget control uses 1. Corpus
construction, native receipt validation, hashing and replay are additional work.
All execution is bounded by these cases and fixed native models, with no route
combination enumeration, random choices or wall-clock-dependent decisions.

The default corpus contains **41 cases**:

- all four destinations under all nine pairs of initial lane states: 36 cases;
- five negative controls on idle lanes: Strowger selector fault, Strowger tone
  mismatch, Panel motor stall, Panel sender disagreement and Crossbar search
  budget exhaustion.

Native faults require both lanes idle so capacity cannot mask the injected fault.
Unsupported faults or arbitrary combinations are invalid input. Each native
fault remains identifiable; a budget-exhausted search never becomes a capacity
rejection or a proof that no route exists.

## Comparison and acceptance

Every result embeds native Strowger and Panel route receipts plus a v172.4
continuity receipt containing the complete Crossbar search receipt. Native
validators run before comparison. Each result also binds the case and adapter
hashes, preserves native outcomes/reasons and records all three pairwise comparisons.

The common projection maps native success to `route_available` and native
capacity rejection to `capacity_blocked`. Other reasons retain architecture-qualified
names, such as `strowger.tone_mismatch` or `crossbar.search_budget_exhausted`.
A failed attempt has no projected achieved destination or successful lane;
partial native selections remain visible in the embedded evidence.

A separate admission oracle checks the first idle lane, destination and exact
expected native reason without consulting the native selectors or event traces.
Each case passes only if request and supported payload bindings hold, all oracle
checks pass, and the observed equivalence matches the fixed fault expectation.
Expected differences cannot hide the wrong native fault or an unrelated failure.

The default battery therefore reports:

```text
case_count:                  41
equivalent_case_count:        36
expected_difference_count:    5
all_cases_equivalent:         false
battery_passed:               true
default_corpus_used:          true
```

`all_cases_equivalent` answers whether all three projections agree in every
supplied case. `battery_passed` answers whether every specified expectation passed,
including negative controls. A replay-valid regression can have `all_passed: false`.
Neither flag is a claim of universal equivalence.

Custom bounded corpora are supported. `default_corpus_used` is true only for the
exact frozen default corpus, including its identities and payloads. A passing
subset cannot claim full default coverage. Consumers requiring that battery
should pin its complete corpus hash.

## Payload and commit boundaries

The shared case envelope includes payload bytes and a declared decoder-output
identity. These fields must not imply capabilities missing from a native model:

- Strowger has neither a payload nor a decoder-reference field. Both remain in
  the outer case envelope; its native projection records null for those identities.
- Panel seals payload bytes and verifies their before/after hashes. It has no
  native decoder-output reference, so that projection remains null.
- Crossbar seals payload bytes and the caller-declared decoder reference. The
  native search and continuity evidence retain both identities.

The battery checks native payload preservation for **Panel and Crossbar only**.
It makes no three-way native payload-equivalence or decoder-correctness claim.
Likewise, `native_commit_present` is retained per architecture and excluded from
the route-decision comparison. Reservation lifecycle equivalence is not claimed.

## Artifacts and replay

All new schemas use contract version `172.5`, sorted-key compact UTF-8 canonical
JSON and SHA-256 excluding their own `sha256` field.

| Artifact | Schema |
|---|---|
| `switch_equivalence_corpus.json` | `qec.switch-equivalence-corpus.v1` |
| `switch_equivalence_adapter_manifest.json` | `qec.switch-equivalence-adapters.v1` |
| `switch_equivalence_matrix.json` | `qec.switch-equivalence-matrix.v1` |
| `switch_equivalence_validation.json` | `qec.switch-equivalence-validation.v1` |
| Embedded case / result | `qec.switch-equivalence-case.v1` / `qec.switch-equivalence-case-result.v1` |

Primary identity: `switch_equivalence_matrix_hash`.

`validate_equivalence_matrix()` rebuilds the fixed adapters, runs every case,
reconstructs all native evidence and comparisons, and requires exact canonical
receipt equality. Rehashed mutations, swapped valid native receipts, omitted or
reordered results, changed mappings and fabricated capability claims are rejected.
A valid native receipt is insufficient if it belongs to another case or snapshot.

Optional `expected_corpus_sha256` and `expected_adapter_sha256` pin validation to
trusted inputs. Without them, replay establishes internal consistency for the
embedded corpus. Validation reports are derived output, not proof authority.
No changes are made to published Strowger, Panel or v172.0–v172.4 contracts.

## API and CLI

```python
from qec.routing.equivalence import (
    demo_equivalence_corpus, run_equivalence_battery, validate_equivalence_matrix,
)

corpus = demo_equivalence_corpus()
matrix = run_equivalence_battery(corpus)
validation = validate_equivalence_matrix(matrix,
    expected_corpus_sha256=corpus.as_dict()["sha256"],
    expected_adapter_sha256=matrix["adapter_manifest"]["sha256"])
```

```bash
qec-crossbar equivalence-demo --output-dir artifacts/crossbar-equivalence
qec-crossbar equivalence \
  --corpus artifacts/crossbar-equivalence/switch_equivalence_corpus.json \
  --output-dir artifacts/crossbar-equivalence
qec-crossbar equivalence-validate \
  --matrix artifacts/crossbar-equivalence/switch_equivalence_matrix.json
```

The validator accepts `--expected-corpus-sha256` and `--expected-adapter-sha256`.
`equivalence` and `equivalence-validate` exit 0 for a passing battery, 2 for a
replay-valid failed expectation, and 1 for invalid input or I/O failure. Expected
negative controls do not cause exit 2. JSON inputs reject duplicate keys.

Tests cover the full availability grid, native replay, negative controls,
capability gaps, mutable input copying, strict types and bounds, rehashed and
substituted evidence, trusted bindings, subset coverage, regression detection,
hash-seed determinism, frozen corpus/output identities and installed CLI round-trips.

These results concern only declared deterministic software behaviour. They do
not establish physical switching fidelity, authenticated provenance, universal
architecture superiority, carrier-grade reliability or quantum advantage.
