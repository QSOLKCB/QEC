<!-- SPDX-License-Identifier: MPL-2.0 -->
# ESS source-bound call processing (v173.1–v173.5 implementation)

Status: **implemented, unreleased**. This is a new `qec.ess-*.v2` schema family,
contract version `173.5`. It neither changes the published v173.0 skeleton API,
schemas, CLI nor regenerates its frozen fixtures. The package's release version
and tags are managed separately. Passing these tests does not constitute a
release or a clean-working-tree declaration.

## Design interpretation and phase mapping

This implementation interprets stored-program control as an immutable, source-bound
software controller over a genuine native ownership ledger. It makes no new
historical hardware claim. Historical material in
[the reference companion](roadmap/TELECOM_SWITCHING_REFERENCES.md) is motivation,
not proof input; the implementation does not emulate a particular deployed ESS.
The [safety companion](roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md) applies.

| Phase | Implemented contract | Generated evidence |
|---|---|---|
| v173.1 | `ProgramManifest`, immutable translations and actual installed source-byte hashes | `ess_program_store_manifest.json`, `ess_translation_table_manifest.json` |
| v173.2 | Exact attempt states and cumulative native reservations/releases | `ess_call_processing_receipt.json` |
| v173.3 | Logical ticks; release, retry, arrival fixed priorities and canonical tie-breaks | `ess_event_schedule_receipt.json` |
| v173.4 | One-hop forwarding, bounded native-rejection retries, independent fan-out | `ess_feature_module_manifest.json` |
| v173.5 | Full replay on two supported multistage Crossbar adapters and semantic comparison | `ess_fabric_equivalence_receipt.json` |

## Separate identities

`qec.routing.ess.programme` exposes:

- `TranslationTable(rows)`: sorted, unique dialed names mapped to one to four
  unique logical destinations. Their declared order is preserved.
- `FeatureModules(forwarding=(), retry_delays=(), max_fanout=1)`: explicit,
  immutable optional features with fixed authority limits.
- `ProgramManifest(translations, features=..., program_version="1",
  marker_id="ess-lifecycle-marker", max_search_evaluations=65536)`.
- `validate_program_manifest` with optional trusted manifest, source and
  translation hashes.

The programme contains hashes of the **complete bytes** of its two implementation
modules, the native Crossbar matrix, marker, path selector, contention and
continuity modules, and the canonical JSON implementation. Each source entry has
its stable package-relative path, byte length and full SHA-256. The source manifest,
explicit routing-logic manifest, translation table and feature manifest have
separate hashes, all bound into the programme identity. No label or source path
loads user-provided executable code.

Source replay checks these files against the installed implementation. A changed
source file, programme version, translation row, feature configuration or search
budget changes execution identity. A receipt from different installed source is
rejected rather than silently interpreted by new code. Preserve the matching
source revision for historical replay. The hashes identify source bytes, not
authenticated provenance; Python, standard-library and dependency/environment
identities belong in the separate release environment manifest.

Calls have their own sealed store, including payload bytes and caller-declared
decoder-output hashes. Changing call data does not change the programme identity.
Constructors detach nested data; instances are frozen, and serialization produces
fresh dictionaries/lists. The input binds programme, calls and adapter separately.

## Public lifecycle API

Import from `qec.routing.ess.lifecycle`; the old package-level exports are unchanged.

```python
from qec.routing.ess.programme import ProgramManifest, TranslationTable
from qec.routing.ess.lifecycle import (
    Call, FabricAdapter, LifecycleInput,
    execute_call_processing, validate_call_processing_receipt,
)

programme = ProgramManifest(TranslationTable((("dial", ("out",)),)))
adapter = FabricAdapter(
    fabric,  # validated native CrossbarFabric, including initial link states
    (("in", "ingress", "H000"),),
    (("out", "egress", "V001"),),
)
call = Call("call-a", "in", "dial", b"opaque-correction", "a" * 64,
            arrival_tick=0, hold_ticks=2)
source = LifecycleInput(programme, (call,), adapter)
receipt = execute_call_processing(source)
validation = validate_call_processing_receipt(
    receipt, expected_input_sha256=source.as_dict()["sha256"],
)
```

An endpoint mapping is `(logical_name, native_matrix_id, native_link_id)`. Sources
must exist on the first stage's horizontal axis and destinations on the last
stage's vertical axis. The manifest retains all native topology and state,
including initially busy, quarantined or unavailable resources. Unknown logical
call endpoints reject during analysis; they are never repaired silently.

### Exact attempt states and resource semantics

A successful branch attempt follows:

```text
intake → analyse → select → connect → verify → release
```

Translation or logical endpoint failure follows `intake → analyse → reject`.
Native route rejection follows `intake → analyse → select → reject`.
No other transition is allowed. A retry creates a **new attempt**, beginning at
`intake`; it cannot change a terminal rejected attempt into a connection.

- `intake` binds the sealed call identity.
- `analyse` records original dialed name, optional forwarding, translated target,
  translation-table hash and feature-manifest hash.
- `select` binds the exact native request with unchanged payload and decoder hash.
- `connect` is an actual atomic, whole-path **software reservation** performed by
  native v172.3 contention. It is not a physical connection/actuation claim.
- `verify` requires complete native replay, trusted input/programme/fabric pins,
  and independent v172.4 forward continuity evidence for every selected coordinate
  and wire. An adapter cannot self-certify.
- `release` executes the native owner/request-bound whole-path release at the
  declared hold expiry. All owned reservations must be released at batch end.
- `reject` preserves the native reason and blocked-resource inventory when a
  native attempt was made. No reservation, path or continuity witness is invented.

Each native command is appended to a cumulative contention batch. Prefixes replay
from the same immutable base with **every prior reservation and release**. They do
not plan independent snapshots. A second simultaneous call sees the first call's
owned resources as busy; a later call sees released resources as idle. Each prefix
is independently checked before its outcome is used. The final receipt embeds the
complete native contention/continuity source, including historical rejections and
release transitions. ESS events also bind the exact prefix receipt hashes.

The native shared search budget is cumulative across the entire batch, not reset
per attempt. Exhaustion remains a native rejection even if a feature requests a
retry. There are no native quarantine commands in this ESS controller; initial
fault snapshots are supported and native rejection evidence retains their effects.

## Logical scheduling and feature authority

Scheduling uses `(logical_tick, fixed_priority, call_id, branch, attempt)`.
Fixed priorities are `release=0`, `retry=1`, `arrival=2`. Equal-tick releases occur
before retries, and retries before new arrivals. Branch and attempt indices are
explicit. Inputs are canonical sorted unique call IDs; no wall clock, random seed,
thread interleaving or set iteration affects order.

Native contention ticks are monotonic **fabric-command ordinals** (`0..31`), not
a second physical clock. The semantic command stream binds every ordinal to its
ESS logical tick and schedule. This preserves exact ESS dispatch order without
allowing the native batch's request-hash tie-break to reorder equal-tick calls.
The schedule receipt records all dispatched events and its fixed policy. Its
validator replays native outcomes as well, because rejection determines retries
and reservation determines the later release event.

Feature limits are immutable and explicit:

- Forwarding performs one exact lookup before translation. Cycles, chains,
  self-forwarding and duplicate source rules are invalid. The forwarded dialed
  name must have a declared translation.
- Retry has zero to two positive logical delays, each measured from the previous
  attempt. Only a native rejection schedules retry; invalid translation or missing
  adapter endpoints do not. A success stops retries for that branch.
- Fan-out permits up to four ordered logical targets, within declared `max_fanout`.
  Branches are independent reservations, **not an atomic multicast group**. They
  may contend for the same source/resource and produce partial success. Retry may
  later succeed after another branch releases. No hidden serialization fallback
  or capacity multiplication is claimed.

All features preserve payload bytes and decoder identity. They only propose
requests to the same validating boundary. None can force acceptance, create
continuity evidence, override a busy resource, or extend the native budget.

## Evidence and equivalence

Each artifact declares schema/version and has a full canonical SHA-256. Canonical
JSON is UTF-8, sorted object keys, compact separators, exact integer types and no
floats. The event chain binds ordered state transitions; omitted, reordered,
substituted or rehashed false claims fail full reconstruction.

`validate_call_processing_receipt` supports external pins for input, programme,
adapter and call store. Without pins, replay proves self-consistency with embedded
inputs, not that those were the caller's intended inputs. Empty batches are valid
but explicitly have no native or connection evidence. Valid rejection is not a
successful connection.

`create_fabric_equivalence_receipt(left, right)` requires exactly equal programme
and call-store snapshots and two **different adapter manifests**. Both boundaries
are executed and independently verified. It compares actual semantic commands,
destination, payload/decoder hashes, attempt outcomes, exact native rejection
reasons, timing and release results. Native coordinates, resource identities and
internal paths remain separately visible in complete source receipts.

The supplied positive comparison uses a three-stage and a two-stage native
Crossbar fabric with explicit logical endpoint mappings. Internal paths differ
while the bounded call corpus produces equal semantic streams. A quarantined
wire on only one adapter is a negative control and yields `equivalent=false`.
The equivalence validator replays both complete sides; trusted left-input,
right-input and programme pins are available. A valid negative comparison has
`all_passed=true` for validation, while `equivalent=false`.

Supported scope is exactly the declared calls and two registered v172.3/v172.4
multistage Crossbar boundaries. Single-stage Crossbar has no compatible published
reservation lifecycle; Strowger, Panel, digital, packet, external protocols and
historical hardware are **not** silently advertised as supported here. Cross-era
migration is separate. Empty comparisons cannot claim nonempty equivalence.

## Bounds and failure behaviour

- At most eight calls; 4,096 payload bytes per call.
- At most 32 translations, forwarding rows, source mappings or destination mappings.
- At most four fan-out destinations and two retries per branch.
- `sum(branches * (1 + retry_count) * 2) <= 32` is required before execution.
  This conservative bound reserves space for both reservation and release even
  for attempts that eventually reject.
- Native topology remains within its published bounds; ESS additionally permits
  at most 256 matrix intersections and 256 interstage links.
- Logical ticks are exact integers in `0..2**31-1`; arrival plus every possible
  retry delay plus hold duration must fit. Hold durations are positive.
- IDs are valid UTF-8 strings of 1–128 characters. Hashes are full lower-case SHA-256.
- JSON preflight rejects over-deep, oversized and non-JSON inputs, including
  untrusted adapter results before native replay. Depth is at most 64 and the
  node budget is two million; individual strings and container sizes are bounded.
- One shared `MAX_JSON_BYTES = 256 * 1024 * 1024` (256 MiB) ceiling applies to
  canonical UTF-8 artifacts and actual CLI input-file bytes. Preflight counts
  object keys, escaping, quotes, punctuation and Unicode encoding exactly; a
  character count is not a byte limit. This accommodates the measured >105 MB
  paired receipt with maximum-size emoji identifiers, with headroom for JSON
  control-character escapes.
- The CLI reads at most the byte cap plus one byte from the actual binary stream,
  with no reliance on a pre-read file stat. Duplicate JSON keys are rejected.
  Its output has no appended newline, so the written bytes exactly match the
  canonical byte budget. No fallback schema is used.
- Every lifecycle and fabric-equivalence receipt is preflighted before returning,
  and every CLI artifact before writing. If generated evidence exceeds any output
  bound, execution fails with `INVALID_INPUT`; it does not return or persist an
  oversized receipt that its own validator/CLI cannot read. Bounds limit artifact
  admission as well as model input dimensions.

Malformed identities, unknown fields, inconsistent evidence and authority changes
raise `ValueError("INVALID_INPUT")`; the CLI exits nonzero. Native routing failures
produce valid explicit rejection receipts. Missing source files are an operational
failure; they never permit unverified execution.

## Reproduce and inspect

```sh
python -m qec.routing.ess.lifecycle_cli demo --output-dir artifacts/ess-lifecycle
python -m qec.routing.ess.lifecycle_cli run \
  --input artifacts/ess-lifecycle/ess_lifecycle_input.json \
  --output-dir artifacts/ess-lifecycle
python -m qec.routing.ess.lifecycle_cli validate \
  --receipt artifacts/ess-lifecycle/ess_call_processing_receipt.json
python -m qec.routing.ess.lifecycle_cli equivalence-demo --output-dir artifacts/ess-lifecycle
python -m pytest -q tests/ess
```

`tests/ess/test_ess_lifecycle.py` covers owned contention, release-before-retry
priority, all rejection boundaries, forwarding/fan-out partial outcomes, source
byte identities, maximum declared expansion, detached/frozen inputs, valid-evidence
substitution, rehashed semantic tampering, external pins, positive and negative
adapter comparisons, hash-seed independence and a frozen v2 fixture. The original
v173.0 fixture and test suite remain untouched. The full repository test suite is
the release gate and must be run separately against the exact final source commit.
