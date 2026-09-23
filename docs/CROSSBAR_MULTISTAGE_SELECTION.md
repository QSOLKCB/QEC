<!-- SPDX-License-Identifier: MPL-2.0 -->
# QEC v172.2 — Multi-Stage Link Selection

## Contract

v172.2 selects the first admissible **complete path plan** through an immutable
layered Crossbar fabric. It adds separate schemas and APIs; the published
v172.0 matrix and v172.1 exact-coordinate marker contracts remain unchanged.

A fabric has two or more ordered stages. Each stage contains one or more
v172.0 matrices. Explicit directed interstage links connect a vertical output
of a matrix to a horizontal input of a matrix in the immediately following
stage. Disconnected fabrics are valid inputs and produce rejection receipts.
Unknown wiring endpoints, duplicate edges, cycles and skipped stages are invalid.

Every selected horizontal, vertical and interstage link must be `idle` in the
bound snapshot. `busy`, `quarantined` and `unavailable` block selection. Search
never changes those states, reserves a resource, actuates a coordinate or
commits a connection. Resource contention and transitions belong to v172.3;
a separate end-to-end continuity receipt belongs to v172.4.

## Inputs and canonical order

`CrossbarFabric(fabric_id, stages, interstage_links)` copies its inputs into
immutable records. Stage order is the declared physical-model order, retained
in the manifest. Within a stage, matrix IDs must be unique and lexicographically
sorted. Matrix IDs are also globally unique across the fabric. Each embedded
matrix passes the unchanged v172.0 replay validator.

`InterstageLink` binds a unique `link_id`, source matrix and vertical-link IDs,
target matrix and horizontal-link IDs, and an initial state. Link records must
already be sorted by:

1. source stage index;
2. source matrix position within its stage;
3. source vertical ordinal;
4. target matrix position within the next stage;
5. target horizontal ordinal;
6. link ID.

Two wires cannot name the same endpoint pair. Noncanonical input ordering is
rejected rather than silently repaired. Dictionaries provide lookups only;
they never determine traversal order.

`MultiStageRequest(source_matrix_id, destination_matrix_id, request)` embeds a
published v172.1 `CrossbarRequest`. Its horizontal endpoint belongs to the
source matrix in the first stage; its vertical endpoint belongs to the
destination matrix in the final stage. Payload bytes and the caller-declared
decoder-output SHA-256 retain the v172.1 bounds and semantics. The decoder
reference is opaque and does not establish mathematical validity or provenance.

## Look-ahead and first-path rule

The selector uses reverse dynamic programming, not enumeration of every path:

1. Seal the full register and verify the fabric-bound programme and budget.
2. Validate the requested source and destination endpoint locations.
3. Visit stages from last to first, matrices in canonical stage order, and
   vertical links in ordinal order. For each vertical, evaluate outgoing wires
   in canonical order using the next stage's already-computed suffix viability.
4. A final-stage vertical is viable only if it is the requested destination and
   idle. An earlier vertical is viable only if an idle interstage wire reaches
   an idle horizontal with a complete admissible suffix.
5. Visit horizontals in ordinal order. An idle horizontal chooses its first
   viable vertical; a non-idle horizontal has no admissible suffix.
6. Starting at the requested source, follow these choices forwards. At each
   step choose the first viable vertical, then its first admissible outgoing
   wire. Return exactly one coordinate per stage and one wire per boundary.
7. Emit the selection/rejection event and release the marker.

This defines a lexicographic first-complete-path rule under the coordinate and
wire orders above. A locally free branch whose downstream suffix is blocked
cannot win. No random choices, ambient topology, timing, external lookup,
operator override or silent fallback participate.

The demo has an ingress matrix, two alternative middle matrices and an egress
matrix. The first branch's exit wire is unavailable by default. Search selects
`entry-b` / `exit-b`; with all links idle it selects `entry-a` / `exit-a`.

## Bounds and failure outcomes

Structural bounds are 2–16 stages, 1–64 matrices per stage, at most 64 matrices
in total, 65,536 total intersections, and 4,096 interstage links. Each matrix
also retains its v172.0 axis/intersection limits. Fabric, matrix and wire names
use the v172.1 marker text bound: nonempty UTF-8 text of at most 4,096 code points.

`max_search_evaluations` is a caller-declared exact integer from 1 through
131,072, hash-bound into the programme. The default is 131,072. Each inspected
interstage wire, vertical and horizontal costs one evaluation; the budget is
checked before that evaluation. The full search cost is:

```text
interstage link count + sum(horizontal count + vertical count over all matrices)
```

Search visits the complete fabric, including disconnected branches. The
structural caps make the default budget sufficient. Canonical manifest
validation, event hashing and forward plan reconstruction are additional
bounded work, not evaluation-counter units. Path combinations are never
materialized. Search adjacency/choice storage scales with axis and wire counts;
embedded manifest size additionally scales with matrix intersections.

| Reason | Meaning | Complete search? |
|---|---|---|
| `first_admissible_complete_path` | Selected the canonical first complete idle path | Yes |
| `no_admissible_complete_path` | Exhaustive suffix analysis found no complete admissible path | Yes |
| `search_budget_exhausted` | The declared evaluation budget ended before full analysis | No |
| `unknown_source_matrix` | Source is absent from the first stage | No |
| `unknown_destination_matrix` | Destination is absent from the last stage | No |
| `unknown_source_horizontal_link` | Source horizontal is absent or has the wrong axis | No |
| `unknown_destination_vertical_link` | Destination vertical is absent or has the wrong axis | No |

Unknown endpoints use the precedence shown above and consume zero evaluations.
All rejections have `path_plan: null`, outcome `rejected`, and marker release.
Budget exhaustion never masquerades as proof that no route exists. Successful
selection has outcome `plan_selected`. Malformed contracts fail with
`ValueError("INVALID_INPUT")` and produce no receipt.

## Artifacts and replay

All new schemas use contract version `172.2`, canonical sorted-key compact
UTF-8 JSON, and SHA-256 excluding the artifact's own `sha256` field.

| Artifact | Schema |
|---|---|
| `crossbar_fabric_manifest.json` | `qec.crossbar-fabric-manifest.v1` |
| `crossbar_path_input_register.json` | `qec.crossbar-path-input-register.v1` |
| `crossbar_path_search_program.json` | `qec.crossbar-path-search-program.v1` |
| `crossbar_path_search_receipt.json` | `qec.crossbar-path-search-receipt.v1` |
| `crossbar_path_search_validation.json` | `qec.crossbar-path-search-validation.v1` |

The receipt embeds the complete fabric, register and programme; payload and
declared decoder identities; ordered evaluation events; selected plan or
rejection; counts; search-completion status; unchanged fabric identity; marker
release; and a machine-readable claim boundary. Its event-chain root binds the
fabric, register and programme hashes. Every event binds its logical index,
details and previous event hash. Indices do not claim physical timing.

`validate_path_search_receipt()` recompiles and reruns the entire search,
requiring exact canonical receipt equality. A correctly rehashed but partial,
non-first, unavailable or fabricated path is rejected. A valid rejection can
pass validation: `all_passed` describes replay validity, while
`first_complete_path_verified` is true only for a selected path.

The primary validation identity is `crossbar_path_search_receipt_hash`.
Optional `expected_fabric_sha256`, `expected_request_sha256` and
`expected_program_sha256` bind the result to trusted inputs. Without those
pins, replay establishes internal consistency rather than external provenance.
Validation JSON is derived output, not a substitute for running the validator.

## API and CLI

```python
from qec.routing.crossbar import (
    CrossbarRequest, MultiStageRequest, compile_path_program,
    demo_fabric, execute_path_program, validate_path_search_receipt,
)

fabric = demo_fabric().as_dict()
request = MultiStageRequest("ingress", "egress",
    CrossbarRequest("example", "H000", "V001", b"opaque-correction", "a" * 64))
program = compile_path_program(fabric, request)
receipt = execute_path_program(fabric, request, program)
validation = validate_path_search_receipt(receipt,
    expected_fabric_sha256=fabric["sha256"],
    expected_request_sha256=request.as_dict()["sha256"],
    expected_program_sha256=program["sha256"])
```

The decoder reference above is a synthetic caller-declared fixture.

```bash
qec-crossbar fabric-demo --output-dir artifacts/crossbar-fabric
qec-crossbar path-search \
  --fabric artifacts/crossbar-fabric/crossbar_fabric_manifest.json \
  --source-matrix-id ingress --destination-matrix-id egress \
  --request-id example --horizontal-link-id H000 --vertical-link-id V001 \
  --payload-file correction.bin \
  --decoder-output-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --output-dir artifacts/crossbar-path
qec-crossbar path-validate \
  --receipt artifacts/crossbar-path/crossbar_path_search_receipt.json
```

`fabric-demo --all-idle` makes both branches available. `path-search` accepts
`--max-search-evaluations` and `--marker-id`, and emits the four path artifacts.
It exits 0 on selection, 2 on a valid rejection, and 1 on invalid input or I/O
failure. `path-validate` accepts the three `--expected-…-sha256` pins; it exits
0 for a replay-valid selection or rejection, otherwise 1. New JSON inputs
reject duplicate keys. Existing matrix and marker commands are unchanged.

## Validation and claim boundary

Regression tests compare dynamic selection against an independent forward DFS
oracle across 256 availability assignments, including blocked intermediate and
endpoint links. They cover dead ends, first-path tie-breaking, all link states,
maximum stage count, structural bounds, exact budgets, canonical wiring,
immutable inputs, recomputed-hash tampering, event chains, trusted bindings,
subprocess/hash-seed determinism, frozen fixtures and CLI round-trips.

The contract establishes deterministic selection under a declared classical
software model. It does not establish an actuated connection, independent
continuity evidence, authenticated provenance, decoder correctness, physical
Crossbar fidelity, carrier-grade reliability or quantum advantage.
