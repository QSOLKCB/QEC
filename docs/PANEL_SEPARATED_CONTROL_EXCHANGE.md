<!-- SPDX-License-Identifier: MPL-2.0 -->
# Panel Separated-Control Exchange — v171.x

QEC v171.x implements Panel-style separated control as an additive deterministic routing contract above the unchanged v170.3.0 Strowger baseline.

```text
request intake
→ canonical digit register
→ sender route programme
→ motor/path actuation plan
→ independent route verification
→ commit or fail closed
```

## Release contracts

| Boundary | Contract |
|---|---|
| v171.0 | `qec.panel-topology.v1` — motor groups, banks, selector limits and path inventory |
| v171.1 | sealed `qec.panel-digit-register.v1`, `qec.panel-sender-program.v1`, sender/register binding |
| v171.2 | `qec.panel-route-receipt.v1` with separated request, payload, control, actuation and outcome identities |
| v171.3 | `qec.panel-translation-table.v1` with exact ordering and declared fallback policy |
| v171.4 | reproducible capacity/control-error battery and machine claim validation |
| v171.5 | offline `viz/panel/` visualisation and WebAudio demonstration |

The completed phase package version is `171.5.0`; artifacts keep their milestone contract versions.

## Artifact family

`qec-panel route` emits:

```text
panel_topology.json
panel_digit_register.json
panel_sender_program.json
panel_route_receipt.json
panel_fault_battery.json
panel_claim_validation.json
```

The route receipt embeds the exact translation table and the sender programme binds its full SHA-256. Primary identities are `panel_sender_register_receipt_hash` and `panel_separated_control_receipt_hash`.

## Invariants

The same complete request and translation table produce the same sender programme. A translation-table change changes the programme hash. Payload bytes are sealed in the register, carried unchanged and independently rechecked. The canonical event validator rejects actuation before `sender_program_sealed`. Multiple candidate paths require the explicit `first_declared_free_path` rule; otherwise construction fails.

The v171.4 battery covers a busy primary bank, all candidates unavailable, motor stall, translation-integrity mismatch and sender disagreement. Each case is declared and replayable; no ambient fallback is used.

`validate_route_receipt()` validates every nested identity, checks the event hash chain, reconstructs the sealed request, reruns the exchange with the recorded conditions and requires a byte-identical receipt.

`compare_strowger_panel()` checks declared destination and outcome equivalence while explicitly allowing different event traces. Full cross-era migration proof remains assigned to v176.x.

## Offline lab boundary

Open `viz/panel/index.html` directly from disk. Its record declares `qec.panel-browser-demonstration.v1`, `browser_demo_only: true` and `canonical_receipt: false`. It has no network dependency and is not canonical cryptographic evidence.

## CLI

```bash
qec-panel route --digits 2,3,4,11 --destination ququart/site-4/pauli-11 --output-dir artifacts/panel
qec-panel validate --receipt artifacts/panel/panel_route_receipt.json
```

Panel receipts prove deterministic software behaviour under their declared inputs and conditions. They do not prove physical telephone fidelity, hardware fault tolerance, decoder correctness merely because a route succeeded, physical truth or quantum advantage.
