<!-- SPDX-License-Identifier: MPL-2.0 -->
# QEC v170.3.0 — Deterministic Strowger Syndrome Exchange

## Purpose

The Strowger Syndrome Exchange is a deterministic classical routing and
verification layer for QEC correction requests. It is inspired by automatic
electromechanical telephone exchanges: linefinders allocate incoming work,
selector stages decode pulse trains and hunt for the first free admissible
trunk, a two-axis connector resolves the final destination, and redundant tones
verify the selected route before commitment.

It does not replace an exact stabilizer decoder and does not claim that
telephone switching is quantum hardware.

Historical architecture reference:

- Mark Csele, “Telephone Switches”:
  <https://markcsele.ca/history/telephone-switches/>

## Exchange pipeline

```text
pending syndrome/correction request
    → first-free linefinder
    → mixed-radix pulse codec
    → hierarchical selector stages
    → deterministic first-free trunk hunting
    → vertical/rotary connector
    → route, check, and dark-reference tones
    → commit or fail closed
    → canonical hash-chained receipt
```

The exchange supports binary, ternary, radix-4, decimal, and general bounded
radices. A zero digit is represented by a full-radix pulse train, preserving the
historical dial convention while remaining explicit in the receipt.

## Optional Operator Desk

Automatic mode is the default and requires no operator.

The optional desk has three modes:

| Mode | Behaviour |
|---|---|
| `automatic` | No operator intervention is accepted. |
| `supervised` | Inspection, quarantine, release and replay actions may be recorded. |
| `manual` | Adds manual stepping and trunk seizure for laboratory demonstrations. |

Every intervention is appended to the same event hash chain as the automatic
switching events. The desk has no force-accept command, cannot alter decoder
outputs, and cannot silently remove evidence.

Example operator event:

```json
{
  "action": "operator_quarantine_trunk",
  "operator_id": "local-console",
  "target": "selector-2:3",
  "reason": "contact-bounce-detected"
}
```

## Tone verification

Each completed route derives three deterministic integer-frequency controls:

- route identity tone;
- independent check tone;
- state-dark contact/reference tone.

The exchange accepts only when every observed tone is within the configured
integer tolerance. Tone verification checks classical routing integrity. It
does not establish decoder correctness or physical quantum behaviour.

## Fault injection

The lab and Python API support deterministic:

- missed and duplicated pulses;
- stuck selectors;
- busy and quarantined trunks;
- route/check/reference tone offsets.

The same request, topology, initial trunk state, operator command sequence and
fault plan produce the same event chain and receipt SHA-256.

## CLI

```bash
qec-strowger route \
  --digits 2,3,4,11 \
  --radices 3,4,10,16 \
  --request-id syndrome-001 \
  --destination ququart/site-4/pauli-11 \
  --output artifacts/strowger-route.json

qec-strowger validate \
  --receipt artifacts/strowger-route.json
```

## Receipt boundary

Schema:

```text
qec.strowger-route-receipt.v1
```

The receipt binds topology, request identity, selected linefinder, selector
trunks, connector coordinates, expected and observed tones, injected faults,
operator commands, every state transition, and the final outcome.

It proves deterministic routing and the declared verification events. It does
not prove physical truth, quantum advantage, hardware fault tolerance, or that
the selected correction is mathematically valid. Mathematical validity remains
the responsibility of the decoder that originated the request.
