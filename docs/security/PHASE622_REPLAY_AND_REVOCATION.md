# Phase 622 — One-use IRC observation and explicit revocation

## Executable boundary

Phase 621 introduced the `IRCServer._dispatch` gate for the non-sensitive,
requester-only `!trusted-observe` command. Phase 622 retains that exact
operation and fixed NOTICE format. Before emitting a NOTICE it now calls
`ObservationReplayLedger.admit`. The ledger runs the real Phase 620 composed
verification, atomically spends a precommitted operation ID before returning
ALLOW, and rejects later uses of that ID even when presented with another
independently valid signed bundle.

**No other IRC operation, protected decoder or covenant executor is modified.**

## Server-owned trust and revocation

The application, not IRC text, owns the ledger and the authority. Trusted
application code may call:

- `ledger.revoke_operation(operation_id)`
- `ledger.revoke_signer(signer_id)`
- `ledger.revoke_capability(capability_hash)`
- `ledger.revoke_policy(policy_rules_hash)`
- `server.install_trusted_observation(authority)` for deliberate rotation.

Revocations are monotonic for the lifetime of the ledger; there is **no
unrevoke** API. Replacing the server's authority does not clear revocations
or consumed IDs. A supplied, shared ledger object can retain that state across
multiple *server instances in the same process*, but this does not provide
durable storage over a process restart. A production service must separately
authenticate operators, retain the ledger in trusted storage, and reconcile
one-use semantics across processes before using the result to authorize any
sensitive or privileged action. A server-side internal client ID is **not**
cryptographic human identity.

## Deterministic refusal evidence

The ledger records bounded local `ObservationGateReceipt` values with
contiguous sequence, verdict, reason, SHA-256 digest of the operation ID,
previous-receipt hash and a canonical, non-self-referential receipt hash.
`verify_observation_gate_chain` validates every link.

Reasons include `COMPOSED_TRUST_ACCEPTED`, `REPLAYED_OPERATION`,
`REVOKED_OPERATION`, `REVOKED_SIGNER`, `REVOKED_CAPABILITY`,
`REVOKED_POLICY`, `COMPOSITION_DENIED`, `WRONG_CLIENT`, and
`INVALID_AUTHORITY`. None of these diagnostic events is a signature,
trusted timestamp or external custody proof. Raw operation identifiers,
signing keys, signed experiment contents and capability grants are never
included in the ledger receipts.

The ledger caps its receipts at 256. **On saturation the protected operation
always DENIES** and sets `saturated=True`; it does not evict earlier spent
IDs to make room, and it cannot append further receipt records. Consequently,
a full ledger cannot be described as a complete audit of rejected attempts;
an external trusted monitor should detect saturation and take operational
action. A high rate of rejected commands may exhaust the allowance and deny
future legitimate commands. The cap is an intentional availability tradeoff
in this bounded demonstration.

## Replay and revocation claims

- Executable positive: one valid, independently signed, verifier-pinned
  bundle yields exactly one fixed NOTICE.
- Executable negative: repeating the same precommitted operation ID yields
  **no second NOTICE**, even if the server installs a different otherwise
  valid bundle or reuses an earlier ALLOW receipt.
- Executable negative: revoking an operation, signer, capability or policy
  before first use suppresses all protected action effects.
- Executable negative: changing the precommitted signer/policy to a mismatched
  value fails the Phase 620 composition gate; a later restored legitimate
  operation may still execute if it was not already consumed.
- Concurrent callers contend on a single ledger lock, so the operation ID
  can be consumed at most once in this process.
- Tests include real loopback IRC socket allow-then-replay and a denial
  output check, in addition to internal dispatch and chain-tampering checks.

## Limits and gates

One-use is **per ledger instance**, not a global distributed replay defense.
There is no external clock, duration/expiry, durable revocation log, trusted
human identity, recovery protocol or guaranteed exactly-once network delivery.
An accepted operation is spent **before** the socket send, so transport failure
may lose its NOTICE; that is preferable to authorizing a duplicate. Explicit
revocation takes effect at the next admission, and cannot retract an already
queued or transmitted NOTICE. Calling the server's Python methods from other
trusted application code is outside IRC's threat boundary.

The existing `authorize_trusted_observation` remains a side-effect-free,
opt-in composition preflight; only the new IRC dispatcher gains one-use state.
Both the focused **real Ed25519** security workflow and the complete QEC
Python/container CI must pass on the exact candidate commit, followed by
Codex code and security review. Historical artifacts and canonical receipt
formats outside this new local diagnostic schema remain unchanged.
