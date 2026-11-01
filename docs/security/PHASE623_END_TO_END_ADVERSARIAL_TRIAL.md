# Phase 623 — Actual IRC Transport Adversarial Trial

## Purpose

Phases 620–622 established signed experiment/capability/firewall composition,
a mandatory gate for a new non-sensitive IRC observation action, and a
bounded, one-use revocation ledger. This phase tests their **composition
through real loopback sockets**, not just internal function results. Existing
runtime, historic receipts and decoder code remain unchanged.

## Experiment and observations

`tests/test_phase623_end_to_end.py` boots the actual `IRCServer` on
`127.0.0.1` with an OS-selected ephemeral port. It connects two genuine TCP
peers, registers both, sends IRC bytes over their writers, checks the exact
bytes read by each peer, and inspects the local ledger's admission/denial chain.

Test inputs reuse Phase 620's genuine, ephemeral Ed25519-signed UFF evidence
and issued capability grants. The independently verifier-pinned contexts and
issuer keys are injected by the server operator, never extracted from the
IRC messages. No actual remote service, permanent key, cloud API, real-world
operator identity, physical QEC hardware or production grant is involved.

### Attack scenarios

| Case | Expected observed outcome |
| --- | --- |
| Valid signed and fully pinned operation | One exact fixed NOTICE, only to requester |
| Repeat after first ALLOW | No second wire NOTICE; `REPLAYED_OPERATION` |
| Valid alternate signed bundle reused with spent operation ID | No NOTICE, despite independent bundle validity |
| Individually valid foreign experiment/grant/receipt/firewall decision | `COMPOSITION_DENIED` and no output |
| Pre-use operation, signer, capability, or policy revocation | Only DENY; never emit protected NOTICE |
| Second IRC client or attempted channel broadcast | No protected data to either client |
| Extra arguments | No output and no operation consumption |
| Modified signed feature + producer's `verified=True` | Denied, not evidence of verification |
| Mutated verifier-selected firewall policy hash | Denied; original legitimate operation remains usable |
| Saturated ledger | Deny even with otherwise valid evidence, no receipt eviction |
| Receipt mutation or reordered history | Local chain verification fails |

Tests verify no channel membership changes or denied output where applicable.
The new trial adds no privileged external action. Its *only* protected effect
is the fixed non-sensitive IRC NOTICE from Phase 621.

## Per-run evidence, not checked-in scientific claims

`.github/workflows/phase623-adversarial.yml` installs the real cryptographic
backend, demonstrates an actual Ed25519 sign/verify operation, runs the
standalone trial suite, then separately validates its JUnit XML inventory.

`scripts/maintenance/verify_phase623_trial.py` requires the exact expected
eleven executed pytest cases spanning eight named scenario families. It
rejects missing, duplicate, skipped, errored, failed and unrelated tests.
Only after successful inventory verification does it emit a per-run JSON
record containing a commit SHA, JUnit SHA-256 and sorted test identifiers.

The CI workflow uploads the generated JUnit even on failure; it uploads
the verified inventory only on success. Both artifacts belong to their
specific Actions run and are **never committed as canonical source
evidence**. JUnit duration and bytes are nondeterministic; the digest
authenticates local bytes only, not independent custody or provenance.
Tests of the JUnit verifier use deliberately **synthetic XML**, clearly
separate from the real signed transport run.

## What Phase 623 does not prove

This trial does not enable operator login authentication, durable replay
across process restart, distributed consensus, background revocation, an
authenticated wall clock, hardware execution, quantum-information safety or
exactly-once network delivery. The Phase 622 ledger's 256-receipt saturation
behaviour is intentionally fail-closed and has an availability tradeoff.
The local chain is not a signed archival record.

## Review/acceptance gates

- Exact-head dedicated real-Ed25519 and two-client socket trial succeeds.
- Independent JUnit inventory verifier confirms all required tests executed.
- Full repository Python and container regression jobs succeed.
- Exact-head Codex code and security reviews report no actionable defects.
- Draft PR remains unmerged until human approval.
