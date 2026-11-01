# Phase 621 — Operational Trust Enforcement (bounded IRC observation)

## Entry point and scope

The real QEC IRC server's `IRCServer._dispatch` handles inbound `PRIVMSG` and
routes replies to socket writers. The new **`!trusted-observe`** command is
intercepted **before channel fan-out** and can emit precisely one fixed
requester-only `NOTICE`:

```
PRIVMSG qec-ircd :!trusted-observe
:qec-ircd NOTICE <registered-nick> :TRUSTED_OBSERVATION_ACCEPTED
```

This is a deliberately narrow, non-sensitive, read-only observation signal.
It neither executes a covenant transition nor changes physical or quantum
state; those are explicitly **outside this phase's enforcement coverage**.
Existing `!status`, `!help`, public IRC traffic and decoder paths are
unchanged. The unconfigured default **never** emits this new NOTICE.

## Mandatory preflight

Before creating any outbound NOTICE, the dispatcher calls
`authorize_trusted_observation` with an `TrustedObservationAuthority`
supplied at server construction time, **not from the IRC message**. The
authority holds:

- A verifier-owned `TrustedOperationContext` precommitted before the request;
- An exact nine-key candidate evidence bundle (experiment, capability grant
  and decision/receipt, capsule, covenant execution, audit report, firewall
  rules and decision);
- Separately supplied trusted signer keys, independently issued grant hash
  registry, and optional verifier-selected audit rules;
- The intended internal connection identifier.

The preflight calls the merged Phase 620
`evaluate_composed_evidence_trust` **on every attempt** and accepts only an
actual `ComposedTrustDecision.allowed is True`. No externally submitted
`verified` or `ALLOW` flags are trusted. It never constructs the expected
context from a claimant's candidate. Missing configuration, wrong connection,
invalid action, missing/extra candidate fields, exception, failed signature,
revocation, cross-run substitution or mismatched policy all return no output.

Any command to a channel (rather than directly to `qec-ircd`) is suppressed
before channel broadcasting, whether trusted or not. Extra arguments and
unauthorised clients are silent. A suppressed command does not queue a
protected NOTICE, forward a message, or change channel membership. If
operator replay auditing is separately enabled, routine input-reception audit
can still record an attempted command; this is **not** an action execution.

## Trust and nonclaims

The `client_id` is an internal routing identifier, **not an authenticated
human identity**. This demo must not be used to authorize sensitive output,
state writes, filesystem effects, network operations beyond the fixed NOTICE,
or privileged jobs without a separately authenticated session and externally
maintained authority. The operator is responsible for secure construction,
custody and storage of the trusted context, issuer registry, keys and rules.
A Python frozen dataclass does not freeze nested dictionaries, so the gate
recomputes trust every time. This phase does not implement anti-replay or
revocation persistence; those belong to later phases.

Tests reuse actual Ed25519-signed UFF positive controls from Phase 620.
They assert that approved invocation emits exactly the expected NOTICE and
that cross-operation evidence, revoked keys/grants, malformed candidates,
missing authority, wrong client, channel attempts and arguments produce
no accepted action. A local asyncio socket test tests the **real transport**
with no configured authority, rather than assuming an internal return value
proves a network outcome.

## Verification and review gate

Run the dedicated composition workflow (cryptography required), the full QEC
Python regression suite, and the full container regression. Exact-head Codex
code and security review is required before merging. Historical receipts,
canonical forms, legacy actions and protected decoder remain untouched.

Neither `IRCSERVER` nor the general covenant engine is globally locked down:
the **only enforced operation** is the new `!trusted-observe` command.
