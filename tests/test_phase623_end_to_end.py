"""Phase 623: adversarial trials through the actual two-client IRC socket path.

Unlike Phase 620's composition unit checks and Phase 622's ledger tests, each
trial submits protocol bytes to a bound loopback server and inspects wire bytes
and an independently checked local receipt chain. No live operator credentials
or externally contacted services are involved. The imported signed fixture
contains actual ephemeral Ed25519 keys and issued test grants.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import hashlib

import pytest

from qec.operator.irc_server import IRCServer
from qec.operator.observation_replay_ledger import (
    ObservationReplayLedger, verify_observation_gate_chain,
)
from qec.operator.trusted_observation import authorize_trusted_observation
from qec.orchestration.evidence_trust_composition import (
    evaluate_composed_evidence_trust,
)
from qec.orchestration.governance_memory_io_boundary_auditor import _default_boundary_rules
from test_evidence_trust_composition import (
    _CANDIDATE_FIELDS, _trusted_observe_authority, valid_bundles,
)


_COMMAND = "PRIVMSG qec-ircd :!trusted-observe"
_ACCEPT = b":qec-ircd NOTICE alice :TRUSTED_OBSERVATION_ACCEPTED\r\n"
_WAIT = 2.0
_NO_DATA = 0.12


class LiveTrial:
    """No socket mocking, no shell, all peers local. No production secrets."""

    def __init__(self, bundle, *, ledger=None, authority=None):
        self.ledger = ledger if ledger is not None else ObservationReplayLedger()
        self.server = IRCServer(
            host="127.0.0.1",
            port=0,
            trusted_observation=(
                _trusted_observe_authority(bundle, client_id="client-0001")
                if authority is None else authority
            ),
            observation_ledger=self.ledger,
        )
        self.readers = []
        self.writers = []

    async def __aenter__(self):
        await self.server.start()
        try:
            port = self.server._server.sockets[0].getsockname()[1]
            for nick in ("alice", "bob"):
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", port), _WAIT
                )
                self.readers.append(reader)
                self.writers.append(writer)
                writer.write(f"NICK {nick}\r\nUSER u 0 * :r\r\n".encode("ascii"))
                await writer.drain()
                welcome = await self.read_line(len(self.readers) - 1)
                assert b" 001 " in welcome and nick.encode("ascii") in welcome
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        for writer in self.writers:
            writer.close()
        for writer in self.writers:
            await writer.wait_closed()
        await self.server.stop()

    async def send(self, peer, line):
        writer = self.writers[peer]
        writer.write((line + "\r\n").encode("utf-8"))
        await writer.drain()

    async def read_line(self, peer):
        return await asyncio.wait_for(self.readers[peer].readline(), _WAIT)

    async def expect_silence(self, *peers):
        for peer in peers:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(self.readers[peer].readline(), _NO_DATA)

    async def join_both(self):
        for peer in (0, 1):
            await self.send(peer, "JOIN #qec")
            joined = await self.read_line(peer)
            assert b" JOIN #qec\r\n" in joined

    def check_receipts(self, reasons):
        receipts = self.ledger.receipts
        assert tuple(r.reason for r in receipts) == tuple(reasons)
        assert tuple(r.sequence for r in receipts) == tuple(
            range(1, len(reasons) + 1)
        )
        assert verify_observation_gate_chain(receipts)
        return receipts


def test_phase623_wire_allow_once_replay_denied_and_receipts(valid_bundles):
    a, _ = valid_bundles
    assert evaluate_composed_evidence_trust(**a).allowed

    async def trial():
        async with LiveTrial(a) as live:
            await live.send(0, _COMMAND)
            assert await live.read_line(0) == _ACCEPT
            await live.expect_silence(1)
            await live.send(0, _COMMAND)
            await live.expect_silence(0, 1)
            receipts = live.check_receipts((
                "COMPOSED_TRUST_ACCEPTED", "REPLAYED_OPERATION"
            ))
            assert {r.operation_digest for r in receipts} == {
                hashlib.sha256(a["context"].operation_id.encode()).hexdigest()
            }
            assert a["context"].operation_id not in repr(receipts)
            assert "signer-alpha" not in repr(receipts)
    asyncio.run(trial())


def test_phase623_wire_transplant_individually_valid_bundle_is_denied(valid_bundles):
    a, b = valid_bundles
    assert evaluate_composed_evidence_trust(**a).allowed
    assert evaluate_composed_evidence_trust(**b).allowed

    async def trial():
        async with LiveTrial(a) as live:
            # Each replacement is genuine evidence, but belongs to B.
            for field in ("experiment", "grant", "privilege_decision",
                          "privilege_receipt", "firewall_decision"):
                foreign = {key: a[key] for key in _CANDIDATE_FIELDS}
                foreign[field] = b[field]
                live.server.install_trusted_observation(
                    _trusted_observe_authority(
                        a, client_id="client-0001", candidate=foreign,
                    )
                )
                await live.send(0, _COMMAND)
                await live.expect_silence(0, 1)
            live.check_receipts(("COMPOSITION_DENIED",) * 5)
            # Five denied actions must not consume an otherwise valid grant.
            live.server.install_trusted_observation(
                _trusted_observe_authority(a, client_id="client-0001")
            )
            await live.send(0, _COMMAND)
            assert await live.read_line(0) == _ACCEPT
            await live.expect_silence(1)
            live.check_receipts(("COMPOSITION_DENIED",) * 5 + (
                "COMPOSED_TRUST_ACCEPTED",
            ))
    asyncio.run(trial())


def test_phase623_wire_reused_operation_identity_denied_after_rotation(valid_bundles):
    a, b = valid_bundles

    async def trial():
        async with LiveTrial(a) as live:
            await live.send(0, _COMMAND)
            assert await live.read_line(0) == _ACCEPT
            changed = replace(
                b["context"], operation_id=a["context"].operation_id
            )
            alternate = _trusted_observe_authority(
                b, client_id="client-0001", context=changed,
            )
            assert authorize_trusted_observation(alternate, "client-0001")
            live.server.install_trusted_observation(alternate)
            await live.send(0, _COMMAND)
            await live.expect_silence(0, 1)
            # A genuinely new operation ID from B is valid once.
            live.server.install_trusted_observation(
                _trusted_observe_authority(b, client_id="client-0001")
            )
            await live.send(0, _COMMAND)
            assert await live.read_line(0) == _ACCEPT
            await live.send(0, _COMMAND)
            await live.expect_silence(0, 1)
            live.check_receipts((
                "COMPOSED_TRUST_ACCEPTED", "REPLAYED_OPERATION",
                "COMPOSED_TRUST_ACCEPTED", "REPLAYED_OPERATION",
            ))
    asyncio.run(trial())


@pytest.mark.parametrize("dimension,reason", (
    ("operation", "REVOKED_OPERATION"),
    ("signer", "REVOKED_SIGNER"),
    ("capability", "REVOKED_CAPABILITY"),
    ("policy", "REVOKED_POLICY"),
))
def test_phase623_wire_each_revocation_denies_before_any_output(
    valid_bundles, dimension, reason,
):
    a, _ = valid_bundles
    ledger = ObservationReplayLedger()
    context = a["context"]
    if dimension == "operation":
        ledger.revoke_operation(context.operation_id)
    elif dimension == "signer":
        ledger.revoke_signer(context.trusted_signer_id)
    elif dimension == "capability":
        ledger.revoke_capability(context.capability_hash)
    else:
        ledger.revoke_policy(context.firewall_policy_rules_hash)

    async def trial():
        async with LiveTrial(a, ledger=ledger) as live:
            for _ in range(2):
                await live.send(0, _COMMAND)
                await live.expect_silence(0, 1)
            live.check_receipts((reason, reason))
            assert len(live.ledger.receipts) == 2
    asyncio.run(trial())


def test_phase623_wire_wrong_client_and_channel_have_zero_protected_effect(valid_bundles):
    a, _ = valid_bundles

    async def trial():
        async with LiveTrial(a) as live:
            await live.join_both()
            # Channel injection is suppressed before normal IRC fan-out.
            await live.send(0, "PRIVMSG #qec :!trusted-observe")
            await live.expect_silence(0, 1)
            # A second logged-in IRC connection has no authority.
            await live.send(1, _COMMAND)
            await live.expect_silence(0, 1)
            # A malformed request cannot consume the legitimate operation ID.
            await live.send(0, _COMMAND + " extra")
            await live.expect_silence(0, 1)
            live.check_receipts(("WRONG_CLIENT",))
            await live.send(0, _COMMAND)
            assert await live.read_line(0) == _ACCEPT
            await live.expect_silence(1)
            live.check_receipts(("WRONG_CLIENT", "COMPOSED_TRUST_ACCEPTED"))
    asyncio.run(trial())


def test_phase623_wire_tampered_proof_and_policy_fail_closed_then_valid(valid_bundles):
    a, b = valid_bundles

    async def trial():
        async with LiveTrial(a) as live:
            forged = {key: a[key] for key in _CANDIDATE_FIELDS}
            experiment = copy.deepcopy(a["experiment"])
            experiment["features"]["energy"] = -12345.0
            experiment["proof"]["verified"] = True
            forged["experiment"] = experiment
            live.server.install_trusted_observation(
                _trusted_observe_authority(
                    a, client_id="client-0001", candidate=forged,
                )
            )
            await live.send(0, _COMMAND)
            await live.expect_silence(0, 1)

            changed = replace(
                a["context"],
                firewall_policy_rules_hash=b["context"].firewall_policy_rules_hash,
            )
            live.server.install_trusted_observation(
                _trusted_observe_authority(
                    a, client_id="client-0001", context=changed,
                )
            )
            await live.send(0, _COMMAND)
            await live.expect_silence(0, 1)
            live.check_receipts(("COMPOSITION_DENIED", "COMPOSITION_DENIED"))
            live.server.install_trusted_observation(
                _trusted_observe_authority(a, client_id="client-0001")
            )
            await live.send(0, _COMMAND)
            assert await live.read_line(0) == _ACCEPT
            await live.expect_silence(1)
            live.check_receipts((
                "COMPOSITION_DENIED", "COMPOSITION_DENIED",
                "COMPOSED_TRUST_ACCEPTED",
            ))
    asyncio.run(trial())


def test_phase623_wire_saturated_ledger_blocks_otherwise_valid_action(valid_bundles):
    a, b = valid_bundles
    ledger = ObservationReplayLedger(max_receipts=1)

    async def trial():
        async with LiveTrial(a, ledger=ledger) as live:
            await live.send(1, _COMMAND)
            await live.expect_silence(0, 1)
            live.check_receipts(("WRONG_CLIENT",))
            live.server.install_trusted_observation(
                _trusted_observe_authority(b, client_id="client-0001")
            )
            await live.send(0, _COMMAND)
            await live.expect_silence(0, 1)
            assert ledger.saturated
            assert len(ledger.receipts) == 1
            live.check_receipts(("WRONG_CLIENT",))
    asyncio.run(trial())


def test_phase623_ledger_chain_independently_rejects_tampering(valid_bundles):
    a, _ = valid_bundles

    async def trial():
        async with LiveTrial(a) as live:
            await live.send(0, _COMMAND)
            assert await live.read_line(0) == _ACCEPT
            await live.send(0, _COMMAND)
            await live.expect_silence(0)
            receipts = live.check_receipts((
                "COMPOSED_TRUST_ACCEPTED", "REPLAYED_OPERATION"
            ))
            assert not verify_observation_gate_chain((
                receipts[1], receipts[0],
            ))
            assert not verify_observation_gate_chain((
                receipts[0], replace(receipts[1], reason="FAKE_ALLOW"),
            ))
    asyncio.run(trial())
