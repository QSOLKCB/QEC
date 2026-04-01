# SPDX-License-Identifier: MIT
"""Tests for the Simulation Adapter Interface — v133.3.0."""

from __future__ import annotations

import math

import pytest

from qec.sims.adapter_interface import (
    SimulationAdapterPayload,
    normalize_for_backend,
    to_adapter_payload,
)
from qec.sims.universe_kernel import UniverseState


def _make_state(
    amplitudes: tuple = (1.0, 0.0, 0.0),
    qutrits: tuple = (0, 1, 2),
    timestep: int = 0,
) -> UniverseState:
    return UniverseState(
        field_amplitudes=amplitudes,
        qutrit_states=qutrits,
        timestep=timestep,
        law_name="test_law",
    )


# ── frozen immutability ──────────────────────────────────────────

class TestFrozenImmutability:
    def test_payload_is_frozen(self):
        p = to_adapter_payload(_make_state())
        with pytest.raises(AttributeError):
            p.state_vector = (9.9,)  # type: ignore[misc]

    def test_payload_fields_cannot_be_deleted(self):
        p = to_adapter_payload(_make_state())
        with pytest.raises(AttributeError):
            del p.timestep  # type: ignore[misc]


# ── tuple-only output ────────────────────────────────────────────

class TestTupleOutput:
    def test_state_vector_is_tuple(self):
        p = to_adapter_payload(_make_state())
        assert isinstance(p.state_vector, tuple)

    def test_qutrit_register_is_tuple(self):
        p = to_adapter_payload(_make_state())
        assert isinstance(p.qutrit_register, tuple)


# ── backend tagging ──────────────────────────────────────────────

class TestBackendTagging:
    @pytest.mark.parametrize("backend", ["generic", "qutip", "qiskit"])
    def test_valid_backends(self, backend: str):
        p = to_adapter_payload(_make_state(), backend_target=backend)
        assert p.backend_target == backend

    def test_default_backend_is_generic(self):
        p = to_adapter_payload(_make_state())
        assert p.backend_target == "generic"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend_target"):
            to_adapter_payload(_make_state(), backend_target="cirq")


# ── schema version ───────────────────────────────────────────────

class TestSchemaVersion:
    def test_schema_version_present(self):
        p = to_adapter_payload(_make_state())
        assert p.schema_version == "133.3.0"


# ── qutip normalization ─────────────────────────────────────────

class TestQutipNormalization:
    def test_unit_norm(self):
        state = _make_state(amplitudes=(3.0, 4.0))
        p = to_adapter_payload(state, backend_target="qutip")
        n = normalize_for_backend(p)
        norm = math.sqrt(sum(x * x for x in n.state_vector))
        assert abs(norm - 1.0) < 1e-12

    def test_zero_vector_unchanged(self):
        state = _make_state(amplitudes=(0.0, 0.0, 0.0))
        p = to_adapter_payload(state, backend_target="qutip")
        n = normalize_for_backend(p)
        assert n.state_vector == (0.0, 0.0, 0.0)

    def test_already_normalized(self):
        state = _make_state(amplitudes=(1.0, 0.0))
        p = to_adapter_payload(state, backend_target="qutip")
        n = normalize_for_backend(p)
        assert abs(n.state_vector[0] - 1.0) < 1e-12
        assert abs(n.state_vector[1] - 0.0) < 1e-12


# ── qiskit clipping ─────────────────────────────────────────────

class TestQiskitClipping:
    def test_values_in_range_unchanged(self):
        state = _make_state(qutrits=(0, 1, 2))
        p = to_adapter_payload(state, backend_target="qiskit")
        n = normalize_for_backend(p)
        assert n.qutrit_register == (0, 1, 2)

    def test_negative_clipped_to_zero(self):
        state = _make_state(qutrits=(-1, -5, 0))
        p = to_adapter_payload(state, backend_target="qiskit")
        n = normalize_for_backend(p)
        assert n.qutrit_register == (0, 0, 0)

    def test_overflow_clipped_to_two(self):
        state = _make_state(qutrits=(3, 7, 100))
        p = to_adapter_payload(state, backend_target="qiskit")
        n = normalize_for_backend(p)
        assert n.qutrit_register == (2, 2, 2)


# ── generic passthrough ─────────────────────────────────────────

class TestGenericPassthrough:
    def test_generic_returns_identical(self):
        p = to_adapter_payload(_make_state(), backend_target="generic")
        n = normalize_for_backend(p)
        assert n is p


# ── replay determinism ──────────────────────────────────────────

class TestReplayDeterminism:
    def test_repeated_export_equality(self):
        state = _make_state(amplitudes=(0.1, 0.2, 0.3), qutrits=(0, 1, 2))
        p1 = to_adapter_payload(state, backend_target="qutip")
        p2 = to_adapter_payload(state, backend_target="qutip")
        assert p1 == p2

    def test_repeated_normalization_equality(self):
        state = _make_state(amplitudes=(3.0, 4.0))
        p = to_adapter_payload(state, backend_target="qutip")
        n1 = normalize_for_backend(p)
        n2 = normalize_for_backend(p)
        assert n1 == n2

    def test_byte_identical_across_calls(self):
        state = _make_state()
        for backend in ("generic", "qutip", "qiskit"):
            p1 = to_adapter_payload(state, backend_target=backend)
            p2 = to_adapter_payload(state, backend_target=backend)
            n1 = normalize_for_backend(p1)
            n2 = normalize_for_backend(p2)
            assert n1 == n2
