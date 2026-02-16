"""
Tests for QEC backend support (QuTiP and Qiskit).
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qec_steane import (
    create_steane_code, QISKIT_AVAILABLE,
    SteaneCode, SteaneCodeQiskit
)


def test_backend_factory_qutip():
    """Test creating SteaneCode with QuTiP backend."""
    code = create_steane_code('qutip')
    assert isinstance(code, SteaneCode)
    assert code.n_qubits == 7
    assert code.n_data == 1
    assert code.distance == 3


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_backend_factory_qiskit():
    """Test creating SteaneCode with Qiskit backend."""
    code = create_steane_code('qiskit')
    assert isinstance(code, SteaneCodeQiskit)
    assert code.n_qubits == 7
    assert code.n_data == 1
    assert code.distance == 3


def test_backend_factory_invalid():
    """Test that invalid backend raises error."""
    with pytest.raises(ValueError):
        create_steane_code('invalid_backend')


def test_qutip_encode_logical_zero():
    """Test QuTiP backend encodes logical zero."""
    code = create_steane_code('qutip')
    state = code.encode_logical_zero()
    
    # Check state is valid
    assert state.shape == (128, 1)  # 2^7 = 128
    assert state.type == 'ket'


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_qiskit_encode_logical_zero():
    """Test Qiskit backend encodes logical zero."""
    code = create_steane_code('qiskit')
    state = code.encode_logical_zero()
    
    # Check state is valid Statevector
    assert len(state) == 128  # 2^7 = 128


def test_qutip_depolarizing_noise():
    """Test QuTiP backend applies depolarizing noise."""
    code = create_steane_code('qutip')
    state = code.encode_logical_zero()
    noisy_state = code.apply_depolarizing_noise(state, 0.01)
    
    # Check returns density matrix
    assert noisy_state.type == 'oper'
    assert noisy_state.shape == (128, 128)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_qiskit_depolarizing_noise():
    """Test Qiskit backend applies depolarizing noise."""
    code = create_steane_code('qiskit')
    state = code.encode_logical_zero()
    noisy_state = code.apply_depolarizing_noise(state, 0.01)
    
    # Check returns DensityMatrix
    from qiskit.quantum_info import DensityMatrix
    assert isinstance(noisy_state, DensityMatrix)


def test_qutip_pauli_spectrum():
    """Test QuTiP Pauli spectrum computation."""
    code = create_steane_code('qutip')
    state = code.encode_logical_zero()
    spectrum = code.compute_pauli_spectrum(state)
    
    # Check spectrum has expected keys
    assert 'X_0' in spectrum
    assert 'Y_0' in spectrum
    assert 'Z_0' in spectrum
    assert len(spectrum) == 21  # 3 Paulis * 7 qubits


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_qiskit_pauli_spectrum():
    """Test Qiskit Pauli spectrum computation."""
    code = create_steane_code('qiskit')
    state = code.encode_logical_zero()
    spectrum = code.compute_pauli_spectrum(state)
    
    # Check spectrum has expected keys
    assert 'X_0' in spectrum
    assert 'Y_0' in spectrum
    assert 'Z_0' in spectrum
    assert len(spectrum) == 21  # 3 Paulis * 7 qubits


def test_qutip_logical_error_rate():
    """Test QuTiP logical error rate calculation."""
    code = create_steane_code('qutip')
    p_log = code.calculate_logical_error_rate(0.001, n_trials=10)
    
    # Check error rate is reasonable
    assert 0 <= p_log <= 1


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_qiskit_logical_error_rate():
    """Test Qiskit logical error rate calculation."""
    code = create_steane_code('qiskit')
    p_log = code.calculate_logical_error_rate(0.001, n_trials=10)
    
    # Check error rate is reasonable
    assert 0 <= p_log <= 1


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_backends_comparable():
    """Test that both backends produce comparable results."""
    qutip_code = create_steane_code('qutip')
    qiskit_code = create_steane_code('qiskit')
    
    # Both should have same basic properties
    assert qutip_code.n_qubits == qiskit_code.n_qubits
    assert qutip_code.n_data == qiskit_code.n_data
    assert qutip_code.distance == qiskit_code.distance
    assert qutip_code.theoretical_threshold == qiskit_code.theoretical_threshold


def test_default_backend():
    """Test that default backend is QuTiP."""
    code = create_steane_code()
    assert isinstance(code, SteaneCode)
