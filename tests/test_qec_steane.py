"""
Tests for QEC Steane code simulation module.
"""

import pytest
import numpy as np

pytest.importorskip("qutip")

from qec_steane import SteaneCode, ThresholdSimulation, SurfaceLattice


def test_steane_code_initialization():
    """Test Steane code initialization."""
    code = SteaneCode()
    assert code.n_qubits == 7
    assert code.n_data == 1
    assert code.distance == 3
    assert code.theoretical_threshold == 9.3e-5


def test_encode_logical_states():
    """Test encoding of logical states."""
    code = SteaneCode()
    
    logical_zero = code.encode_logical_zero()
    assert logical_zero is not None
    assert logical_zero.shape == (128, 1)  # 2^7 = 128
    
    logical_one = code.encode_logical_one()
    assert logical_one is not None
    assert logical_one.shape == (128, 1)
