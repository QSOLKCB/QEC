"""
Tests for info-mass-gravity module.
"""

import pytest
import numpy as np
from qutip import basis, bell_state, tensor, Qobj
from src.info_mass_gravity import InfoMassGravity


def test_initialization():
    """Test InfoMassGravity initialization."""
    img = InfoMassGravity()
    
    assert img.info_mass_constant == 1.0
    assert img.gravity_constant == 1.0
    assert img.speed_of_info == 1.0


def test_von_neumann_entropy():
    """Test von Neumann entropy calculation."""
    img = InfoMassGravity()
    
    # Pure state should have zero entropy
    pure = basis(2, 0)
    entropy_pure = img.von_neumann_entropy(pure)
    assert entropy_pure == 0.0
    
    # Mixed state should have positive entropy
    mixed = 0.5 * basis(2, 0) * basis(2, 0).dag() + 0.5 * basis(2, 1) * basis(2, 1).dag()
    entropy_mixed = img.von_neumann_entropy(mixed)
    assert entropy_mixed > 0.0
    assert entropy_mixed <= 1.0  # Max entropy for qubit is 1 bit


def test_purity():
    """Test purity calculation."""
    img = InfoMassGravity()
    
    # Pure state should have purity 1
    pure = basis(2, 0)
    assert img.purity(pure) == 1.0
    
    # Maximally mixed state should have purity 0.5
    mixed = 0.5 * basis(2, 0) * basis(2, 0).dag() + 0.5 * basis(2, 1) * basis(2, 1).dag()
    purity_mixed = img.purity(mixed)
    assert 0.49 < purity_mixed < 0.51  # Account for numerical errors


def test_mutual_information():
    """Test mutual information calculation."""
    img = InfoMassGravity()
    
    # Product state should have zero mutual information
    product = tensor(basis(2, 0), basis(2, 0))
    mi_product = img.mutual_information(product, [0], [1])
    assert mi_product < 0.01  # Should be ~0
    
    # Bell state should have high mutual information
    bell = bell_state('00')
    mi_bell = img.mutual_information(bell, [0], [1])
    assert mi_bell > 0.5  # Highly correlated


def test_information_mass():
    """Test information mass calculation."""
    img = InfoMassGravity()
    
    # Pure state should have low mass
    pure = basis(2, 0)
    mass_pure = img.information_mass(pure)
    assert mass_pure >= 0.0
    assert mass_pure < 0.01  # Near zero for pure state
    
    # Mixed state should have higher mass
    mixed = 0.5 * basis(2, 0) * basis(2, 0).dag() + 0.5 * basis(2, 1) * basis(2, 1).dag()
    mass_mixed = img.information_mass(mixed)
    assert mass_mixed > mass_pure


def test_information_density():
    """Test information density calculation."""
    img = InfoMassGravity()
    
    mixed = 0.7 * basis(2, 0) * basis(2, 0).dag() + 0.3 * basis(2, 1) * basis(2, 1).dag()
    
    density1 = img.information_density(mixed, volume=1.0)
    density2 = img.information_density(mixed, volume=2.0)
    
    # Density should scale inversely with volume
    assert density2 < density1
    assert abs(density1 / density2 - 2.0) < 0.01


def test_entanglement_force():
    """Test entanglement force calculation."""
    img = InfoMassGravity()
    
    # Product state should have low force
    product = tensor(basis(2, 0), basis(2, 0))
    force_product = img.entanglement_force(product, [0], [1])
    assert force_product >= 0.0
    
    # Bell state should have higher force
    bell = bell_state('00')
    force_bell = img.entanglement_force(bell, [0], [1])
    assert force_bell > force_product


def test_fidelity():
    """Test quantum fidelity calculation."""
    img = InfoMassGravity()
    
    state1 = basis(2, 0)
    state2 = basis(2, 1)
    
    # Identical states should have fidelity 1
    assert img.fidelity(state1, state1) == 1.0
    
    # Orthogonal states should have fidelity 0
    fid_orth = img.fidelity(state1, state2)
    assert fid_orth < 0.01
    
    # Partial overlap
    state3 = (basis(2, 0) + basis(2, 1)).unit()
    fid_overlap = img.fidelity(state1, state3)
    assert 0.4 < fid_overlap < 0.6


def test_bures_distance():
    """Test Bures distance calculation."""
    img = InfoMassGravity()
    
    state1 = basis(2, 0)
    state2 = basis(2, 1)
    
    # Identical states should have distance 0
    assert img.bures_distance(state1, state1) < 0.01
    
    # Orthogonal states should have maximum distance
    dist = img.bures_distance(state1, state2)
    assert dist > 1.0


def test_trace_distance():
    """Test trace distance calculation."""
    img = InfoMassGravity()
    
    state1 = basis(2, 0)
    state2 = basis(2, 1)
    
    # Identical states should have distance 0
    assert img.trace_distance(state1, state1) < 0.01
    
    # Orthogonal pure states should have distance 1
    dist = img.trace_distance(state1, state2)
    assert 0.9 < dist <= 1.0


def test_information_profile():
    """Test information profile generation."""
    img = InfoMassGravity()
    
    state = 0.7 * basis(2, 0) * basis(2, 0).dag() + 0.3 * basis(2, 1) * basis(2, 1).dag()
    profile = img.information_profile(state)
    
    # Check all keys present
    assert 'entropy' in profile
    assert 'purity' in profile
    assert 'mass' in profile
    assert 'density' in profile
    assert 'potential' in profile
    
    # Check reasonable values
    assert 0.0 <= profile['entropy'] <= 1.0
    assert 0.0 <= profile['purity'] <= 1.0
    assert profile['mass'] >= 0.0


def test_entanglement_network():
    """Test entanglement network calculation."""
    img = InfoMassGravity()
    
    # Create 3-qubit state
    state = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
    network = img.entanglement_network(state, 3)
    
    # Check shape
    assert network.shape == (3, 3)
    
    # Check symmetry
    assert np.allclose(network, network.T)
    
    # Check diagonal is zero
    assert np.allclose(np.diag(network), 0)


def test_center_of_information_mass():
    """Test center of mass calculation."""
    img = InfoMassGravity()
    
    # Three states with different masses
    pure = basis(2, 0)
    mixed1 = 0.7 * basis(2, 0) * basis(2, 0).dag() + 0.3 * basis(2, 1) * basis(2, 1).dag()
    mixed2 = 0.5 * basis(2, 0) * basis(2, 0).dag() + 0.5 * basis(2, 1) * basis(2, 1).dag()
    
    states = [pure, mixed1, mixed2]
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    
    center = img.center_of_information_mass(states, positions)
    
    # Check shape
    assert center.shape == (3,)
    
    # Center should be between extremes
    assert 0.0 <= center[0] <= 2.0


def test_information_curvature():
    """Test information curvature calculation."""
    img = InfoMassGravity()
    
    states = [
        basis(2, 0),
        0.5 * basis(2, 0) * basis(2, 0).dag() + 0.5 * basis(2, 1) * basis(2, 1).dag()
    ]
    
    curvature = img.information_curvature(states)
    
    # Curvature should be finite and positive
    assert curvature >= 0.0
    assert np.isfinite(curvature)


def test_gravitational_potential_energy():
    """Test gravitational potential energy calculation."""
    img = InfoMassGravity()
    
    states = [
        basis(2, 0),
        0.7 * basis(2, 0) * basis(2, 0).dag() + 0.3 * basis(2, 1) * basis(2, 1).dag()
    ]
    
    positions = np.array([[0, 0, 0], [1, 0, 0]])
    energy = img.gravitational_potential_energy(states, positions)
    
    # Energy should be negative (attractive)
    assert energy <= 0.0


def test_relative_entropy():
    """Test relative entropy calculation."""
    img = InfoMassGravity()
    
    rho = basis(2, 0) * basis(2, 0).dag()
    sigma = 0.5 * basis(2, 0) * basis(2, 0).dag() + 0.5 * basis(2, 1) * basis(2, 1).dag()
    
    rel_entropy = img.relative_entropy(rho, sigma)
    
    # Relative entropy should be non-negative
    assert rel_entropy >= 0.0


def test_information_potential():
    """Test information potential calculation."""
    img = InfoMassGravity()
    
    state = 0.7 * basis(2, 0) * basis(2, 0).dag() + 0.3 * basis(2, 1) * basis(2, 1).dag()
    potential = img.information_potential(state)
    
    # Potential should be finite
    assert np.isfinite(potential)
