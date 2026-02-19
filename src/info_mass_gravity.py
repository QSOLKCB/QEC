"""
Information-Mass-Gravity Module for QEC Simulations

This module provides physics-inspired metrics for quantum information:
- Information theory: von Neumann entropy, mutual information, relative entropy
- Mass analogies: information mass, density, center of mass
- Gravity analogies: entanglement attraction, information curvature
- Geometric interpretations: information distance, geodesics

These concepts provide intuitive visualizations and metrics for
quantum error correction and entanglement structure.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.linalg import logm


class InfoMassGravity:
    """
    Information-theoretic and physics-inspired metrics for quantum states.
    """
    
    def __init__(self):
        """Initialize info-mass-gravity calculator."""
        # Physical constants (dimensionless units)
        self.info_mass_constant = 1.0  # Maps entropy to "mass"
        self.gravity_constant = 1.0     # Coupling strength for entanglement
        self.speed_of_info = 1.0        # Maximum information propagation speed
        
    # ========== Information Theory Measures ==========
    
    def von_neumann_entropy(self, state: Qobj) -> float:
        """
        Calculate von Neumann entropy S = -Tr(ρ log ρ).
        
        Args:
            state: Quantum state (ket or density matrix)
            
        Returns:
            Von Neumann entropy (0 for pure states, >0 for mixed)
        """
        from qutip import entropy_vn
        if state.type == 'ket':
            # Pure state has zero entropy
            return 0.0
        return entropy_vn(state, base=2)  # Use base 2 for qubits
    
    def mutual_information(
        self,
        state: Qobj,
        subsystem_a: List[int],
        subsystem_b: List[int]
    ) -> float:
        """
        Calculate mutual information I(A:B) = S(A) + S(B) - S(AB).
        
        Measures correlations between subsystems.
        
        Args:
            state: Joint quantum state
            subsystem_a: Indices of subsystem A
            subsystem_b: Indices of subsystem B
            
        Returns:
            Mutual information (0 for product states, >0 for correlated)
        """
        from qutip import ptrace, entropy_vn
        # Ensure state is density matrix
        if state.type == 'ket':
            state = state * state.dag()

        # Calculate entropies
        rho_a = ptrace(state, subsystem_a)
        rho_b = ptrace(state, subsystem_b)

        s_a = entropy_vn(rho_a, base=2)
        s_b = entropy_vn(rho_b, base=2)
        s_ab = entropy_vn(state, base=2)
        
        # I(A:B) = S(A) + S(B) - S(AB)
        return s_a + s_b - s_ab
    
    def relative_entropy(self, rho: Qobj, sigma: Qobj) -> float:
        """
        Calculate relative entropy (quantum Kullback-Leibler divergence).
        
        S(ρ||σ) = Tr(ρ log ρ) - Tr(ρ log σ)
        
        Measures distinguishability between states.
        
        Args:
            rho: First density matrix
            sigma: Second density matrix
            
        Returns:
            Relative entropy (always >= 0)
        """
        # Ensure both are density matrices
        if rho.type == 'ket':
            rho = rho * rho.dag()
        if sigma.type == 'ket':
            sigma = sigma * sigma.dag()
        
        # Calculate relative entropy
        rho_mat = rho.full()
        sigma_mat = sigma.full()
        
        # S(ρ||σ) = Tr(ρ log ρ - ρ log σ)
        term1 = np.trace(rho_mat @ logm(rho_mat))
        term2 = np.trace(rho_mat @ logm(sigma_mat))
        
        return np.real(term1 - term2) / np.log(2)  # Convert to base 2
    
    def purity(self, state: Qobj) -> float:
        """
        Calculate purity P = Tr(ρ²).
        
        Args:
            state: Quantum state
            
        Returns:
            Purity (1 for pure states, <1 for mixed)
        """
        if state.type == 'ket':
            return 1.0
        rho = state.full()
        return np.real(np.trace(rho @ rho))
    
    # ========== Mass Analogies ==========
    
    def information_mass(self, state: Qobj) -> float:
        """
        Calculate "information mass" as entropy-weighted measure.
        
        m_info = k * S(ρ) * (1 - P(ρ))
        
        Higher entropy and lower purity → higher mass.
        
        Args:
            state: Quantum state
            
        Returns:
            Information mass (dimensionless)
        """
        entropy = self.von_neumann_entropy(state)
        purity = self.purity(state)
        
        # Mass scales with entropy and mixedness
        mass = self.info_mass_constant * entropy * (1 - purity + 0.01)
        return mass
    
    def information_density(self, state: Qobj, volume: float = 1.0) -> float:
        """
        Calculate information density ρ_info = m_info / V.
        
        Args:
            state: Quantum state
            volume: Effective Hilbert space volume
            
        Returns:
            Information density
        """
        mass = self.information_mass(state)
        return mass / volume
    
    def center_of_information_mass(
        self,
        states: List[Qobj],
        positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate center of mass for multiple quantum states.
        
        r_cm = Σ m_i * r_i / Σ m_i
        
        Args:
            states: List of quantum states
            positions: Position vectors (default: equally spaced on line)
            
        Returns:
            Center of mass position vector
        """
        n_states = len(states)
        
        # Default positions if not provided
        if positions is None:
            positions = np.array([[i, 0, 0] for i in range(n_states)])
        
        # Calculate masses
        masses = np.array([self.information_mass(s) for s in states])
        total_mass = np.sum(masses)
        
        if total_mass == 0:
            return np.mean(positions, axis=0)
        
        # Weighted average
        center = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
        return center
    
    # ========== Gravity Analogies ==========
    
    def entanglement_force(
        self,
        state: Qobj,
        subsystem_a: List[int],
        subsystem_b: List[int],
        distance: float = 1.0
    ) -> float:
        """
        Calculate "gravitational" force from entanglement.
        
        F_ent = G * m_A * m_B * I(A:B) / r²
        
        Stronger entanglement → stronger "attraction".
        
        Args:
            state: Joint quantum state
            subsystem_a: Indices of subsystem A
            subsystem_b: Indices of subsystem B
            distance: Separation distance
            
        Returns:
            Entanglement force (attractive, always >= 0)
        """
        from qutip import ptrace
        # Get mutual information (entanglement measure)
        mutual_info = self.mutual_information(state, subsystem_a, subsystem_b)

        # Get subsystem masses
        rho_a = ptrace(state, subsystem_a)
        rho_b = ptrace(state, subsystem_b)
        mass_a = self.information_mass(rho_a)
        mass_b = self.information_mass(rho_b)
        
        # Gravitational-like force law
        force = (self.gravity_constant * mass_a * mass_b * 
                (1 + mutual_info) / (distance ** 2 + 0.01))
        
        return force
    
    def information_potential(
        self,
        state: Qobj,
        reference_state: Optional[Qobj] = None
    ) -> float:
        """
        Calculate information potential energy.
        
        U = m_info * φ, where φ is relative entropy to reference.
        
        Args:
            state: Quantum state
            reference_state: Reference state (default: maximally mixed)
            
        Returns:
            Information potential energy
        """
        mass = self.information_mass(state)
        
        # Default reference: maximally mixed state
        if reference_state is None:
            from qutip import Qobj
            n_dims = state.shape[0]
            reference_state = Qobj(np.eye(n_dims) / n_dims)
        
        # Potential from relative entropy
        potential = self.relative_entropy(state, reference_state)
        
        return mass * potential
    
    def information_curvature(
        self,
        states: List[Qobj],
        positions: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate "spacetime curvature" from information density.
        
        R ≈ 8πG * ρ_info (Einstein-like equation)
        
        Args:
            states: List of quantum states
            positions: Spatial positions
            
        Returns:
            Information curvature (Ricci scalar analog)
        """
        # Calculate total information density
        total_density = 0.0
        for state in states:
            total_density += self.information_density(state)
        
        # Average density
        avg_density = total_density / len(states)
        
        # Curvature from density (Einstein equation analog)
        curvature = 8 * np.pi * self.gravity_constant * avg_density
        
        return curvature
    
    # ========== Information Geometry ==========
    
    def fidelity(self, rho: Qobj, sigma: Qobj) -> float:
        """
        Calculate quantum fidelity F(ρ, σ).
        
        For pure states: F = |⟨ψ|φ⟩|²
        For mixed states: F = (Tr√(√ρ σ √ρ))²
        
        Args:
            rho: First quantum state
            sigma: Second quantum state
            
        Returns:
            Fidelity (1 for identical states, 0 for orthogonal)
        """
        # Handle pure states efficiently
        if rho.type == 'ket' and sigma.type == 'ket':
            overlap = rho.dag() * sigma
            return np.abs(overlap) ** 2
        
        # Convert to density matrices
        if rho.type == 'ket':
            rho = rho * rho.dag()
        if sigma.type == 'ket':
            sigma = sigma * sigma.dag()
        
        # General fidelity calculation
        rho_mat = rho.full()
        sigma_mat = sigma.full()
        
        # F = (Tr√(√ρ σ √ρ))²
        sqrt_rho = np.sqrt(rho_mat + 1e-10 * np.eye(rho_mat.shape[0]))
        prod = sqrt_rho @ sigma_mat @ sqrt_rho
        sqrt_prod = np.sqrt(prod + 1e-10 * np.eye(prod.shape[0]))
        
        fidelity = np.real(np.trace(sqrt_prod)) ** 2
        return np.clip(fidelity, 0, 1)
    
    def bures_distance(self, rho: Qobj, sigma: Qobj) -> float:
        """
        Calculate Bures distance (information geometry metric).
        
        D_B(ρ, σ) = √(2(1 - √F(ρ, σ)))
        
        Args:
            rho: First quantum state
            sigma: Second quantum state
            
        Returns:
            Bures distance (0 for identical states)
        """
        fid = self.fidelity(rho, sigma)
        return np.sqrt(2 * (1 - np.sqrt(fid)))
    
    def trace_distance(self, rho: Qobj, sigma: Qobj) -> float:
        """
        Calculate trace distance (operational distinguishability).
        
        D_tr(ρ, σ) = (1/2) ||ρ - σ||₁
        
        Args:
            rho: First quantum state
            sigma: Second quantum state
            
        Returns:
            Trace distance (0 for identical, 1 for orthogonal)
        """
        if rho.type == 'ket':
            rho = rho * rho.dag()
        if sigma.type == 'ket':
            sigma = sigma * sigma.dag()
        
        diff = (rho - sigma).full()
        eigenvals = np.linalg.eigvalsh(diff)
        return 0.5 * np.sum(np.abs(eigenvals))
    
    # ========== Composite Metrics ==========
    
    def information_profile(self, state: Qobj) -> Dict[str, float]:
        """
        Generate comprehensive information profile of a state.
        
        Args:
            state: Quantum state
            
        Returns:
            Dictionary of information metrics
        """
        return {
            'entropy': self.von_neumann_entropy(state),
            'purity': self.purity(state),
            'mass': self.information_mass(state),
            'density': self.information_density(state),
            'potential': self.information_potential(state)
        }
    
    def entanglement_network(
        self,
        state: Qobj,
        n_qubits: int
    ) -> np.ndarray:
        """
        Calculate entanglement network (mutual information matrix).
        
        Args:
            state: Multi-qubit quantum state
            n_qubits: Number of qubits
            
        Returns:
            n_qubits × n_qubits matrix of mutual information
        """
        network = np.zeros((n_qubits, n_qubits))
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                try:
                    mi = self.mutual_information(state, [i], [j])
                    network[i, j] = mi
                    network[j, i] = mi
                except:
                    # If calculation fails, set to zero
                    network[i, j] = 0
                    network[j, i] = 0
        
        return network
    
    def gravitational_potential_energy(
        self,
        states: List[Qobj],
        positions: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate total gravitational potential energy of system.
        
        U = -Σᵢⱼ G * mᵢ * mⱼ / rᵢⱼ
        
        Args:
            states: List of quantum states
            positions: Spatial positions (default: equally spaced)
            
        Returns:
            Total potential energy (negative for bound systems)
        """
        n_states = len(states)
        
        # Default positions
        if positions is None:
            positions = np.array([[i, 0, 0] for i in range(n_states)])
        
        # Calculate masses
        masses = np.array([self.information_mass(s) for s in states])
        
        # Calculate pairwise potential
        total_energy = 0.0
        for i in range(n_states):
            for j in range(i + 1, n_states):
                r = np.linalg.norm(positions[i] - positions[j])
                if r > 0.01:  # Avoid singularity
                    u_ij = -self.gravity_constant * masses[i] * masses[j] / r
                    total_energy += u_ij
        
        return total_energy


def demo_info_mass_gravity():
    """
    Demonstrate info-mass-gravity module functionality.
    """
    print("=== Info-Mass-Gravity Module Demo ===\n")
    
    img = InfoMassGravity()
    
    # Demo 1: Information measures
    print("Demo 1: Information Theory Measures")
    print("-" * 40)
    
    # Create test states
    from qutip import basis, bell_state
    
    # Pure state
    pure = basis(2, 0)
    print(f"Pure state |0⟩:")
    print(f"  Entropy: {img.von_neumann_entropy(pure):.6f}")
    print(f"  Purity: {img.purity(pure):.6f}")
    
    # Mixed state
    mixed = 0.7 * basis(2, 0) * basis(2, 0).dag() + 0.3 * basis(2, 1) * basis(2, 1).dag()
    print(f"\nMixed state (0.7|0⟩⟨0| + 0.3|1⟩⟨1|):")
    print(f"  Entropy: {img.von_neumann_entropy(mixed):.6f}")
    print(f"  Purity: {img.purity(mixed):.6f}")
    
    # Bell state
    bell = bell_state('00')
    print(f"\nBell state |Φ⁺⟩:")
    print(f"  Entropy: {img.von_neumann_entropy(bell):.6f}")
    print(f"  Mutual Information I(0:1): {img.mutual_information(bell, [0], [1]):.6f}")
    
    # Demo 2: Mass analogies
    print("\n\nDemo 2: Information Mass")
    print("-" * 40)
    
    states = [pure, mixed, bell]
    names = ["Pure |0⟩", "Mixed", "Bell |Φ⁺⟩"]
    
    for name, state in zip(names, states):
        profile = img.information_profile(state)
        print(f"\n{name}:")
        for key, value in profile.items():
            print(f"  {key}: {value:.6f}")
    
    # Demo 3: Gravity analogies
    print("\n\nDemo 3: Entanglement Force")
    print("-" * 40)
    
    # Product state (no entanglement)
    from qutip import tensor
    product = tensor(basis(2, 0), basis(2, 0))
    force_product = img.entanglement_force(product, [0], [1])
    print(f"Product state |00⟩:")
    print(f"  Entanglement force: {force_product:.6f}")
    
    # Bell state (maximal entanglement)
    force_bell = img.entanglement_force(bell, [0], [1])
    print(f"\nBell state |Φ⁺⟩:")
    print(f"  Entanglement force: {force_bell:.6f}")
    print(f"  Force ratio (Bell/Product): {force_bell/max(force_product, 0.001):.2f}x")
    
    # Demo 4: Information geometry
    print("\n\nDemo 4: Information Geometry")
    print("-" * 40)
    
    state1 = basis(2, 0)
    state2 = basis(2, 1)
    state3 = (basis(2, 0) + basis(2, 1)).unit()
    
    print("Distance metrics:")
    print(f"  |0⟩ to |1⟩:")
    print(f"    Fidelity: {img.fidelity(state1, state2):.6f}")
    print(f"    Bures distance: {img.bures_distance(state1, state2):.6f}")
    print(f"    Trace distance: {img.trace_distance(state1, state2):.6f}")
    
    print(f"\n  |0⟩ to |+⟩:")
    print(f"    Fidelity: {img.fidelity(state1, state3):.6f}")
    print(f"    Bures distance: {img.bures_distance(state1, state3):.6f}")
    print(f"    Trace distance: {img.trace_distance(state1, state3):.6f}")
    
    # Demo 5: Entanglement network
    print("\n\nDemo 5: Entanglement Network (3-qubit GHZ)")
    print("-" * 40)
    
    # GHZ state
    ghz = (tensor(basis(2, 0), basis(2, 0), basis(2, 0)) + 
           tensor(basis(2, 1), basis(2, 1), basis(2, 1))).unit()
    
    network = img.entanglement_network(ghz, 3)
    print("Mutual information matrix:")
    print(network)
    
    # Demo 6: Center of mass
    print("\n\nDemo 6: Center of Information Mass")
    print("-" * 40)
    
    test_states = [
        basis(2, 0),  # Pure
        mixed,         # Mixed
        bell           # Entangled
    ]
    
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    center = img.center_of_information_mass(test_states, positions)
    print(f"States at positions: {positions[:, 0]}")
    print(f"Center of mass: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    
    # Demo 7: Curvature
    print("\n\nDemo 7: Information Curvature")
    print("-" * 40)
    
    curvature = img.information_curvature(test_states)
    print(f"Information curvature: {curvature:.6f}")
    print(f"(Higher curvature indicates denser information)")
    
    print("\n=== Demo Complete ===")
    
    return {
        'img': img,
        'states': {
            'pure': pure,
            'mixed': mixed,
            'bell': bell,
            'ghz': ghz
        }
    }


if __name__ == '__main__':
    demo_info_mass_gravity()
