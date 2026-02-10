#!/usr/bin/env python
"""
Example: Using the Info-Mass-Gravity Module

Demonstrates physics-inspired metrics for quantum information:
- Information theory measures
- Mass and gravity analogies
- Information geometry
- Integration with QEC simulations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from qutip import basis, bell_state, tensor
from info_mass_gravity import InfoMassGravity
from qec_steane import SteaneCode


def example_1_information_measures():
    """Example 1: Basic information theory measures."""
    print("=" * 60)
    print("Example 1: Information Theory Measures")
    print("=" * 60)
    
    img = InfoMassGravity()
    
    # Create different quantum states
    pure = basis(2, 0)
    mixed = 0.7 * basis(2, 0) * basis(2, 0).dag() + 0.3 * basis(2, 1) * basis(2, 1).dag()
    bell = bell_state('00')
    
    print("\n1. Pure state |0⟩:")
    print(f"   Entropy: {img.von_neumann_entropy(pure):.6f} bits")
    print(f"   Purity:  {img.purity(pure):.6f}")
    
    print("\n2. Mixed state (70% |0⟩, 30% |1⟩):")
    print(f"   Entropy: {img.von_neumann_entropy(mixed):.6f} bits")
    print(f"   Purity:  {img.purity(mixed):.6f}")
    
    print("\n3. Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:")
    print(f"   Entropy: {img.von_neumann_entropy(bell):.6f} bits")
    print(f"   Mutual Information I(A:B): {img.mutual_information(bell, [0], [1]):.6f} bits")
    print("   → Maximal correlation between qubits!")


def example_2_mass_analogies():
    """Example 2: Information mass analogies."""
    print("\n" + "=" * 60)
    print("Example 2: Information Mass Analogies")
    print("=" * 60)
    
    img = InfoMassGravity()
    steane = SteaneCode()
    
    # Create states with different noise levels
    error_rates = [0.0, 0.01, 0.05, 0.1]
    
    print("\nInformation mass vs. error rate:")
    print("-" * 60)
    print(f"{'Error Rate':<12} {'Entropy':<12} {'Purity':<12} {'Info Mass':<12}")
    print("-" * 60)
    
    for rate in error_rates:
        state = steane.encode_logical_zero()
        if rate > 0:
            state = steane.apply_depolarizing_noise(state, rate)
        
        profile = img.information_profile(state)
        print(f"{rate:<12.3f} {profile['entropy']:<12.4f} "
              f"{profile['purity']:<12.4f} {profile['mass']:<12.4f}")
    
    print("\n→ Higher error rates increase information mass (more mixedness)")


def example_3_entanglement_force():
    """Example 3: Entanglement as gravitational force."""
    print("\n" + "=" * 60)
    print("Example 3: Entanglement as Gravitational Force")
    print("=" * 60)
    
    img = InfoMassGravity()
    
    # Product state (no entanglement)
    product = tensor(basis(2, 0), basis(2, 0))
    force_product = img.entanglement_force(product, [0], [1])
    
    # Bell state (maximal entanglement)
    bell = bell_state('00')
    force_bell = img.entanglement_force(bell, [0], [1])
    
    # Partially entangled state
    partial = (np.sqrt(0.8) * tensor(basis(2, 0), basis(2, 0)) + 
               np.sqrt(0.2) * tensor(basis(2, 1), basis(2, 1))).unit()
    force_partial = img.entanglement_force(partial, [0], [1])
    
    print("\nEntanglement forces (F ∝ G·m_A·m_B·I(A:B)/r²):")
    print("-" * 60)
    print(f"Product state |00⟩:         F = {force_product:.6f}")
    print(f"Partial entanglement (80:20): F = {force_partial:.6f}")
    print(f"Bell state |Φ⁺⟩:           F = {force_bell:.6f}")
    print(f"\n→ Force ratio (Bell/Product): {force_bell/max(force_product, 0.001):.1f}x")
    print("→ Stronger entanglement = stronger 'gravitational attraction'")


def example_4_information_geometry():
    """Example 4: Information geometry and distances."""
    print("\n" + "=" * 60)
    print("Example 4: Information Geometry")
    print("=" * 60)
    
    img = InfoMassGravity()
    
    # Create states
    state_0 = basis(2, 0)
    state_1 = basis(2, 1)
    state_plus = (basis(2, 0) + basis(2, 1)).unit()
    state_minus = (basis(2, 0) - basis(2, 1)).unit()
    
    states = [
        ("| 0⟩", state_0),
        ("| 1⟩", state_1),
        ("| +⟩", state_plus),
        ("| -⟩", state_minus)
    ]
    
    print("\nFidelity matrix (1 = identical, 0 = orthogonal):")
    print("-" * 60)
    print(f"{'State':<8}", end="")
    for name, _ in states:
        print(f"{name:<12}", end="")
    print()
    print("-" * 60)
    
    for name1, state1 in states:
        print(f"{name1:<8}", end="")
        for name2, state2 in states:
            fid = img.fidelity(state1, state2)
            print(f"{fid:<12.4f}", end="")
        print()
    
    print("\nBures distance (quantum metric):")
    print("-" * 60)
    print(f"| 0⟩ to | 1⟩: D_B = {img.bures_distance(state_0, state_1):.4f}")
    print(f"| 0⟩ to | +⟩: D_B = {img.bures_distance(state_0, state_plus):.4f}")
    print(f"| +⟩ to | -⟩: D_B = {img.bures_distance(state_plus, state_minus):.4f}")


def example_5_qec_integration():
    """Example 5: Integration with QEC simulations."""
    print("\n" + "=" * 60)
    print("Example 5: QEC Integration - Tracking Error Evolution")
    print("=" * 60)
    
    img = InfoMassGravity()
    steane = SteaneCode()
    
    # Simulate error accumulation
    error_rates = np.logspace(-3, -1, 5)
    
    print("\nError evolution in Steane [[7,1,3]] code:")
    print("-" * 60)
    print(f"{'p_phys':<10} {'Entropy':<12} {'Fidelity':<12} {'Info Mass':<12}")
    print("-" * 60)
    
    # Reference: perfect logical zero
    ref_state = steane.encode_logical_zero()
    
    for rate in error_rates:
        # Add noise
        noisy = steane.apply_depolarizing_noise(steane.encode_logical_zero(), rate)
        
        # Calculate metrics
        entropy = img.von_neumann_entropy(noisy)
        fidelity = img.fidelity(ref_state, noisy)
        mass = img.information_mass(noisy)
        
        print(f"{rate:<10.4f} {entropy:<12.6f} {fidelity:<12.6f} {mass:<12.6f}")
    
    print("\n→ Information mass tracks error accumulation")
    print("→ Fidelity quantifies distance from ideal state")


def example_6_entanglement_network():
    """Example 6: Visualize entanglement network."""
    print("\n" + "=" * 60)
    print("Example 6: Entanglement Network Visualization")
    print("=" * 60)
    
    img = InfoMassGravity()
    
    # Create GHZ state: (|000⟩ + |111⟩)/√2
    ghz = (tensor(basis(2, 0), basis(2, 0), basis(2, 0)) + 
           tensor(basis(2, 1), basis(2, 1), basis(2, 1))).unit()
    
    network = img.entanglement_network(ghz, 3)
    
    print("\nMutual information network for GHZ state:")
    print("(|000⟩ + |111⟩)/√2")
    print("-" * 60)
    print("     Qubit 0  Qubit 1  Qubit 2")
    for i in range(3):
        print(f"Q{i}  ", end="")
        for j in range(3):
            print(f"{network[i,j]:<9.4f}", end="")
        print()
    
    print("\n→ All pairs maximally correlated (I = 2 bits)")
    print("→ Characteristic of GHZ entanglement")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  INFO-MASS-GRAVITY MODULE: COMPREHENSIVE EXAMPLES  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        example_1_information_measures()
        example_2_mass_analogies()
        example_3_entanglement_force()
        example_4_information_geometry()
        example_5_qec_integration()
        example_6_entanglement_network()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("• Information mass quantifies quantum mixedness")
        print("• Entanglement force provides intuitive 'attraction' measure")
        print("• Information geometry tracks state evolution")
        print("• Integration with QEC enables error tracking")
        print("• Physics-inspired metrics aid quantum understanding")
        print()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
