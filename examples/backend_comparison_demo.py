#!/usr/bin/env python3
"""
Backend Comparison Demo

Demonstrates the dual backend support (QuTiP vs Qiskit) for Steane code simulations.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qec_steane import create_steane_code, QISKIT_AVAILABLE


def main():
    """Run backend comparison demonstration."""
    print("=" * 60)
    print("Steane Code Backend Comparison Demo")
    print("=" * 60)
    print()
    
    # QuTiP Backend
    print("1. QuTiP Backend (default)")
    print("-" * 60)
    qutip_code = create_steane_code('qutip')
    print(f"   Class: {qutip_code.__class__.__name__}")
    print(f"   Qubits: {qutip_code.n_qubits}")
    print(f"   Distance: {qutip_code.distance}")
    print(f"   Threshold: {qutip_code.theoretical_threshold:.2e}")
    
    # Encode and test
    state = qutip_code.encode_logical_zero()
    print(f"   Encoded |0⟩: {state.shape}")
    
    # Apply noise
    noisy = qutip_code.apply_depolarizing_noise(state, 0.01)
    print(f"   Noisy state: {noisy.shape}")
    
    # Compute spectrum (first 3 values)
    spectrum = qutip_code.compute_pauli_spectrum(noisy)
    items = list(spectrum.items())[:3]
    print(f"   Pauli spectrum (first 3): {items}")
    
    # Error rate
    p_log = qutip_code.calculate_logical_error_rate(0.01, n_trials=20)
    print(f"   Logical error rate (p=0.01, 20 trials): {p_log:.4f}")
    print()
    
    # Qiskit Backend
    if QISKIT_AVAILABLE:
        print("2. Qiskit Backend")
        print("-" * 60)
        qiskit_code = create_steane_code('qiskit')
        print(f"   Class: {qiskit_code.__class__.__name__}")
        print(f"   Qubits: {qiskit_code.n_qubits}")
        print(f"   Distance: {qiskit_code.distance}")
        print(f"   Threshold: {qiskit_code.theoretical_threshold:.2e}")
        
        # Encode and test
        state = qiskit_code.encode_logical_zero()
        print(f"   Encoded |0⟩: {len(state)} dimensions")
        
        # Apply noise
        noisy = qiskit_code.apply_depolarizing_noise(state, 0.01)
        print(f"   Noisy state: DensityMatrix")
        
        # Compute spectrum (first 3 values)
        spectrum = qiskit_code.compute_pauli_spectrum(noisy)
        items = list(spectrum.items())[:3]
        print(f"   Pauli spectrum (first 3): {items}")
        
        # Error rate
        p_log = qiskit_code.calculate_logical_error_rate(0.01, n_trials=20)
        print(f"   Logical error rate (p=0.01, 20 trials): {p_log:.4f}")
        print()
        
        # Comparison
        print("3. Comparison")
        print("-" * 60)
        print("   Both backends implement the same Steane [[7,1,3]] code")
        print("   with equivalent functionality:")
        print("   - Encoding logical states")
        print("   - Applying depolarizing noise")
        print("   - Computing Pauli spectra")
        print("   - Calculating logical error rates")
        print()
        print("   Choose backend based on:")
        print("   - QuTiP: Simpler, faster for small systems")
        print("   - Qiskit: Industry standard, extensive ecosystem")
    else:
        print("2. Qiskit Backend")
        print("-" * 60)
        print("   Qiskit not installed - install with:")
        print("   pip install qiskit qiskit-aer")
        print()
    
    print("=" * 60)
    print("Demo complete!")
    print()


if __name__ == '__main__':
    main()
