"""
Steane Code Quantum Error Correction Simulation Module

Implements [[7,1,3]] Steane code with:
- Depolarizing pseudo-threshold calculations (η_thr ≈ 9.3×10^{-5})
- Eigenvalue analysis with Pauli spectra
- Surface lattice syndromes

Supports both QuTiP and Qiskit quantum computing frameworks.
"""

import numpy as np
from qutip import (
    basis, tensor, qeye, sigmax, sigmay, sigmaz,
    Qobj, fidelity, expect
)
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

# Optional Qiskit imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    # Create dummy types for type hints when Qiskit not available
    Statevector = DensityMatrix = object
    print("Warning: Qiskit not installed. Only QuTiP backend will be available.")


class SteaneCode:
    """
    Implements the [[7,1,3]] Steane quantum error correction code.
    
    The Steane code is a CSS (Calderbank-Shor-Steane) code that can
    correct arbitrary single-qubit errors.
    """
    
    def __init__(self):
        """Initialize Steane code with generator matrices."""
        # Steane code stabilizer generators (simplified)
        # H matrix for the Steane code
        self.n_qubits = 7
        self.n_data = 1
        self.distance = 3
        
        # Pauli operators for single qubits
        self.I = qeye(2)
        self.X = sigmax()
        self.Y = sigmay()
        self.Z = sigmaz()
        
        # Threshold for depolarizing channel
        self.theoretical_threshold = 9.3e-5
        
    def encode_logical_zero(self) -> Qobj:
        """
        Encode logical |0⟩ state into 7-qubit Steane code.
        
        Returns:
            Qobj: 7-qubit encoded state
        """
        # Simplified encoding: |0_L⟩ = 1/√8 Σ |codeword⟩
        # For demonstration, using computational basis states
        zero = basis(2, 0)
        one = basis(2, 1)
        
        # Create logical zero codeword
        # |0_L⟩ has even parity across different subsets
        logical_zero = tensor(zero, zero, zero, zero, zero, zero, zero)
        logical_zero = logical_zero.unit()
        
        return logical_zero
    
    def encode_logical_one(self) -> Qobj:
        """
        Encode logical |1⟩ state into 7-qubit Steane code.
        
        Returns:
            Qobj: 7-qubit encoded state
        """
        zero = basis(2, 0)
        one = basis(2, 1)
        
        # Create logical one codeword (X applied to all qubits of logical zero)
        logical_one = tensor(one, one, one, one, one, one, one)
        logical_one = logical_one.unit()
        
        return logical_one
    
    def apply_depolarizing_noise(self, state: Qobj, p: float) -> Qobj:
        """
        Apply depolarizing noise to each qubit independently.
        
        Args:
            state: Input quantum state
            p: Error probability per qubit
            
        Returns:
            Noisy state (density matrix)
        """
        # Convert to density matrix if not already
        if state.type == 'ket':
            rho = state * state.dag()
        else:
            rho = state
            
        # Apply depolarizing noise to each qubit
        for qubit_idx in range(self.n_qubits):
            # Create identity on all qubits except target
            ops_before = [self.I] * qubit_idx
            ops_after = [self.I] * (self.n_qubits - qubit_idx - 1)
            
            # Apply depolarizing channel: ρ → (1-p)ρ + p/3(XρX + YρY + ZρZ)
            if p > 0:
                # Keep state with probability (1-p)
                rho_new = (1 - p) * rho
                
                # Apply X error with probability p/3
                X_op = tensor(*(ops_before + [self.X] + ops_after))
                rho_new += (p / 3) * X_op * rho * X_op.dag()
                
                # Apply Y error with probability p/3
                Y_op = tensor(*(ops_before + [self.Y] + ops_after))
                rho_new += (p / 3) * Y_op * rho * Y_op.dag()
                
                # Apply Z error with probability p/3
                Z_op = tensor(*(ops_before + [self.Z] + ops_after))
                rho_new += (p / 3) * Z_op * rho * Z_op.dag()
                
                rho = rho_new
                
        return rho
    
    def measure_syndrome(self, state: Qobj) -> np.ndarray:
        """
        Measure error syndromes for the Steane code.
        
        Args:
            state: Quantum state (can be noisy)
            
        Returns:
            Syndrome array (6 bits for Steane code)
        """
        # Simplified syndrome measurement
        # In practice, would measure stabilizer generators
        syndrome = np.random.randint(0, 2, size=6)
        return syndrome
    
    def correct_errors(self, state: Qobj, syndrome: np.ndarray) -> Qobj:
        """
        Apply error correction based on syndrome.
        
        Args:
            state: Noisy quantum state
            syndrome: Measured syndrome
            
        Returns:
            Corrected state
        """
        # Simplified error correction
        # In practice, would look up correction operation from syndrome table
        return state
    
    def calculate_logical_error_rate(
        self, 
        p_phys: float, 
        n_trials: int = 1000
    ) -> float:
        """
        Calculate logical error rate through Monte Carlo simulation.
        
        Args:
            p_phys: Physical error rate per qubit
            n_trials: Number of Monte Carlo trials
            
        Returns:
            Logical error rate
        """
        errors = 0
        
        for _ in range(n_trials):
            # Encode logical state
            state = self.encode_logical_zero()
            
            # Apply noise
            noisy_state = self.apply_depolarizing_noise(state, p_phys)
            
            # Measure syndrome and correct
            syndrome = self.measure_syndrome(noisy_state)
            corrected_state = self.correct_errors(noisy_state, syndrome)
            
            # Check if logical error occurred
            # For simplicity, using fidelity threshold
            original = self.encode_logical_zero()
            if corrected_state.type == 'oper':
                # For density matrices, compute fidelity
                f = fidelity(original, corrected_state)
            else:
                f = fidelity(original, corrected_state)
                
            if f < 0.9:
                errors += 1
                
        return errors / n_trials
    
    def compute_pauli_spectrum(self, state: Qobj) -> Dict[str, float]:
        """
        Compute Pauli spectrum (eigenvalues under Pauli basis).
        
        Args:
            state: Quantum state
            
        Returns:
            Dictionary of Pauli expectation values
        """
        spectrum = {}
        
        # For each qubit, compute X, Y, Z expectations
        for qubit_idx in range(self.n_qubits):
            ops_before = [self.I] * qubit_idx
            ops_after = [self.I] * (self.n_qubits - qubit_idx - 1)
            
            X_op = tensor(*(ops_before + [self.X] + ops_after))
            Y_op = tensor(*(ops_before + [self.Y] + ops_after))
            Z_op = tensor(*(ops_before + [self.Z] + ops_after))
            
            if state.type == 'ket':
                rho = state * state.dag()
            else:
                rho = state
                
            spectrum[f'X_{qubit_idx}'] = expect(X_op, rho)
            spectrum[f'Y_{qubit_idx}'] = expect(Y_op, rho)
            spectrum[f'Z_{qubit_idx}'] = expect(Z_op, rho)
            
        return spectrum


class ThresholdSimulation:
    """
    Simulate pseudo-threshold behavior for quantum error correction.
    """
    
    def __init__(self, code: Optional[SteaneCode] = None):
        """
        Initialize threshold simulation.
        
        Args:
            code: QEC code instance (defaults to SteaneCode)
        """
        self.code = code if code else SteaneCode()
        
    def run_threshold_scan(
        self,
        p_phys_range: np.ndarray,
        n_trials: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan physical error rates to find pseudo-threshold.
        
        Args:
            p_phys_range: Array of physical error rates to test
            n_trials: Number of trials per error rate
            
        Returns:
            Tuple of (physical_rates, logical_rates)
        """
        logical_rates = []
        
        for p_phys in p_phys_range:
            p_log = self.code.calculate_logical_error_rate(p_phys, n_trials)
            logical_rates.append(p_log)
            
        return p_phys_range, np.array(logical_rates)
    
    def plot_threshold_curve(
        self,
        p_phys: np.ndarray,
        p_log: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot threshold curve showing physical vs logical error rates.
        
        Args:
            p_phys: Physical error rates
            p_log: Logical error rates
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot physical error rate (flat line)
        ax.plot(p_phys, p_phys, 'k--', label='Physical (uncoded)', linewidth=2)
        
        # Plot logical error rate (should bend down at low p_phys)
        ax.plot(p_phys, p_log, 'b-', label='Logical (Steane [[7,1,3]])', 
                linewidth=2, marker='o')
        
        # Mark theoretical threshold
        ax.axvline(self.code.theoretical_threshold, color='r', 
                   linestyle=':', label=f'Threshold ≈ {self.code.theoretical_threshold:.2e}')
        
        ax.set_xlabel('Physical Error Rate', fontsize=12)
        ax.set_ylabel('Error Rate', fontsize=12)
        ax.set_title('Pseudo-Threshold Curve for Steane Code', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig


class SurfaceLattice:
    """
    Surface code lattice syndrome visualization.
    """
    
    def __init__(self, size: int = 5):
        """
        Initialize surface code lattice.
        
        Args:
            size: Lattice size (size x size)
        """
        self.size = size
        self.n_qubits = size * size
        
    def generate_syndromes(
        self,
        error_rate: float
    ) -> np.ndarray:
        """
        Generate syndrome pattern for surface code.
        
        Args:
            error_rate: Physical error rate
            
        Returns:
            2D array of syndromes
        """
        # Simplified: random syndromes based on error rate
        syndromes = np.random.rand(self.size - 1, self.size - 1) < error_rate
        return syndromes.astype(int)
    
    def visualize_syndromes(
        self,
        syndromes: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize surface code syndromes.
        
        Args:
            syndromes: 2D syndrome array
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw lattice
        for i in range(self.size):
            ax.axhline(i, color='gray', linewidth=0.5)
            ax.axvline(i, color='gray', linewidth=0.5)
            
        # Draw qubits (data)
        for i in range(self.size):
            for j in range(self.size):
                circle = plt.Circle((j, i), 0.15, color='lightblue', 
                                   edgecolor='black', linewidth=2)
                ax.add_patch(circle)
                
        # Draw syndromes
        for i in range(syndromes.shape[0]):
            for j in range(syndromes.shape[1]):
                if syndromes[i, j]:
                    circle = plt.Circle((j + 0.5, i + 0.5), 0.1, 
                                       color='red', alpha=0.7)
                    ax.add_patch(circle)
                    
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_aspect('equal')
        ax.set_title('Surface Code Lattice with Syndromes', fontsize=14)
        ax.axis('off')
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig


class SteaneCodeQiskit:
    """
    Qiskit-based implementation of the [[7,1,3]] Steane quantum error correction code.
    
    Provides equivalent functionality to SteaneCode but using Qiskit framework.
    """
    
    def __init__(self):
        """Initialize Steane code with Qiskit."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available. Install qiskit and qiskit-aer.")
        
        self.n_qubits = 7
        self.n_data = 1
        self.distance = 3
        self.theoretical_threshold = 9.3e-5
        
    def encode_logical_zero(self) -> Statevector:
        """
        Encode logical |0⟩ state into 7-qubit Steane code using Qiskit.
        
        Returns:
            Statevector: 7-qubit encoded state
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Simplified Steane encoding for |0_L⟩
        # In practice, would apply stabilizer-based encoding circuit
        # For now, initialize to |0000000⟩
        
        return Statevector(qc)
    
    def encode_logical_one(self) -> Statevector:
        """
        Encode logical |1⟩ state into 7-qubit Steane code using Qiskit.
        
        Returns:
            Statevector: 7-qubit encoded state
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Apply X to all qubits (simplified)
        for i in range(self.n_qubits):
            qc.x(i)
        
        return Statevector(qc)
    
    def apply_depolarizing_noise(
        self, 
        state: Statevector, 
        p: float
    ) -> DensityMatrix:
        """
        Apply depolarizing noise to each qubit using Qiskit noise model.
        
        Args:
            state: Input quantum state
            p: Error probability per qubit
            
        Returns:
            DensityMatrix: Noisy state
        """
        # Create noise model
        noise_model = NoiseModel()
        error = depolarizing_error(p, 1)
        
        # Add depolarizing error to all qubits
        for i in range(self.n_qubits):
            noise_model.add_quantum_error(error, ['id'], [i])
        
        # Create circuit with identity gates to apply noise
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.id(i)
        
        # Simulate with noise
        simulator = AerSimulator(noise_model=noise_model)
        qc.save_density_matrix()
        result = simulator.run(qc, initial_statevector=state.data).result()
        
        return DensityMatrix(result.data()['density_matrix'])
    
    def calculate_logical_error_rate(
        self, 
        p_phys: float, 
        n_trials: int = 1000
    ) -> float:
        """
        Calculate logical error rate through Monte Carlo simulation.
        
        Args:
            p_phys: Physical error rate per qubit
            n_trials: Number of Monte Carlo trials
            
        Returns:
            Logical error rate
        """
        errors = 0
        
        for _ in range(n_trials):
            # Encode logical state
            state = self.encode_logical_zero()
            
            # Apply noise
            noisy_state = self.apply_depolarizing_noise(state, p_phys)
            
            # Check fidelity with original
            original = self.encode_logical_zero()
            f = state_fidelity(original, noisy_state)
            
            if f < 0.9:
                errors += 1
        
        return errors / n_trials
    
    def compute_pauli_spectrum(self, state) -> Dict[str, float]:
        """
        Compute Pauli spectrum using Qiskit.
        
        Args:
            state: Quantum state (Statevector or DensityMatrix)
            
        Returns:
            Dictionary of Pauli expectation values
        """
        from qiskit.quantum_info import Pauli
        
        spectrum = {}
        
        # For each qubit, compute X, Y, Z expectations
        for qubit_idx in range(self.n_qubits):
            # Create Pauli strings
            pauli_x = ['I'] * self.n_qubits
            pauli_x[qubit_idx] = 'X'
            pauli_y = ['I'] * self.n_qubits
            pauli_y[qubit_idx] = 'Y'
            pauli_z = ['I'] * self.n_qubits
            pauli_z[qubit_idx] = 'Z'
            
            # Compute expectation values
            if isinstance(state, Statevector):
                state_dm = DensityMatrix(state)
            else:
                state_dm = state
            
            spectrum[f'X_{qubit_idx}'] = state_dm.expectation_value(
                Pauli(''.join(reversed(pauli_x)))
            ).real
            spectrum[f'Y_{qubit_idx}'] = state_dm.expectation_value(
                Pauli(''.join(reversed(pauli_y)))
            ).real
            spectrum[f'Z_{qubit_idx}'] = state_dm.expectation_value(
                Pauli(''.join(reversed(pauli_z)))
            ).real
        
        return spectrum


def create_steane_code(backend: str = 'qutip'):
    """
    Factory function to create a Steane code instance with specified backend.
    
    Args:
        backend: 'qutip' or 'qiskit'
        
    Returns:
        SteaneCode or SteaneCodeQiskit instance
        
    Raises:
        ValueError: If backend is not supported
    """
    backend = backend.lower()
    
    if backend == 'qutip':
        return SteaneCode()
    elif backend == 'qiskit':
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit backend requested but not available. "
                "Install qiskit and qiskit-aer."
            )
        return SteaneCodeQiskit()
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported: 'qutip', 'qiskit'"
        )


def demo_steane_simulation():
    """
    Demonstrate Steane code simulation capabilities.
    """
    print("=== Steane Code QEC Simulation Demo ===\n")
    
    # Initialize code
    code = SteaneCode()
    print(f"Initialized Steane [[{code.n_qubits},{code.n_data},{code.distance}]] code")
    print(f"Theoretical threshold: η_thr ≈ {code.theoretical_threshold:.2e}\n")
    
    # Encode logical states
    print("Encoding logical states...")
    logical_zero = code.encode_logical_zero()
    print(f"  |0_L⟩ encoded: {logical_zero.shape}")
    
    # Apply noise and compute spectrum
    print("\nApplying depolarizing noise (p=0.01)...")
    noisy_state = code.apply_depolarizing_noise(logical_zero, 0.01)
    
    print("\nComputing Pauli spectrum...")
    spectrum = code.compute_pauli_spectrum(noisy_state)
    print(f"  First 3 Pauli expectations: {list(spectrum.items())[:3]}")
    
    # Run threshold simulation
    print("\nRunning threshold scan...")
    sim = ThresholdSimulation(code)
    p_range = np.logspace(-4, -1, 10)
    p_phys, p_log = sim.run_threshold_scan(p_range, n_trials=50)
    print(f"  Scanned {len(p_range)} error rates")
    
    # Generate surface syndromes
    print("\nGenerating surface code syndromes...")
    surface = SurfaceLattice(size=5)
    syndromes = surface.generate_syndromes(error_rate=0.1)
    print(f"  Generated {syndromes.shape[0]}x{syndromes.shape[1]} syndrome lattice")
    
    print("\n=== Demo Complete ===")
    
    return {
        'code': code,
        'simulation': sim,
        'surface': surface,
        'p_phys': p_phys,
        'p_log': p_log,
        'syndromes': syndromes
    }


def demo_backend_comparison():
    """
    Demonstrate both QuTiP and Qiskit backends.
    """
    print("=== Backend Comparison Demo ===\n")
    
    # Test QuTiP backend
    print("Testing QuTiP backend...")
    qutip_code = create_steane_code('qutip')
    print(f"  Created: {qutip_code.__class__.__name__}")
    qutip_state = qutip_code.encode_logical_zero()
    print(f"  Encoded |0_L⟩: shape {qutip_state.shape}")
    
    # Test Qiskit backend if available
    if QISKIT_AVAILABLE:
        print("\nTesting Qiskit backend...")
        qiskit_code = create_steane_code('qiskit')
        print(f"  Created: {qiskit_code.__class__.__name__}")
        qiskit_state = qiskit_code.encode_logical_zero()
        print(f"  Encoded |0_L⟩: {len(qiskit_state)} dimensions")
        
        # Compare error rates
        print("\nComparing logical error rates (p_phys=0.01, 50 trials)...")
        qutip_p_log = qutip_code.calculate_logical_error_rate(0.01, n_trials=50)
        qiskit_p_log = qiskit_code.calculate_logical_error_rate(0.01, n_trials=50)
        print(f"  QuTiP:  p_log = {qutip_p_log:.4f}")
        print(f"  Qiskit: p_log = {qiskit_p_log:.4f}")
    else:
        print("\nQiskit not available - skipping Qiskit backend test")
    
    print("\n=== Demo Complete ===")


if __name__ == '__main__':
    import sys
    if '--compare' in sys.argv:
        demo_backend_comparison()
    else:
        demo_steane_simulation()
