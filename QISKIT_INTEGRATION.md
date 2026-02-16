# Qiskit Backend Integration

## Overview

The fusion-qec-sim project now supports **dual quantum computing backends**: QuTiP and Qiskit. This provides users with flexibility to choose the framework that best suits their needs.

## Features

### 1. Backend Selection

Choose your preferred quantum computing framework:

- **QuTiP** (default): Quantum Toolbox in Python - lightweight, fast for small systems
- **Qiskit**: IBM's industry-standard quantum computing framework - extensive ecosystem, real hardware integration

### 2. API Consistency

Both backends implement the same interface:

```python
from src.qec_steane import create_steane_code

# Create with QuTiP (default)
code = create_steane_code('qutip')

# Create with Qiskit
code = create_steane_code('qiskit')

# Both support the same methods:
state = code.encode_logical_zero()
noisy = code.apply_depolarizing_noise(state, 0.01)
spectrum = code.compute_pauli_spectrum(noisy)
p_log = code.calculate_logical_error_rate(0.01, n_trials=100)
```

### 3. Runtime Switching

Switch backends dynamically in the IRC bot:

```
User: !backend
Bot: Current backend: QUTIP | Use: !backend [qutip|qiskit]

User: !backend qiskit
Bot: Switched to QISKIT backend

User: !threshold
Bot: Steane [[7,1,3]] pseudo-threshold: η_thr ≈ 9.30e-05 | Below this rate, QEC provides net benefit | Backend: QISKIT
```

### 4. Environment Configuration

Set default backend via environment variable:

```bash
# Use QuTiP (default)
python run_bot.py --demo

# Use Qiskit
export QEC_BACKEND=qiskit
python run_bot.py --demo
```

## Implementation Details

### Architecture

The implementation uses a factory pattern with two parallel implementations:

- `SteaneCode`: QuTiP-based implementation (original)
- `SteaneCodeQiskit`: Qiskit-based implementation (new)
- `create_steane_code(backend)`: Factory function for instantiation

### Key Files Modified

1. **src/qec_steane.py**
   - Added Qiskit imports with graceful fallback
   - Implemented `SteaneCodeQiskit` class
   - Added `create_steane_code()` factory function
   - Added `demo_backend_comparison()` function

2. **src/integrated_bot.py**
   - Added `backend` parameter to `__init__()`
   - Implemented `cmd_backend()` for runtime switching
   - Updated `cmd_threshold()` to show current backend
   - Added help text for `!backend` command

3. **requirements.txt**
   - Added `qiskit>=0.45.0`
   - Added `qiskit-aer>=0.13.0`

4. **tests/test_qec_backends.py** (new)
   - 13 comprehensive tests for both backends
   - Tests for factory function
   - Comparison tests between backends
   - Skip Qiskit tests if not installed

### Backward Compatibility

- QuTiP remains the default backend
- All existing code works without modification
- Qiskit is an optional dependency (gracefully handled if not installed)
- No breaking changes to existing API

## Usage Examples

### Example 1: Basic Backend Comparison

```python
from src.qec_steane import create_steane_code

# QuTiP backend
qutip_code = create_steane_code('qutip')
qutip_state = qutip_code.encode_logical_zero()
print(f"QuTiP state: {qutip_state.shape}")

# Qiskit backend
qiskit_code = create_steane_code('qiskit')
qiskit_state = qiskit_code.encode_logical_zero()
print(f"Qiskit state: {len(qiskit_state)} dimensions")
```

### Example 2: Error Rate Comparison

```python
from src.qec_steane import create_steane_code

backends = ['qutip', 'qiskit']
for backend in backends:
    code = create_steane_code(backend)
    p_log = code.calculate_logical_error_rate(0.01, n_trials=100)
    print(f"{backend.upper()}: p_log = {p_log:.6f}")
```

### Example 3: Run Demo Script

```bash
# Compare both backends
python examples/backend_comparison_demo.py

# Or use built-in comparison
python src/qec_steane.py --compare
```

## Testing

All functionality is fully tested:

```bash
# Run all tests
python -m pytest tests/

# Run only backend tests
python -m pytest tests/test_qec_backends.py -v

# Run with coverage
python -m pytest tests/test_qec_backends.py --cov=src.qec_steane
```

Test coverage:
- Factory function creation
- Backend initialization
- State encoding
- Noise application
- Pauli spectrum computation
- Logical error rate calculation
- Backend comparison
- Graceful fallback when Qiskit unavailable

## Performance Considerations

### QuTiP Backend
- **Pros**: Lightweight, fast for small systems, simpler installation
- **Cons**: Limited to simulation, no real hardware access
- **Best for**: Quick prototyping, education, small-scale simulations

### Qiskit Backend
- **Pros**: Industry standard, extensive ecosystem, real hardware integration
- **Cons**: Heavier installation, more complex setup
- **Best for**: Production use, integration with IBM Quantum, large-scale projects

## Future Enhancements

Potential additions:
- Additional QEC codes (Surface codes, Color codes)
- Hardware backend integration via IBM Quantum
- Performance benchmarking between backends
- Circuit visualization for Qiskit backend
- Transpiler optimization options
- Noise model customization

## Dependencies

### Required (QuTiP backend)
```
qutip>=4.6.0
numpy>=1.22.0
scipy>=1.7.0
```

### Optional (Qiskit backend)
```
qiskit>=0.45.0
qiskit-aer>=0.13.0
```

## References

- [QuTiP Documentation](http://qutip.org/docs/latest/)
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Steane Code Paper](https://arxiv.org/abs/quant-ph/9605021)

## Support

For issues or questions:
1. Check the documentation in `docs/IRC_BOT_GUIDE.md`
2. Run the demo: `python examples/backend_comparison_demo.py`
3. Open an issue on GitHub

---

**Last Updated**: 2025-10-11  
**Version**: 0.3.0  
**Status**: Production Ready ✓
