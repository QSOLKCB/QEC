"""Tiny coherent repetition-code correctness pilot; optional real cuStateVec.

Little-endian amplitudes. No mid-round syndrome measurement. Final ideal ZZ
syndrome projectors and conditional single-X recovery define the logical channel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

SCHEMA = 'qec.nvidia_coherent_memory.v1'
TOLERANCES = {'complex128': 1e-10, 'complex64': 2e-4}


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def matrix(name, theta=0):
    if name == 'X':
        return np.array([[0, 1], [1, 0]], complex)
    if name == 'H':
        return np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
    if name == 'S':
        return np.diag([1, 1j])
    if name == 'RX':
        return math.cos(theta/2)*np.eye(2) - 1j*math.sin(theta/2)*matrix('X')
    if name == 'RZ':
        return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])
    raise ValueError(name)


def circuit(kind, rounds, theta):
    gates = []
    if kind in ('memory_y', 'ghz_phase'):
        gates.append(('H', 0, None, 0.))
        if kind == 'memory_y':
            gates.append(('S', 0, None, 0.))
        gates += [('X', 1, 0, 0.), ('X', 2, 0, 0.)]
    if kind in ('memory_zero', 'memory_y'):
        gates += [('RX', q, None, theta) for _ in range(rounds) for q in range(3)]
    elif kind == 'ghz_phase':
        gates += [('RZ', 0, None, theta)] * rounds
        gates += [('X', 2, 0, 0.), ('X', 1, 0, 0.), ('H', 0, None, 0.)]
    else:
        raise ValueError(kind)
    return gates


def cpu_state(gates):
    """Independent full 8x8 operator construction, never used as GPU fallback."""
    state = np.eye(8, dtype=complex)[:, 0]
    for name, target, control, angle in gates:
        gate = matrix(name, angle)
        if control is None:
            full = np.array([[1.]])
            for q in reversed(range(3)):
                full = np.kron(full, gate if q == target else np.eye(2))
        else:
            full = np.zeros((8, 8), complex)
            for col in range(8):
                if not ((col >> control) & 1):
                    full[col, col] = 1
                else:
                    for bit in (0, 1):
                        row = (col & ~(1 << target)) | (bit << target)
                        full[row, col] = gate[bit, (col >> target) & 1]
        state = full @ state
    return state


def recover(state):
    """Project ZZ syndromes, apply minimum-weight X, trace out syndrome.

    Return unnormalized syndrome blocks and their sum; coherence within each
    block is retained. This is a quantum instrument, not majority relabeling.
    """
    blocks = []
    for syndrome in range(4):
        logical = np.zeros(2, complex)
        for basis in range(8):
            b = [(basis >> q) & 1 for q in range(3)]
            measured = (b[0] ^ b[1]) | ((b[1] ^ b[2]) << 1)
            if measured != syndrome:
                continue
            correction = {0: 0, 1: 1, 3: 2, 2: 4}[syndrome]
            corrected = basis ^ correction
            assert corrected in (0, 7)
            logical[int(corrected == 7)] += state[basis]
        blocks.append(np.outer(logical, logical.conj()))
    return np.array(blocks)


def analytic_blocks(kind, rounds, theta):
    c, s = math.cos(rounds*theta/2), math.sin(rounds*theta/2)
    amplitudes = [c**(3-w)*(-1j*s)**w for w in range(4)]
    initial = np.array([1., 0.], complex) if kind == 'memory_zero' else np.array([1., 1j])/math.sqrt(2)
    blocks = []
    for w in (0, 1, 1, 1):
        out = (amplitudes[w]*np.eye(2) + amplitudes[3-w]*matrix('X')) @ initial
        blocks.append(np.outer(out, out.conj()))
    return np.array(blocks)


def twirled_density(kind, rounds, theta):
    p = (1 - math.cos(theta)**rounds)/2
    failure = 3*p*p - 2*p*p*p
    v = np.array([1., 0.], complex) if kind == 'memory_zero' else np.array([1., 1j])/math.sqrt(2)
    rho = np.outer(v, v.conj())
    return (1-failure)*rho + failure*matrix('X') @ rho @ matrix('X')


def complex_json(a):
    return {'real': np.asarray(a).real.tolist(), 'imag': np.asarray(a).imag.tolist()}


class BackendUnavailable(RuntimeError):
    pass


def gpu_state(gates, precision):
    """Low-level cuStateVec apply_matrix on device pointers, with no fallback."""
    try:
        import cupy as cp
        import cuquantum
        from cuquantum.bindings import custatevec as sv
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise BackendUnavailable('No CUDA device visible')
            device = cp.cuda.Device()
            properties = cp.cuda.runtime.getDeviceProperties(device.id)
            free_before, total = cp.cuda.runtime.memGetInfo()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise BackendUnavailable(f'CUDA initialization failed: {exc}') from exc
    except (ImportError, OSError) as exc:
        raise BackendUnavailable(f'{type(exc).__name__}: {exc}') from exc
    dtype = np.dtype(precision)
    data_type = getattr(cuquantum.cudaDataType, 'CUDA_C_32F' if precision == 'complex64' else 'CUDA_C_64F')
    compute = getattr(cuquantum.ComputeType, 'COMPUTE_32F' if precision == 'complex64' else 'COMPUTE_64F')
    handle = sv.create()
    try:
        # Explicit default-stream agreement prevents host/device races.
        sv.set_stream(handle, cp.cuda.Stream.null.ptr)
        with cp.cuda.Stream.null:
            state = cp.zeros(8, dtype=dtype)
            state[0] = 1
            matrices = [np.ascontiguousarray(matrix(g[0], g[3]), dtype=dtype) for g in gates]
            sizes = [sv.apply_matrix_get_workspace_size(handle, data_type, 3, m.ctypes.data, data_type, sv.MatrixLayout.ROW, 0, 1, int(g[2] is not None), compute) for g, m in zip(gates, matrices)]
            workspace_size = max(sizes, default=0)
            if workspace_size + state.nbytes > min(64*1024**2, free_before//4):
                raise MemoryError('cuStateVec workspace exceeds conservative 64 MiB/free-memory cap')
            workspace = cp.cuda.alloc(workspace_size) if workspace_size else None
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            for gate, m in zip(gates, matrices):
                _, target, control, _ = gate
                controls = [] if control is None else [control]
                sv.apply_matrix(handle, state.data.ptr, data_type, 3, m.ctypes.data, data_type, sv.MatrixLayout.ROW, 0, [target], 1, controls, [1]*len(controls), len(controls), compute, workspace.ptr if workspace else 0, workspace_size)
            cp.cuda.Stream.null.synchronize()
            seconds = time.perf_counter() - start
            free_after, _ = cp.cuda.runtime.memGetInfo()
            result = cp.asnumpy(state)
        name = properties['name']
        if isinstance(name, bytes):
            name = name.decode()
        receipt = dict(device_name=name, device_ordinal=device.id, compute_capability=[properties['major'], properties['minor']], device_total_bytes=total, device_free_before_bytes=free_before, device_free_after_bytes=free_after, state_bytes=state.nbytes, workspace_bytes=workspace_size, memory_observation='snapshots, not process peak; includes other workloads', driver_version=cp.cuda.runtime.driverGetVersion(), runtime_version=cp.cuda.runtime.runtimeGetVersion(), cupy_version=cp.__version__, cuquantum_version=cuquantum.__version__, custatevec_version=sv.get_version(), synchronized_gate_wall_seconds=seconds)
        return result, receipt
    finally:
        sv.destroy(handle)


def run_experiment(*, backend='cpu', precision='complex128', rounds=(1, 3, 9), theta=.125):
    if backend not in ('cpu', 'cuquantum') or precision not in TOLERANCES:
        raise ValueError('unsupported backend or precision')
    if not rounds or len(rounds) > 16 or any(type(r) is not int or not 1 <= r <= 150 for r in rounds):
        raise ValueError('1 to 16 round counts, each integer within [1,150], required')
    if not math.isfinite(theta) or abs(theta) > math.pi:
        raise ValueError('theta must be finite within [-pi,pi]')
    if backend == 'cpu' and precision != 'complex128':
        raise ValueError('CPU reference is always complex128')
    receipt = dict(schema=SCHEMA, status='completed', backend=backend, gpu_verified=False, config=dict(precision=precision, rounds=list(rounds), theta=theta, qubits=3, intermediate_recovery=False), versions=dict(python=platform.python_version(), numpy=np.__version__), source_sha256={Path(__file__).name: digest(Path(__file__).read_bytes())}, tolerance=TOLERANCES[precision], records=[])
    receipt['config_sha256'] = digest(receipt['config'])
    for r in rounds:
        for kind in ('memory_zero', 'memory_y', 'ghz_phase'):
            gates = circuit(kind, r, theta)
            reference = cpu_state(gates)
            hardware = None
            if backend == 'cuquantum':
                try:
                    state, hardware = gpu_state(gates, precision)
                except BackendUnavailable as exc:
                    receipt.update(status='unavailable', reason=str(exc))
                    receipt['receipt_sha256'] = digest(receipt)
                    return receipt
                except Exception as exc:
                    receipt.update(status='failed', reason=f'GPU execution failed: {type(exc).__name__}: {exc}')
                    receipt['receipt_sha256'] = digest(receipt)
                    return receipt
            else:
                state = reference
            state = np.asarray(state)
            if state.shape != (8,) or not np.all(np.isfinite(state)):
                receipt.update(status='failed', reason='Backend returned nonfinite or wrong-shape state')
                receipt['receipt_sha256'] = digest(receipt)
                return receipt
            overlap = np.vdot(reference, state)
            aligned = state * np.exp(-1j*np.angle(overlap))
            error = float(np.max(np.abs(aligned-reference)))
            record = dict(state_vector=complex_json(state), kind=kind, rounds=r, circuit_sha256=digest(gates), state_sha256=digest(np.asarray(state).tobytes()), norm_error=float(abs(np.vdot(state,state).real-1)), cpu_phase_aligned_max_error=error, hardware=hardware)
            if kind.startswith('memory'):
                blocks = recover(state)
                expected = analytic_blocks(kind, r, theta)
                record.update(syndrome_logical_density=complex_json(blocks), syndrome_probabilities=np.trace(blocks, axis1=1, axis2=2).real.tolist(), recovered_logical_density=complex_json(blocks.sum(axis=0)), twirled_logical_density=complex_json(twirled_density(kind,r,theta)), analytic_max_error=float(np.max(np.abs(blocks-expected))))
            else:
                expected = np.zeros(8)
                expected[0] = math.cos(r*theta/2)**2
                expected[1] = math.sin(r*theta/2)**2
                record['analytic_max_error'] = float(np.max(np.abs(abs(state)**2-expected)))
            record['passed'] = max(record['norm_error'], error, record['analytic_max_error']) <= TOLERANCES[precision]
            receipt['records'].append(record)
    if not all(x['passed'] for x in receipt['records']):
        receipt['status'] = 'failed'
    receipt['gpu_verified'] = backend == 'cuquantum' and receipt['status'] == 'completed'
    receipt['receipt_sha256'] = digest(receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'cuquantum'], default='cpu')
    p.add_argument('--precision', choices=list(TOLERANCES), default='complex128')
    p.add_argument('--rounds', nargs='+', type=int, default=[1,3,9])
    p.add_argument('--theta', type=float, default=.125)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run_experiment(backend=args.backend, precision=args.precision, rounds=args.rounds, theta=args.theta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return {'completed': 0, 'unavailable': 2, 'failed': 1}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
