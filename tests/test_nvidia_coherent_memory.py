"""Independent small-system controls; GPU availability never makes a CPU pass.

Stim is optional here but installed in the dedicated CPU CI lane. No test in
this module claims to have executed on NVIDIA hardware.
"""
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bench.interop import nvidia_coherent_memory as pilot


ROOT = Path(__file__).resolve().parents[1]
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1., -1.])
I = np.eye(2)


def _local(op, qubit):
    factors = [op if q == qubit else I for q in (2, 1, 0)]
    return np.kron(np.kron(factors[0], factors[1]), factors[2])


def _initial(kind):
    result = np.zeros(8, dtype=complex)
    if kind == 'memory_zero':
        result[0] = 1
    else:
        result[0], result[7] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    return result


def _project_and_recover(state):
    """Dense stabilizer projectors, independent of the implementation's masks."""
    z01 = _local(Z, 0) @ _local(Z, 1)
    z12 = _local(Z, 1) @ _local(Z, 2)
    embedding = np.eye(8, dtype=complex)[:, [0, 7]]
    corrections = (np.eye(8), _local(X, 0), _local(X, 2), _local(X, 1))
    blocks = []
    for syndrome, correction in enumerate(corrections):
        projector = (np.eye(8) + (-1)**(syndrome & 1) * z01) / 2
        projector = projector @ (np.eye(8) + (-1)**((syndrome >> 1) & 1) * z12) / 2
        logical = embedding.conj().T @ correction @ projector @ state
        blocks.append(np.outer(logical, logical.conj()))
    return np.asarray(blocks)


def _from_json(value):
    return np.asarray(value['real']) + 1j * np.asarray(value['imag'])


@pytest.mark.parametrize('kind', ['memory_zero', 'memory_y'])
@pytest.mark.parametrize('rounds,theta', [(1, 0), (1, .125), (3, .125), (9, -.31), (150, .021), (3, math.pi / 2)])
def test_dense_state_and_full_syndrome_instrument(kind, rounds, theta):
    angle = rounds * theta / 2
    rx = np.cos(angle) * I - 1j * np.sin(angle) * X
    expected_state = np.kron(np.kron(rx, rx), rx) @ _initial(kind)
    actual = pilot.cpu_state(pilot.circuit(kind, rounds, theta))
    np.testing.assert_allclose(actual, expected_state, atol=2e-13, rtol=0)
    blocks = _project_and_recover(expected_state)
    np.testing.assert_allclose(pilot.recover(actual), blocks, atol=2e-13, rtol=0)
    np.testing.assert_allclose(pilot.analytic_blocks(kind, rounds, theta), blocks, atol=2e-13, rtol=0)
    assert np.trace(blocks.sum(axis=0)).real == pytest.approx(1, abs=2e-13)
    for block in blocks:
        np.testing.assert_allclose(block, block.conj().T, atol=2e-13)
        assert np.linalg.eigvalsh(block).min() >= -2e-13


def test_recovery_corrects_each_single_x_and_preserves_quantum_coherence():
    initial = _initial('memory_y')
    rho = np.outer(initial[[0, 7]], initial[[0, 7]].conj())
    for q in (0, 1, 2):
        blocks = pilot.recover(_local(X, q) @ initial)
        np.testing.assert_allclose(blocks.sum(axis=0), rho, atol=1e-14)
        assert np.count_nonzero(np.trace(blocks, axis1=1, axis2=2).real > .5) == 1
    # A population-only majority label would erase this off-diagonal element.
    assert abs(rho[0, 1]) == pytest.approx(.5)


@pytest.mark.parametrize('rounds,theta', [(1, .125), (3, .125), (9, .31), (150, -.021)])
def test_majority_formula_and_coherent_logical_off_diagonal(rounds, theta):
    state = pilot.cpu_state(pilot.circuit('memory_zero', rounds, theta))
    density = pilot.recover(state).sum(axis=0)
    p = math.sin(rounds * theta / 2)**2
    assert density[1, 1].real == pytest.approx(3*p*p - 2*p*p*p, abs=2e-13)
    if abs(math.sin(rounds * theta)) > .01:
        assert abs(density[0, 1]) > 1e-8
    twirled = pilot.twirled_density('memory_zero', rounds, theta)
    q = (1 - math.cos(theta)**rounds) / 2
    assert twirled[1, 1].real == pytest.approx(3*q*q - 2*q*q*q, abs=2e-13)
    assert twirled[0, 1] == 0


def test_single_round_population_match_is_not_channel_equivalence():
    theta = .6
    coherent_zero = pilot.recover(pilot.cpu_state(pilot.circuit('memory_zero', 1, theta))).sum(axis=0)
    twirled_zero = pilot.twirled_density('memory_zero', 1, theta)
    np.testing.assert_allclose(np.diag(coherent_zero), np.diag(twirled_zero), atol=1e-14)
    assert np.max(np.abs(coherent_zero - twirled_zero)) > .01
    coherent_y = pilot.recover(pilot.cpu_state(pilot.circuit('memory_y', 1, theta))).sum(axis=0)
    assert np.max(np.abs(coherent_y - pilot.twirled_density('memory_y', 1, theta))) > .01


@pytest.mark.parametrize('rounds,theta', [(1, 0), (1, .125), (3, .125), (9, -.4), (150, .013)])
def test_ghz_phase_probe_interference(rounds, theta):
    state = pilot.cpu_state(pilot.circuit('ghz_phase', rounds, theta))
    expected = np.zeros(8, dtype=complex)
    expected[0] = math.cos(rounds * theta / 2)
    expected[1] = -1j * math.sin(rounds * theta / 2)
    np.testing.assert_allclose(state, expected, atol=2e-13, rtol=0)


def _stim_preparation(stim, kind):
    circuit = stim.Circuit('I 0 1 2')
    if kind != 'memory_zero':
        circuit.append('H', [0])
        if kind == 'memory_y':
            circuit.append('S', [0])
        circuit.append('CX', [0, 1, 0, 2])
    return circuit


def _stim_state(stim, circuit):
    simulator = stim.TableauSimulator()
    simulator.do(circuit)
    return np.asarray(simulator.state_vector(endian='little'), dtype=complex)


@pytest.mark.parametrize('kind', ['memory_zero', 'memory_y', 'ghz_phase'])
@pytest.mark.parametrize('rounds', [1, 3])
@pytest.mark.parametrize('theta', [0., math.pi / 2, -math.pi / 2, math.pi])
def test_actual_stim_clifford_state_overlap(kind, rounds, theta):
    stim = pytest.importorskip('stim')
    circuit = _stim_preparation(stim, kind)
    rx = {0.: 'I', math.pi / 2: 'SQRT_X', -math.pi / 2: 'SQRT_X_DAG', math.pi: 'X'}
    rz = {0.: 'I', math.pi / 2: 'S', -math.pi / 2: 'S_DAG', math.pi: 'Z'}
    for _ in range(rounds):
        circuit.append(rz[theta] if kind == 'ghz_phase' else rx[theta], [0] if kind == 'ghz_phase' else [0, 1, 2])
    if kind == 'ghz_phase':
        circuit.append('CX', [0, 2, 0, 1])
        circuit.append('H', [0])
    expected = _stim_state(stim, circuit)
    actual = pilot.cpu_state(pilot.circuit(kind, rounds, theta))
    # Stim returns complex64 and is free to choose a different global phase.
    assert abs(np.vdot(expected, actual)) == pytest.approx(1, abs=2e-7)
    if kind.startswith('memory'):
        np.testing.assert_allclose(pilot.recover(actual), _project_and_recover(expected), atol=2e-7, rtol=0)


@pytest.mark.parametrize('kind', ['memory_zero', 'memory_y'])
@pytest.mark.parametrize('rounds', [1, 2])
def test_exact_weighted_stim_pauli_histories(kind, rounds):
    stim = pytest.importorskip('stim')
    theta = .43
    p = math.sin(theta / 2)**2
    density = np.zeros((2, 2), dtype=complex)
    # Enumerate every deterministic history, not seeded stochastic samples.
    # This independently checks composition by parity across rounds.
    for history in product((0, 1), repeat=3*rounds):
        circuit = _stim_preparation(stim, kind)
        for index, event in enumerate(history):
            if event:
                circuit.append('X', [index % 3])
        count = sum(history)
        weight = p**count * (1-p)**(len(history)-count)
        density += weight * _project_and_recover(_stim_state(stim, circuit)).sum(axis=0)
    np.testing.assert_allclose(pilot.twirled_density(kind, rounds, theta), density, atol=2e-7, rtol=0)


@pytest.mark.parametrize('bad', [
    {'backend': 'auto'}, {'precision': 'float64'}, {'precision': 'complex64'},
    {'rounds': ()}, {'rounds': (0,)}, {'rounds': (151,)}, {'rounds': (True,)},
    {'rounds': (1.5,)}, {'rounds': tuple(range(1, 18))},
    {'theta': float('nan')}, {'theta': float('inf')}, {'theta': math.pi + .01},
])
def test_input_preflight_rejects_before_allocation(bad, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError('invalid request reached state allocation')
    monkeypatch.setattr(pilot, 'cpu_state', must_not_run)
    monkeypatch.setattr(pilot, 'gpu_state', must_not_run)
    with pytest.raises(ValueError):
        pilot.run_experiment(**bad)


def test_cpu_receipt_replay_integrity_and_controls():
    receipt = pilot.run_experiment(rounds=(1, 3, 9))
    assert receipt == pilot.run_experiment(rounds=(1, 3, 9))
    assert receipt['schema'] == 'qec.nvidia_coherent_memory.v1'
    assert receipt['status'] == 'completed'
    assert receipt['gpu_verified'] is False
    assert len(receipt['records']) == 9
    assert all(record['passed'] and record['hardware'] is None for record in receipt['records'])
    expected_hash = receipt.pop('receipt_sha256')
    assert pilot.digest(receipt) == expected_hash
    assert receipt['config_sha256'] == pilot.digest(receipt['config'])
    assert receipt['source_sha256']['nvidia_coherent_memory.py'] == pilot.digest(Path(pilot.__file__).read_bytes())
    for record in receipt['records']:
        if record['kind'].startswith('memory'):
            rho = _from_json(record['recovered_logical_density'])
            blocks = _from_json(record['syndrome_logical_density'])
            assert blocks.shape == (4, 2, 2)
            np.testing.assert_allclose(blocks.sum(axis=0), rho, atol=1e-14)
            np.testing.assert_allclose(np.trace(blocks, axis1=1, axis2=2).real, record['syndrome_probabilities'], atol=1e-14)
            assert np.trace(rho).real == pytest.approx(1, abs=1e-12)
            assert sum(record['syndrome_probabilities']) == pytest.approx(1, abs=1e-12)


def test_missing_gpu_is_explicit_and_never_verified(monkeypatch):
    def unavailable(*args):
        raise pilot.BackendUnavailable('test: no CUDA device')
    monkeypatch.setattr(pilot, 'gpu_state', unavailable)
    receipt = pilot.run_experiment(backend='cuquantum', precision='complex64')
    assert receipt['status'] == 'unavailable'
    assert receipt['backend'] == 'cuquantum'
    assert receipt['gpu_verified'] is False
    assert receipt['records'] == []
    assert 'no CUDA device' in receipt['reason']
    assert pilot.digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) == receipt['receipt_sha256']


def test_gpu_runtime_error_is_not_an_unavailable_or_cpu_success(monkeypatch):
    def failure(*args):
        raise RuntimeError('test: kernel failure')
    monkeypatch.setattr(pilot, 'gpu_state', failure)
    receipt = pilot.run_experiment(backend='cuquantum')
    assert receipt['status'] == 'failed'
    assert receipt['gpu_verified'] is False
    assert 'kernel failure' in receipt['reason']


def test_bad_gpu_numerics_fail_validation(monkeypatch):
    def corrupted(gates, precision):
        return pilot.cpu_state(gates) * 1.1, {'test_double': True}
    monkeypatch.setattr(pilot, 'gpu_state', corrupted)
    receipt = pilot.run_experiment(backend='cuquantum', rounds=(1,))
    assert receipt['status'] == 'failed'
    assert receipt['gpu_verified'] is False
    assert not any(record['passed'] for record in receipt['records'])


def test_cpu_import_has_no_optional_nvidia_imports():
    code = "from bench.interop import nvidia_coherent_memory; import sys; assert not any(x.split('.')[0] in ('cupy', 'cuquantum') for x in sys.modules)"
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, check=True, capture_output=True, text=True)


def test_cli_cpu_receipt(tmp_path):
    path = tmp_path / 'nested' / 'cpu.json'
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    completed = subprocess.run([sys.executable, '-m', 'bench.interop.nvidia_coherent_memory', '--backend', 'cpu', '--rounds', '1', '3', '--theta', '.125', '--output', str(path)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'completed' and receipt['gpu_verified'] is False
    assert len(receipt['records']) == 6


@pytest.mark.parametrize('precision', ['complex64', 'complex128'])
@pytest.mark.parametrize('kind', ['memory_zero', 'memory_y', 'ghz_phase'])
def test_gate_plan_reuses_buffers_without_changing_order(kind, precision):
    gates = pilot.circuit(kind, 150, -.125)
    queries = []
    def query(m, controls):
        queries.append((m, controls))
        return 128 if controls else 64
    plan, size = pilot._prepare_gates(gates, np.dtype(precision), query)
    assert len(plan) == len(gates)
    assert len(queries) == len({(g[0], g[3], g[2] is not None) for g in gates})
    assert size == max(128 if g[2] is not None else 64 for g in gates)
    buffers = {}
    for gate, (target, control, m) in zip(gates, plan):
        name, expected_target, expected_control, angle = gate
        assert (target, control) == (expected_target, expected_control)
        assert m.dtype == np.dtype(precision) and m.flags.c_contiguous
        np.testing.assert_array_equal(m, pilot.matrix(name, angle).astype(precision))
        key = (name, angle)
        assert buffers.setdefault(key, m) is m
    assert len(buffers) == len({(g[0], g[3]) for g in gates})


def test_gate_plan_separates_angles_controls_and_execution_lifetimes():
    gates = [('RX', 0, None, .1), ('RX', 2, 1, .1),
             ('RX', 1, None, -.1), ('RX', 2, None, .1)]
    queries = []
    def query(m, controls):
        queries.append((m, controls))
        return 32 * controls
    first, size = pilot._prepare_gates(gates, np.dtype('complex128'), query)
    assert len(queries) == 3 and size == 32
    assert first[0][2] is first[1][2] is first[3][2]
    assert first[0][2] is not first[2][2]
    second, _ = pilot._prepare_gates(gates, np.dtype('complex64'), query)
    assert second[0][2] is not first[0][2]
    assert second[0][2].dtype == np.dtype('complex64')
    assert pilot._prepare_gates([], np.dtype('complex128'), query) == ([], 0)
