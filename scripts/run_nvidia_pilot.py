#!/usr/bin/env python3
"""Run the local correctness pilot and export an integrity-only evidence bundle."""
from __future__ import annotations

import argparse
import cmath
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'src/bench/interop/nvidia_coherent_memory.py'
RECEIPTS = ('cpu.json', 'gpu-complex64.json', 'gpu-complex128.json')
KINDS = ('memory_zero', 'memory_y', 'ghz_phase')
DEPENDENCIES = ('numpy', 'stim', 'pytest', 'cupy', 'cuquantum')
MAX_BUNDLE_BYTES = 16 * 1024**2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def strict_json(data):
    """Reject ambiguous or nonfinite JSON before checking its canonical digest."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON value: {value}')

    result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)
    canonical(result)  # Also rejects overflowing numeric literals such as 1e400.
    return result


def dependency_preflight():
    """Inspect dependencies using only the stdlib; never import NumPy or CUDA."""
    python = tuple(sys.version_info[:2])
    supported = (3, 11) <= python <= (3, 14)
    profile = 'requirements/memory-py311.txt' if python == (3, 11) else 'requirements/memory-py312plus.txt'
    available = {}
    for name in DEPENDENCIES:
        try:
            available[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available[name] = False
    missing = [name for name, found in available.items() if not found]
    guidance = [
        f'Use Python 3.11 with requirements/memory-py311.txt, or Python 3.12-3.14 with requirements/memory-py312plus.txt.',
        f'{sys.executable} -m pip install -r {profile} pytest',
        f'{sys.executable} -m pip install -r requirements/nvidia-cu13.txt',
        'GPU execution also requires a compatible NVIDIA driver and visible CUDA device; there is no CPU fallback.',
    ]
    return dict(python='.'.join(map(str, sys.version_info[:3])), supported_python=supported,
                profile=profile, available=available, missing=missing,
                passed=supported and not missing, guidance=guidance)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def numeric_array(value, shape):
    if not shape:
        return finite_number(value)
    return (isinstance(value, list) and len(value) == shape[0]
            and all(numeric_array(item, shape[1:]) for item in value))


def complex_array(value, shape):
    return (isinstance(value, dict) and set(value) == {'real', 'imag'}
            and all(numeric_array(value[part], shape) for part in ('real', 'imag')))


def state_vector(record):
    value = record['state_vector']
    return [complex(real, imag) for real, imag in zip(value['real'], value['imag'])]


def expected_gates(kind, rounds, theta):
    """The v1 receipt circuit contract, not imported or executed archived code."""
    gates = []
    if kind in ('memory_y', 'ghz_phase'):
        gates.append(('H', 0, None, 0.))
        if kind == 'memory_y':
            gates.append(('S', 0, None, 0.))
        gates += [('X', 1, 0, 0.), ('X', 2, 0, 0.)]
    if kind in ('memory_zero', 'memory_y'):
        gates += [('RX', qubit, None, theta) for _ in range(rounds) for qubit in range(3)]
    else:
        gates += [('RZ', 0, None, theta)] * rounds
        gates += [('X', 2, 0, 0.), ('X', 1, 0, 0.), ('H', 0, None, 0.)]
    return gates


def verify_receipt(receipt, source, name):
    """Check v1 integrity and scientific acceptance fields without executing code.

    Serialized state hashes use the recorded precision. V1 recorded native-endian
    NumPy bytes but no byte-order field, so either IEEE little/big-endian encoding
    is accepted. This is a consistency check, not independent hardware evidence.
    """
    if not isinstance(receipt, dict):
        raise ValueError(f'Invalid receipt object: {name}')
    body = {key: value for key, value in receipt.items() if key != 'receipt_sha256'}
    if sha(canonical(body)) != receipt.get('receipt_sha256'):
        raise ValueError(f'Receipt hash mismatch: {name}')
    if receipt.get('source_sha256') != {Path(SOURCE).name: sha(source)}:
        raise ValueError(f'Source binding mismatch: {name}')
    if name not in RECEIPTS:
        raise ValueError(f'Unknown receipt member: {name}')
    precision = 'complex64' if name == 'gpu-complex64.json' else 'complex128'
    backend = 'cpu' if name == 'cpu.json' else 'cuquantum'
    tolerance = 2e-4 if precision == 'complex64' else 1e-10
    config = receipt.get('config')
    if not isinstance(config, dict) or set(config) != {'precision', 'rounds', 'theta', 'qubits', 'intermediate_recovery'}:
        return False
    rounds = config['rounds']
    valid_config = (config['precision'] == precision and type(config['qubits']) is int
                    and config['qubits'] == 3 and config['intermediate_recovery'] is False
                    and isinstance(rounds, list) and 1 <= len(rounds) <= 16
                    and all(type(r) is int and 1 <= r <= 150 for r in rounds)
                    and finite_number(config['theta']) and abs(config['theta']) <= math.pi)
    if not valid_config:
        return False
    if not (receipt.get('schema') == 'qec.nvidia_coherent_memory.v1'
            and receipt.get('status') == 'completed' and receipt.get('backend') == backend
            and receipt.get('gpu_verified') is (backend == 'cuquantum')
            and sha(canonical(config)) == receipt.get('config_sha256')
            and finite_number(receipt.get('tolerance')) and receipt['tolerance'] == tolerance):
        return False
    records = receipt.get('records')
    expected_cases = [(r, kind) for r in rounds for kind in KINDS]
    if not isinstance(records, list) or len(records) != len(expected_cases):
        return False
    for record, (r, kind) in zip(records, expected_cases):
        if (not isinstance(record, dict) or type(record.get('rounds')) is not int
                or record['rounds'] != r or record.get('kind') != kind or record.get('passed') is not True):
            return False
        if backend == 'cuquantum':
            if not isinstance(record.get('hardware'), dict) or not record['hardware']:
                return False
        elif 'hardware' not in record or record['hardware'] is not None:
            return False
        for metric in ('norm_error', 'cpu_phase_aligned_max_error', 'analytic_max_error'):
            value = record.get(metric)
            if not finite_number(value) or not 0 <= value <= tolerance:
                return False
        if not complex_array(record.get('state_vector'), (8,)):
            return False
        vector = state_vector(record)
        try:
            norm = math.fsum(abs(value)**2 for value in vector)
            components = [component for value in vector for component in (value.real, value.imag)]
            scalar_format = 'f' if precision == 'complex64' else 'd'
            hashes = {sha(struct.pack(endian + scalar_format * 16, *components)) for endian in ('<', '>')}
        except (OverflowError, struct.error):
            return False
        if not math.isfinite(norm) or abs(norm - 1) > tolerance or record.get('state_sha256') not in hashes:
            return False
        if record.get('circuit_sha256') != sha(canonical(expected_gates(kind, r, config['theta']))):
            return False
        if kind.startswith('memory'):
            shapes = {'syndrome_logical_density': (4, 2, 2), 'recovered_logical_density': (2, 2),
                      'twirled_logical_density': (2, 2)}
            if not all(complex_array(record.get(field), shape) for field, shape in shapes.items()):
                return False
            probabilities = record.get('syndrome_probabilities')
            if (not numeric_array(probabilities, (4,)) or any(p < -tolerance or p > 1 + tolerance for p in probabilities)
                    or abs(math.fsum(probabilities) - 1) > tolerance):
                return False
    return True


def science_receipts(members):
    passed = True
    common_config = None
    for name in RECEIPTS:
        if name not in members:
            passed = False
            continue
        if SOURCE not in members:
            raise ValueError('Missing source bound to receipt')
        receipt = strict_json(members[name])
        accepted = verify_receipt(receipt, members[SOURCE], name)
        passed &= accepted
        if accepted:
            comparable = {key: value for key, value in receipt['config'].items() if key != 'precision'}
            if common_config is None:
                common_config = comparable
            passed &= comparable == common_config
    return bool(passed)


def verify(path):
    """Verify historical/current bytes and receipts; never extract/import code."""
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError('Archive exceeds 16-MiB verification limit')
    checksum = path.with_suffix(path.suffix + '.sha256').read_text().split()
    if not checksum or sha(path.read_bytes()) != checksum[0]:
        raise ValueError('Archive SHA-256 mismatch')
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 32 or sum(item.file_size for item in archive.infolist()) > MAX_BUNDLE_BYTES:
            raise ValueError('Bundle exceeds 32-member/16-MiB verification limit')
        names = archive.namelist()
        if len(names) != len(set(names)) or 'manifest.json' not in names:
            raise ValueError('Duplicate members or missing manifest')
        manifest = strict_json(archive.read('manifest.json'))
        if not isinstance(manifest, dict) or set(names) != set(manifest) | {'manifest.json'}:
            raise ValueError('Manifest membership mismatch')
        members = {name: archive.read(name) for name in manifest}
        for name, expected_hash in manifest.items():
            if sha(members[name]) != expected_hash:
                raise ValueError(f'Member hash mismatch: {name}')
        if 'run.json' not in members:
            raise ValueError('Missing run.json')
        result = strict_json(members['run.json'])
        if not isinstance(result, dict):
            raise ValueError('Invalid run metadata')
        result['integrity_verified'] = True
        result['science_receipts_passed'] = science_receipts(members)
        return result


def run(output):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()

    preflight = dependency_preflight()
    head = git('rev-parse', 'HEAD')
    # Observe before creating the run outputs if user selected a path in repo.
    status = git('status', '--porcelain', '--untracked-files=normal')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'dependency-preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    commands = [
        ('stim-preflight', [sys.executable, '-c', 'import stim; print(stim.__version__)'], ('stim',)),
        ('cpu', [sys.executable, '-m', 'bench.interop.nvidia_coherent_memory', '--backend', 'cpu', '--output', str(output / 'cpu.json')], ('numpy',)),
        *[(f'gpu-{precision}', [sys.executable, '-m', 'bench.interop.nvidia_coherent_memory', '--backend', 'cuquantum', '--precision', precision, '--output', str(output / f'gpu-{precision}.json')], ('numpy', 'cupy', 'cuquantum')) for precision in ('complex64', 'complex128')],
        ('cpu-tests', [sys.executable, '-m', 'pytest', '-q', 'tests/test_nvidia_coherent_memory.py', 'tests/test_nvidia_evidence_bundle.py'], ('numpy', 'stim', 'pytest')),
        ('gpu-tests', [sys.executable, '-m', 'pytest', '-q', 'tests/test_nvidia_gpu_acceptance.py'], ('numpy', 'pytest', 'cupy', 'cuquantum')),
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src')
    results = []
    for name, command, dependencies in commands:
        command_env = env.copy()
        # GPU tests must fail on unavailable hardware; they cannot skip to green.
        if name == 'gpu-tests':
            command_env['QEC_REQUIRE_GPU'] = '1'
        missing = [dependency for dependency in dependencies if not preflight['available'][dependency]]
        if not preflight['supported_python'] or missing:
            code = 2
            log = ('Dependency preflight failed: ' + (', '.join(missing) if preflight['supported_python'] else 'unsupported Python version')
                   + '\n' + '\n'.join(preflight['guidance']) + '\n')
        else:
            try:
                completed = subprocess.run(command, cwd=ROOT, env=command_env, text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
                code, log = completed.returncode, completed.stdout
            except subprocess.TimeoutExpired as exc:
                captured = exc.stdout or ''
                if isinstance(captured, bytes):
                    captured = captured.decode(errors='replace')
                code, log = 124, captured + '\nCommand exceeded 300-second limit.\n'
        (output / f'{name}.log').write_text(log)
        result = dict(name=name, command=command, exit_code=code)
        if name == 'gpu-tests':
            result['required_environment'] = {'QEC_REQUIRE_GPU': '1'}
        results.append(result)
    # Narrow logs only: no environment dump, credentials, filesystem search or
    # hardware serial numbers are collected by this wrapper.
    metadata = dict(commit=head, worktree_clean=not bool(status), changed_path_status=status.splitlines(),
                    commands=results, dependency_preflight_passed=preflight['passed'],
                    claim='integrity only, not authenticity or independent execution attestation',
                    verified=not bool(status) and preflight['passed'] and all(r['exit_code'] == 0 for r in results))
    inventory = sorted({(distribution.metadata.get('Name', 'unknown'), distribution.version)
                        for distribution in importlib.metadata.distributions()})
    (output / 'packages.json').write_text(json.dumps(inventory, indent=2) + '\n')
    metadata['head_after'] = git('rev-parse', 'HEAD')
    metadata['worktree_clean_after'] = not bool(git('status', '--porcelain', '--untracked-files=normal'))
    metadata['verified'] &= metadata['head_after'] == head and metadata['worktree_clean_after']
    members = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    for name in (SOURCE, 'scripts/run_nvidia_pilot.py', 'tests/test_nvidia_coherent_memory.py',
                 'tests/test_nvidia_evidence_bundle.py', 'tests/test_nvidia_gpu_acceptance.py',
                 'requirements/memory-py312plus.txt', 'requirements/memory-py311.txt', 'requirements/nvidia-cu13.txt',
                 'tests/fixtures/nvidia_coherent_memory/cpu_v1.json',
                 'tests/fixtures/nvidia_coherent_memory/source_v1.py',
                 'tests/fixtures/nvidia_coherent_memory/cpu_v2.json'):
        members[name] = (ROOT / name).read_bytes()
    metadata['science_receipts_passed'] = science_receipts(members)
    metadata['verified'] &= metadata['science_receipts_passed']
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    members['run.json'] = (output / 'run.json').read_bytes()
    manifest = {name: sha(data) for name, data in members.items()}
    members['manifest.json'] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    archive = output / 'evidence.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for name, data in members.items():
            zipped.writestr(name, data)
    archive.with_suffix('.zip.sha256').write_text(f'{sha(archive.read_bytes())}  {archive.name}\n')
    print(archive)
    return 0 if metadata['verified'] else 1


def replay(archive, output):
    """Rerun trusted, exactly matching local source; never execute archive code."""
    prior = verify(archive)
    with zipfile.ZipFile(archive) as zipped:
        if SOURCE not in zipped.namelist() or zipped.read(SOURCE) != (ROOT / SOURCE).read_bytes():
            raise ValueError('Check out the exact recorded source before replay')
        if not prior['science_receipts_passed'] or prior.get('verified') is not True:
            raise ValueError('Replay comparison requires previously accepted clean CPU/GPU/test run')
        old = {name: strict_json(zipped.read(name)) for name in RECEIPTS}
    code = run(output)
    comparisons = []
    for name, earlier in old.items():
        try:
            current = strict_json((output / name).read_bytes())
            accepted = verify_receipt(current, (ROOT / SOURCE).read_bytes(), name)
        except (FileNotFoundError, ValueError) as exc:
            comparisons.append(dict(receipt=name, passed=False, reason=type(exc).__name__))
            continue
        if current.get('config') != earlier['config']:
            raise ValueError('Replay config mismatch; this wrapper supports its default configuration only')
        passed = accepted and len(current['records']) == len(earlier['records'])
        maximum = 0.
        if accepted:
            for before, after in zip(earlier['records'], current['records']):
                x, y = state_vector(before), state_vector(after)
                overlap = sum(a.conjugate() * b for a, b in zip(x, y))
                phase = cmath.exp(-1j * cmath.phase(overlap))
                error = max(abs(a - b * phase) for a, b in zip(x, y))
                maximum = max(maximum, error)
                passed &= (before['kind'] == after['kind'] and before['rounds'] == after['rounds']
                           and math.isfinite(error) and error <= 2 * earlier['tolerance'])
        comparisons.append(dict(receipt=name, passed=bool(passed), max_phase_aligned_error=maximum))
    report = dict(prior_archive_sha256=sha(archive.read_bytes()), comparisons=comparisons,
                  passed=code == 0 and all(comparison['passed'] for comparison in comparisons),
                  scope='fresh CPU/GPU execution and phase-aligned state comparison; not hardware attestation')
    (output / 'replay.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--verify', type=Path)
    group.add_argument('--replay', type=Path)
    group.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        if args.output_dir:
            parser.error('--preflight does not use --output-dir')
        result = dependency_preflight()
        print(json.dumps(result, indent=2))
        return 0 if result['passed'] else 1
    if args.verify:
        if args.output_dir:
            parser.error('--verify does not use --output-dir')
        print(json.dumps(verify(args.verify), indent=2))
        return 0
    if not args.output_dir:
        parser.error('--output-dir is required for execution or replay')
    if args.replay:
        return replay(args.replay, args.output_dir.resolve())
    return run(args.output_dir.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
