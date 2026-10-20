#!/usr/bin/env python3
"""Run the local correctness pilot and export an integrity-only evidence bundle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'src/bench/interop/nvidia_coherent_memory.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(path):
    """Verify archive member hashes and canonical receipts, without extracting."""
    if path.stat().st_size > 16*1024**2:
        raise ValueError('Archive exceeds 16-MiB verification limit')
    expected = path.with_suffix(path.suffix+'.sha256').read_text().split()[0]
    if sha(path.read_bytes()) != expected:
        raise ValueError('Archive SHA-256 mismatch')
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) > 32 or sum(i.file_size for i in z.infolist()) > 16*1024**2:
            raise ValueError('Bundle exceeds 32-member/16-MiB verification limit')
        names = z.namelist()
        if len(names) != len(set(names)) or 'manifest.json' not in names:
            raise ValueError('Duplicate members or missing manifest')
        manifest = json.loads(z.read('manifest.json'))
        if set(names) != set(manifest) | {'manifest.json'}:
            raise ValueError('Manifest membership mismatch')
        for name, expected_hash in manifest.items():
            if sha(z.read(name)) != expected_hash:
                raise ValueError(f'Member hash mismatch: {name}')
        science_passed = True
        common_config = None
        for name in ('cpu.json', 'gpu-complex64.json', 'gpu-complex128.json'):
            if name not in names:
                science_passed = False
                continue
            receipt = json.loads(z.read(name))
            expected_hash = receipt.pop('receipt_sha256')
            canonical = json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if sha(canonical) != expected_hash:
                raise ValueError(f'Receipt hash mismatch: {name}')
            if receipt['source_sha256'][Path(SOURCE).name] != sha(z.read(SOURCE)):
                raise ValueError(f'Source binding mismatch: {name}')
            precision = 'complex64' if name == 'gpu-complex64.json' else 'complex128'
            backend = 'cpu' if name == 'cpu.json' else 'cuquantum'
            tolerance = 2e-4 if precision == 'complex64' else 1e-10
            config = receipt['config']
            config_bytes = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            comparable = {k:v for k,v in config.items() if k != 'precision'}
            if common_config is None:
                common_config = comparable
            expected_cases = [(r,k) for r in config['rounds'] for k in ('memory_zero','memory_y','ghz_phase')]
            actual_cases = [(r['rounds'],r['kind']) for r in receipt['records']]
            science_passed &= (receipt['schema'] == 'qec.nvidia_coherent_memory.v1'
                and receipt['status'] == 'completed' and receipt['backend'] == backend
                and receipt['gpu_verified'] == (backend == 'cuquantum')
                and config['precision'] == precision and comparable == common_config
                and sha(config_bytes) == receipt['config_sha256']
                and receipt['tolerance'] == tolerance and bool(expected_cases)
                and actual_cases == expected_cases)
            for record in receipt['records']:
                science_passed &= record['passed'] is True
                for metric in ('norm_error','cpu_phase_aligned_max_error','analytic_max_error'):
                    value = record[metric]
                    science_passed &= math.isfinite(value) and 0 <= value <= tolerance
                science_passed &= (record['hardware'] is not None) == (backend == 'cuquantum')
        result = json.loads(z.read('run.json'))
        result['integrity_verified'] = True
        result['science_receipts_passed'] = bool(science_passed)
        return result


def run(output):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    head = git('rev-parse', 'HEAD')
    # Observe before creating the run outputs if user selected a path in repo.
    status = git('status', '--porcelain', '--untracked-files=normal')
    output.mkdir(parents=True, exist_ok=False)
    commands = [
        ('stim-preflight', [sys.executable, '-c', 'import stim; print(stim.__version__)']),
        ('cpu', [sys.executable, '-m', 'bench.interop.nvidia_coherent_memory', '--backend', 'cpu', '--output', str(output/'cpu.json')]),
        *[(f'gpu-{precision}', [sys.executable, '-m', 'bench.interop.nvidia_coherent_memory', '--backend', 'cuquantum', '--precision', precision, '--output', str(output/f'gpu-{precision}.json')]) for precision in ('complex64', 'complex128')],
        ('tests', [sys.executable, '-m', 'pytest', '-q', 'tests/test_nvidia_coherent_memory.py', 'tests/test_nvidia_evidence_bundle.py']),
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src')
    results = []
    for name, command in commands:
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            code, log = completed.returncode, completed.stdout
        except subprocess.TimeoutExpired as exc:
            captured = exc.stdout or ''
            if isinstance(captured, bytes):
                captured = captured.decode(errors='replace')
            code, log = 124, captured + '\nCommand exceeded 300-second limit.\n'
        (output/f'{name}.log').write_text(log)
        results.append(dict(name=name, command=command, exit_code=code))
    # Logs are deliberately narrow: no environment dump, credentials, home paths,
    # filesystem search or hardware serial numbers collected by this wrapper.
    metadata = dict(commit=head, worktree_clean=not bool(status), changed_path_status=status.splitlines(), commands=results, claim='integrity only, not authenticity or independent execution attestation', verified=not bool(status) and all(r['exit_code'] == 0 for r in results))
    inventory = sorted({(d.metadata.get('Name', 'unknown'), d.version) for d in importlib.metadata.distributions()})
    (output/'packages.json').write_text(json.dumps(inventory, indent=2)+'\n')
    metadata['head_after'] = git('rev-parse', 'HEAD')
    metadata['worktree_clean_after'] = not bool(git('status', '--porcelain', '--untracked-files=normal'))
    metadata['verified'] &= metadata['head_after'] == head and metadata['worktree_clean_after']
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    members = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    for name in (SOURCE, 'scripts/run_nvidia_pilot.py', 'tests/test_nvidia_coherent_memory.py', 'tests/test_nvidia_evidence_bundle.py', 'requirements-nvidia.txt'):
        members[name] = (ROOT/name).read_bytes()
    manifest = {name: sha(data) for name, data in members.items()}
    members['manifest.json'] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    archive = output/'evidence.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    archive.with_suffix('.zip.sha256').write_text(f'{sha(archive.read_bytes())}  {archive.name}\n')
    print(archive)
    return 0 if metadata['verified'] else 1


def replay(archive, output):
    """Rerun trusted local code on this machine; never execute archive code."""
    import numpy as np
    prior = verify(archive)
    with zipfile.ZipFile(archive) as z:
        if z.read(SOURCE) != (ROOT/SOURCE).read_bytes():
            raise ValueError('Check out the exact recorded source before replay')
        old = {name: json.loads(z.read(name)) for name in ('cpu.json', 'gpu-complex64.json', 'gpu-complex128.json')}
    if not prior['science_receipts_passed'] or not prior.get('verified', False):
        raise ValueError('Replay comparison requires previously accepted clean CPU/GPU/test run')
    code = run(output)
    comparisons = []
    for name, earlier in old.items():
        try:
            current = json.loads((output/name).read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            comparisons.append(dict(receipt=name, passed=False, reason=type(exc).__name__))
            continue
        if current['config'] != earlier['config']:
            raise ValueError('Replay config mismatch; this wrapper supports its default configuration only')
        passed = current['status'] == 'completed' and len(current['records']) == len(earlier['records'])
        maximum = 0.
        for a, b in zip(earlier['records'], current['records']):
            def state(record):
                v = record['state_vector']
                return np.asarray(v['real']) + 1j*np.asarray(v['imag'])
            x, y = state(a), state(b)
            y *= np.exp(-1j*np.angle(np.vdot(x,y)))
            error = float(np.max(np.abs(x-y)))
            maximum = max(maximum, error)
            passed &= a['kind'] == b['kind'] and a['rounds'] == b['rounds'] and error <= 2*earlier['tolerance']
        comparisons.append(dict(receipt=name, passed=bool(passed), max_phase_aligned_error=maximum))
    report = dict(prior_archive_sha256=sha(archive.read_bytes()), comparisons=comparisons, passed=code == 0 and all(c['passed'] for c in comparisons), scope='fresh CPU/GPU execution and phase-aligned state comparison; not hardware attestation')
    (output/'replay.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--verify', type=Path)
    group.add_argument('--replay', type=Path)
    args = parser.parse_args()
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
