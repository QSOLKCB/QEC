"""The bundle attests byte integrity, not trusted execution or real hardware."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from bench.interop.nvidia_coherent_memory import run_experiment

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/nvidia_coherent_memory'
spec = importlib.util.spec_from_file_location('pilot_bundle', ROOT / 'scripts/run_nvidia_pilot.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def write_bundle(path, members):
    members = {**members, 'manifest.json': json.dumps({name: bundle.sha(data) for name, data in members.items()}).encode()}
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    path.with_suffix('.zip.sha256').write_text(bundle.sha(path.read_bytes()) + '  evidence.zip\n')
    return path


def fixture_bundle(tmp_path):
    members = {bundle.SOURCE: (ROOT / bundle.SOURCE).read_bytes(), 'run.json': b'{"verified": false}'}
    receipt = run_experiment()
    # These are explicitly CPU records, not fake GPU success.
    for name in bundle.RECEIPTS:
        members[name] = json.dumps(receipt).encode()
    return write_bundle(tmp_path / 'evidence.zip', members)


def resign(receipt):
    receipt.pop('receipt_sha256', None)
    receipt['receipt_sha256'] = bundle.sha(bundle.canonical(receipt))
    return receipt


def test_integrity_not_gpu_authenticity(tmp_path):
    result = bundle.verify(fixture_bundle(tmp_path))
    assert result['integrity_verified']
    assert not result['science_receipts_passed']


def test_archive_tamper(tmp_path):
    path = fixture_bundle(tmp_path)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Archive SHA'):
        bundle.verify(path)


def test_member_tamper_even_with_updated_zip_hash(tmp_path):
    path = fixture_bundle(tmp_path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members['run.json'] = b'{"verified": true}'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    path.with_suffix('.zip.sha256').write_text(bundle.sha(path.read_bytes()))
    with pytest.raises(ValueError, match='Member hash mismatch'):
        bundle.verify(path)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_backend_state_fails_closed(monkeypatch, bad):
    import numpy as np
    from bench.interop import nvidia_coherent_memory as pilot
    monkeypatch.setattr(pilot, 'gpu_state', lambda *args: (np.full(8, bad, complex), {'test_double': True}))
    receipt = pilot.run_experiment(backend='cuquantum', rounds=(1,))
    assert receipt['status'] == 'failed'
    assert not receipt['gpu_verified']
    assert 'nonfinite' in receipt['reason']
    json.dumps(receipt, allow_nan=False)


def test_incomplete_failure_bundle_can_verify(tmp_path):
    path = write_bundle(tmp_path / 'evidence.zip', {'run.json': b'{"verified": false}'})
    result = bundle.verify(path)
    assert result['integrity_verified'] and not result['science_receipts_passed']


def test_replay_rejects_source_mismatch_before_execution(tmp_path, monkeypatch):
    path = fixture_bundle(tmp_path)
    fake_root = tmp_path / 'other'
    (fake_root / Path(bundle.SOURCE).parent).mkdir(parents=True)
    (fake_root / bundle.SOURCE).write_text('different source')
    monkeypatch.setattr(bundle, 'ROOT', fake_root)
    monkeypatch.setattr(bundle, 'run', lambda *args: pytest.fail('Mismatched source reached execution'))
    with pytest.raises(ValueError, match='exact recorded source'):
        bundle.replay(path, tmp_path / 'replay')


def test_historical_fixture_is_bound_to_inert_source_snapshot():
    """Read historical bytes only. Never import or execute source_v1.py."""
    source = (FIXTURES / 'source_v1.py').read_bytes()
    stored = bundle.strict_json((FIXTURES / 'cpu_v1.json').read_bytes())
    assert stored['receipt_sha256'] == 'ea3961efd3f7110eb5ad3bfa6035da5bffc2048a67eaf6fff00b66db9f76d277'
    assert bundle.verify_receipt(stored, source, 'cpu.json')
    assert source != (ROOT / bundle.SOURCE).read_bytes()
    with pytest.raises(ValueError, match='Source binding mismatch'):
        bundle.verify_receipt(stored, (ROOT / bundle.SOURCE).read_bytes(), 'cpu.json')


def test_historical_archive_verifies_without_numpy_or_execution(tmp_path):
    path = write_bundle(tmp_path / 'historical.zip', {
        bundle.SOURCE: (FIXTURES / 'source_v1.py').read_bytes(),
        'cpu.json': (FIXTURES / 'cpu_v1.json').read_bytes(),
        'run.json': b'{"verified": false}',
        # Historical member names remain valid, without restoring root files.
        'requirements-memory.txt': (ROOT / 'requirements/memory-py312plus.txt').read_bytes(),
        'requirements-nvidia.txt': (ROOT / 'requirements/nvidia-cu13.txt').read_bytes(),
    })
    code = '''
import importlib.abc, runpy, sys
from pathlib import Path
class NoNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'stim', 'cupy', 'cuquantum', 'pytest'):
            raise AssertionError('Numerical import during archive verification: ' + fullname)
sys.meta_path.insert(0, NoNumericalImports())
wrapper = runpy.run_path(sys.argv[1], run_name='verification_test')
result = wrapper['verify'](Path(sys.argv[2]))
assert result['integrity_verified'] and not result['science_receipts_passed']
'''
    subprocess.run([sys.executable, '-S', '-c', code, str(ROOT / 'scripts/run_nvidia_pilot.py'), str(path)],
                   check=True, capture_output=True, text=True)


def test_current_cpu_fixture_is_source_bound_and_replays_exact_source():
    from bench.interop import nvidia_coherent_memory as pilot
    import numpy as np
    stored = bundle.strict_json((FIXTURES / 'cpu_v2.json').read_bytes())
    assert bundle.verify_receipt(stored, Path(pilot.__file__).read_bytes(), 'cpu.json')
    replayed = pilot.run_experiment()
    assert stored['config'] == replayed['config']
    assert len(stored['records']) == len(replayed['records'])
    for old, new in zip(stored['records'], replayed['records']):
        assert old['kind'] == new['kind'] and old['rounds'] == new['rounds']
        for part in ('real', 'imag'):
            np.testing.assert_allclose(old['state_vector'][part], new['state_vector'][part], atol=1e-12, rtol=0)


@pytest.mark.parametrize('key,value', [
    ('qubits', 4), ('qubits', True), ('intermediate_recovery', True), ('intermediate_recovery', 0),
    ('rounds', []), ('rounds', [0]), ('rounds', [151]), ('rounds', [True]),
    ('rounds', [1.]), ('rounds', list(range(1, 18))), ('rounds', '1'),
    ('theta', True), ('theta', math.pi + .01), ('precision', 'complex64'), ('extra', 1),
])
def test_resigned_out_of_domain_config_fails_closed(key, value):
    receipt = run_experiment()
    receipt['config'][key] = value
    receipt['config_sha256'] = bundle.sha(bundle.canonical(receipt['config']))
    assert not bundle.verify_receipt(resign(receipt), (ROOT / bundle.SOURCE).read_bytes(), 'cpu.json')


@pytest.mark.parametrize('field,value', [
    ('schema', 'unknown'), ('status', 'unavailable'), ('backend', 'cuquantum'),
    ('gpu_verified', 0), ('tolerance', 1), ('config_sha256', '0' * 64),
])
def test_resigned_invalid_acceptance_fields_fail_closed(field, value):
    receipt = run_experiment()
    receipt[field] = value
    assert not bundle.verify_receipt(resign(receipt), (ROOT / bundle.SOURCE).read_bytes(), 'cpu.json')


@pytest.mark.parametrize('field,value', [
    ('passed', 1), ('rounds', True), ('kind', 'unknown'), ('hardware', {}),
    ('norm_error', True), ('analytic_max_error', -1), ('cpu_phase_aligned_max_error', 1),
    ('state_sha256', '0' * 64), ('circuit_sha256', '0' * 64),
    ('state_vector', {'real': [0.] * 8, 'imag': [0.] * 8}),
    ('state_vector', {'real': [0.] * 7, 'imag': [0.] * 8}),
    ('syndrome_logical_density', {'real': [], 'imag': []}),
    ('syndrome_probabilities', [1., 1., 1., 1.]),
])
def test_resigned_corrupted_serialized_record_fails_closed(field, value):
    receipt = run_experiment()
    receipt['records'][0][field] = value
    assert not bundle.verify_receipt(resign(receipt), (ROOT / bundle.SOURCE).read_bytes(), 'cpu.json')


def test_state_hash_binds_serialized_amplitudes_even_with_resigned_receipt():
    receipt = run_experiment()
    state = receipt['records'][0]['state_vector']
    # A global phase preserves norm and physics, but changes the stored bytes.
    state['real'] = [-value for value in state['real']]
    state['imag'] = [-value for value in state['imag']]
    assert not bundle.verify_receipt(resign(receipt), (ROOT / bundle.SOURCE).read_bytes(), 'cpu.json')


@pytest.mark.parametrize('data', [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e400}', b'{"x":1,"x":2}'])
def test_nonfinite_and_ambiguous_json_rejected(data):
    with pytest.raises(ValueError):
        bundle.strict_json(data)


@pytest.mark.parametrize('version,profile,supported', [
    ((3, 11, 9), 'requirements/memory-py311.txt', True),
    ((3, 12, 1), 'requirements/memory-py312plus.txt', True),
    ((3, 14, 0), 'requirements/memory-py312plus.txt', True),
    ((3, 10, 9), 'requirements/memory-py312plus.txt', False),
    ((3, 15, 0), 'requirements/memory-py312plus.txt', False),
])
def test_dependency_preflight_selects_actionable_profile(monkeypatch, version, profile, supported):
    monkeypatch.setattr(bundle.sys, 'version_info', version)
    monkeypatch.setattr(bundle.importlib.util, 'find_spec', lambda name: None if name == 'numpy' else object())
    result = bundle.dependency_preflight()
    assert result['supported_python'] is supported
    assert result['profile'] == profile
    assert result['missing'] == ['numpy']
    assert not result['passed']
    assert any(f'-r {profile}' in item for item in result['guidance'])


def test_dependency_preflight_and_import_are_stdlib_only():
    code = '''
import runpy, sys
wrapper = runpy.run_path(sys.argv[1], run_name='preflight_test')
wrapper['dependency_preflight']()
assert not any(name.split('.')[0] in ('numpy', 'stim', 'cupy', 'cuquantum', 'pytest') for name in sys.modules)
'''
    subprocess.run([sys.executable, '-S', '-c', code, str(ROOT / 'scripts/run_nvidia_pilot.py')],
                   check=True, capture_output=True, text=True)


@pytest.mark.parametrize('legacy_requirement_paths', [False, True])
def test_run_and_replay_wrapper_with_explicit_test_doubles(tmp_path, monkeypatch, legacy_requirement_paths):
    from bench.interop import nvidia_coherent_memory as pilot
    monkeypatch.setattr(bundle.subprocess, 'check_output', lambda command, **kw: '' if 'status' in command else 'test-commit')
    monkeypatch.setattr(bundle.importlib.util, 'find_spec', lambda name: object())

    def fake_gpu(gates, precision):
        return pilot.cpu_state(gates).astype(precision), {'test_double_not_hardware': True}

    monkeypatch.setattr(pilot, 'gpu_state', fake_gpu)
    commands = []

    def fake_command(command, **kwargs):
        commands.append((command, kwargs))
        if '--output' in command:
            backend = command[command.index('--backend') + 1]
            precision = command[command.index('--precision') + 1] if '--precision' in command else 'complex128'
            receipt = pilot.run_experiment(backend=backend, precision=precision)
            Path(command[command.index('--output') + 1]).write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0, stdout='TEST DOUBLE; no GPU executed\n')

    monkeypatch.setattr(bundle.subprocess, 'run', fake_command)
    first = tmp_path / 'first'
    assert bundle.run(first) == 0
    assert bundle.verify(first / 'evidence.zip')['science_receipts_passed']
    metadata = json.loads((first / 'run.json').read_text())
    assert metadata['commands'][0]['name'] == 'stim-preflight'
    assert [entry['name'] for entry in metadata['commands'][-2:]] == ['cpu-tests', 'gpu-tests']
    gpu_tests = [kwargs for command, kwargs in commands if 'tests/test_nvidia_gpu_acceptance.py' in command]
    assert len(gpu_tests) == 1 and gpu_tests[0]['env']['QEC_REQUIRE_GPU'] == '1'
    cpu_tests = [command for command, _ in commands if 'tests/test_nvidia_coherent_memory.py' in command]
    assert len(cpu_tests) == 1 and 'tests/test_nvidia_gpu_acceptance.py' not in cpu_tests[0]
    with zipfile.ZipFile(first / 'evidence.zip') as archive:
        for member in ('requirements/memory-py311.txt', 'requirements/memory-py312plus.txt',
                       'requirements/nvidia-cu13.txt', 'tests/test_nvidia_gpu_acceptance.py'):
            assert archive.read(member) == (ROOT / member).read_bytes()
    prior = first / 'evidence.zip'
    if legacy_requirement_paths:
        # Only this temporary test bundle uses old names. Retained fixtures and
        # their hashes stay untouched, and neither path layout executes code.
        old_paths = {
            'requirements/memory-py311.txt': 'requirements-memory-py311.txt',
            'requirements/memory-py312plus.txt': 'requirements-memory.txt',
            'requirements/nvidia-cu13.txt': 'requirements-nvidia.txt',
        }
        with zipfile.ZipFile(prior) as archive:
            members = {old_paths.get(name, name): archive.read(name)
                       for name in archive.namelist() if name != 'manifest.json'}
        prior = write_bundle(tmp_path / 'legacy-paths.zip', members)
        assert bundle.verify(prior)['science_receipts_passed']
    assert bundle.replay(prior, tmp_path / 'replayed') == 0
    with pytest.raises(FileExistsError):
        bundle.run(first)

    monkeypatch.setattr(bundle.importlib.util, 'find_spec', lambda name: None if name == 'stim' else object())
    assert bundle.run(tmp_path / 'missing-stim') == 1
    missing = json.loads((tmp_path / 'missing-stim' / 'run.json').read_text())
    assert not missing['verified'] and not missing['dependency_preflight_passed']
    assert 'requirements/memory-' in (tmp_path / 'missing-stim' / 'stim-preflight.log').read_text()
    assert (tmp_path / 'missing-stim' / 'cpu.json').exists()

    def incomplete_run(output):
        output.mkdir()
        (output / 'cpu.json').write_bytes((first / 'cpu.json').read_bytes())
        return 1

    monkeypatch.setattr(bundle, 'run', incomplete_run)
    assert bundle.replay(first / 'evidence.zip', tmp_path / 'incomplete') == 1
    report = json.loads((tmp_path / 'incomplete' / 'replay.json').read_text())
    assert not report['passed']
    assert report['comparisons'][1]['reason'] == 'FileNotFoundError'


def test_missing_numpy_blocks_execution_with_actionable_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle.subprocess, 'check_output', lambda command, **kw: '' if 'status' in command else 'test-commit')
    monkeypatch.setattr(bundle.importlib.util, 'find_spec', lambda name: None)
    monkeypatch.setattr(bundle.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Missing dependencies reached execution'))
    output = tmp_path / 'missing-dependencies'
    assert bundle.run(output) == 1
    assert bundle.verify(output / 'evidence.zip')['science_receipts_passed'] is False
    assert 'numpy' in (output / 'cpu.log').read_text()
    assert 'requirements/memory-py311.txt' in (output / 'cpu.log').read_text()
