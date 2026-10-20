"""The bundle attests byte integrity, not trusted execution."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from bench.interop.nvidia_coherent_memory import run_experiment

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_bundle', ROOT/'scripts/run_nvidia_pilot.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def fixture_bundle(tmp_path):
    members = {bundle.SOURCE: (ROOT/bundle.SOURCE).read_bytes(), 'run.json': b'{"verified": false}'}
    receipt = run_experiment()
    # These are explicitly CPU records, not fake GPU success.
    for name in ('cpu.json', 'gpu-complex64.json', 'gpu-complex128.json'):
        members[name] = json.dumps(receipt).encode()
    members['manifest.json'] = json.dumps({name: bundle.sha(data) for name,data in members.items()}).encode()
    path = tmp_path/'evidence.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in members.items():
            z.writestr(name,data)
    path.with_suffix('.zip.sha256').write_text(bundle.sha(path.read_bytes())+'  evidence.zip\n')
    return path


def test_integrity_not_gpu_authenticity(tmp_path):
    result = bundle.verify(fixture_bundle(tmp_path))
    assert result['integrity_verified']
    assert not result['science_receipts_passed']


def test_archive_tamper(tmp_path):
    path = fixture_bundle(tmp_path)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Archive SHA'):
        bundle.verify(path)


def test_member_tamper_even_with_updated_zip_hash(tmp_path):
    path = fixture_bundle(tmp_path)
    with zipfile.ZipFile(path) as z:
        members = {name:z.read(name) for name in z.namelist()}
    members['run.json'] = b'{"verified": true}'
    with zipfile.ZipFile(path,'w') as z:
        for name, data in members.items():
            z.writestr(name,data)
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
    members = {'run.json': b'{"verified": false}'}
    members['manifest.json'] = json.dumps({name: bundle.sha(data) for name,data in members.items()}).encode()
    path = tmp_path/'evidence.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for name,data in members.items():
            z.writestr(name,data)
    path.with_suffix('.zip.sha256').write_text(bundle.sha(path.read_bytes()))
    result = bundle.verify(path)
    assert result['integrity_verified'] and not result['science_receipts_passed']


def test_replay_rejects_source_mismatch_before_execution(tmp_path, monkeypatch):
    path = fixture_bundle(tmp_path)
    fake_root = tmp_path/'other'
    (fake_root/Path(bundle.SOURCE).parent).mkdir(parents=True)
    (fake_root/bundle.SOURCE).write_text('different source')
    monkeypatch.setattr(bundle, 'ROOT', fake_root)
    with pytest.raises(ValueError, match='exact recorded source'):
        bundle.replay(path, tmp_path/'replay')


def test_checked_cpu_fixture_is_source_bound_and_replays():
    from bench.interop import nvidia_coherent_memory as pilot
    stored = json.loads((ROOT/'tests/fixtures/nvidia_coherent_memory/cpu_v1.json').read_text())
    receipt_hash = stored.pop('receipt_sha256')
    assert pilot.digest(stored) == receipt_hash
    assert stored['source_sha256'][Path(pilot.__file__).name] == pilot.digest(Path(pilot.__file__).read_bytes())
    replayed = pilot.run_experiment()
    assert stored['config'] == replayed['config']
    import numpy as np
    for old, new in zip(stored['records'], replayed['records']):
        assert old['kind'] == new['kind'] and old['rounds'] == new['rounds']
        for part in ('real','imag'):
            np.testing.assert_allclose(old['state_vector'][part],new['state_vector'][part],atol=1e-12,rtol=0)


def test_run_and_replay_wrapper_with_explicit_test_doubles(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from bench.interop import nvidia_coherent_memory as pilot
    monkeypatch.setattr(bundle.subprocess, 'check_output', lambda command, **kw: '' if 'status' in command else 'test-commit')
    def fake_gpu(gates, precision):
        return pilot.cpu_state(gates).astype(precision), {'test_double_not_hardware': True}
    monkeypatch.setattr(pilot, 'gpu_state', fake_gpu)
    def fake_command(command, **kw):
        if '--output' in command:
            backend = command[command.index('--backend')+1]
            precision = command[command.index('--precision')+1] if '--precision' in command else 'complex128'
            receipt = pilot.run_experiment(backend=backend, precision=precision)
            Path(command[command.index('--output')+1]).write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0, stdout='TEST DOUBLE; no GPU executed\n')
    monkeypatch.setattr(bundle.subprocess, 'run', fake_command)
    first = tmp_path/'first'
    assert bundle.run(first) == 0
    assert bundle.verify(first/'evidence.zip')['science_receipts_passed']
    metadata = json.loads((first/'run.json').read_text())
    assert metadata['commands'][0]['name'] == 'stim-preflight'
    assert bundle.replay(first/'evidence.zip', tmp_path/'replayed') == 0
    with pytest.raises(FileExistsError):
        bundle.run(first)
    def no_stim(command, **kw):
        if command[1] == '-c':
            return SimpleNamespace(returncode=1, stdout='TEST DOUBLE: Stim missing')
        return fake_command(command, **kw)
    monkeypatch.setattr(bundle.subprocess, 'run', no_stim)
    assert bundle.run(tmp_path/'missing-stim') == 1
    assert not json.loads((tmp_path/'missing-stim'/'run.json').read_text())['verified']
    def incomplete_run(output):
        output.mkdir()
        (output/'cpu.json').write_bytes((first/'cpu.json').read_bytes())
        return 1
    monkeypatch.setattr(bundle, 'run', incomplete_run)
    assert bundle.replay(first/'evidence.zip', tmp_path/'incomplete') == 1
    report = json.loads((tmp_path/'incomplete'/'replay.json').read_text())
    assert not report['passed']
    assert report['comparisons'][1]['reason'] == 'FileNotFoundError'
