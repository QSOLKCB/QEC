"""Real GPU tests: optional on CPU CI, mandatory in the local evidence wrapper."""
import math
import os

import pytest

from bench.interop import nvidia_coherent_memory as pilot


@pytest.mark.parametrize('precision', ['complex64', 'complex128'])
@pytest.mark.parametrize('theta', [0., .125, -.31, math.pi / 2, math.pi])
def test_real_gpu_coherent_controls(precision, theta):
    receipt = pilot.run_experiment(backend='cuquantum', precision=precision,
                                   rounds=(1, 3, 150), theta=theta)
    if receipt['status'] == 'unavailable' and os.environ.get('QEC_REQUIRE_GPU') != '1':
        pytest.skip(receipt['reason'])
    assert receipt['status'] == 'completed', receipt.get('reason')
    assert receipt['gpu_verified'] is True
    assert len(receipt['records']) == 9
    assert all(record['passed'] and record['hardware'] is not None
               for record in receipt['records'])
