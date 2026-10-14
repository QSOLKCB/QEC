# SPDX-License-Identifier: MPL-2.0
"""Export exact classical acoustic known answers and separate run metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
import platform
from pathlib import Path
import subprocess

from qec.experiments.acoustic.channel import AcousticSource, ChannelConfig, encode_symbols, validate_source, wav_bytes
from qec.sonify.canonical import canonical_json, canonical_sha256
from qec.experiments.acoustic._common import exact

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/acoustic/fixtures"


def generate(output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty")
    # Record the source checkout before generated, unignored output can make
    # it dirty. Metadata describes this input snapshot, not our own writes.
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    output.mkdir(parents=True, exist_ok=True)
    sources=[]
    for d in (3,4):
        source=AcousticSource(ChannelConfig(d),tuple(range(d)),hashlib.sha256(f'qec.acoustic.clean_q{d}.v1'.encode()).hexdigest())
        value=source.as_dict()
        fixture=json.loads((FIXTURES/f'clean_q{d}.json').read_text(encoding='utf-8'))
        expected_source=fixture['source']
        validation=validate_source(value,expected_source_sha256=expected_source['sha256'])
        exact(value,expected_source)
        wav=wav_bytes(encode_symbols(source.measured_symbols,source.config))
        if hashlib.sha256(wav).hexdigest()!=fixture['wav_sha256'] or wav!=(FIXTURES/f'clean_q{d}.wav').read_bytes():
            raise ValueError('INVALID_INPUT')
        (output/f'clean_q{d}.json').write_text(canonical_json(value)+'\n',encoding='utf-8',newline='\n')
        (output/f'clean_q{d}.wav').write_bytes(wav)
        sources.append({'alphabet_size':d,'source_sha256':value['sha256'],'wav_sha256':hashlib.sha256(wav).hexdigest(),
                        'validation_sha256':validation['sha256']})
    identity={'schema':'qec.acoustic-clean-identity.v1','sources':sources,'physical_quantum_protection':False}
    identity['sha256']=canonical_sha256(identity)
    (output/'channel_identity.json').write_text(canonical_json(identity)+'\n',encoding='utf-8',newline='\n')
    environment={'schema':'qec.acoustic-run-metadata.v1','source_commit_sha':commit,
                 'python':platform.python_version(),'platform':platform.platform(),'package_version':version('qec'),
                 'numpy_version':version('numpy'),'scipy_version':version('scipy'),'pytest_version':version('pytest'),
                 'clean_working_tree':not bool(dirty),'canonical_channel_identity_sha256':identity['sha256'],
                 'full_suite_gate':'separate_exact_head_workflow','physical_measurement':False}
    (output/'environment.json').write_text(canonical_json(environment)+'\n',encoding='utf-8',newline='\n')
    return identity


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    print(canonical_json(generate(args.output)))


if __name__=='__main__':
    main()
