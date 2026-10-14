# SPDX-License-Identifier: MPL-2.0
"""Generate replayed candidate evidence; never creates a release or hardware claim."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from qec.sonify.canonical import canonical_json, canonical_sha256


def run(module, *args):
    completed = subprocess.run([sys.executable, '-m', module, *map(str, args)],
                               check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def generate(output: Path, eras: tuple[str, ...], *, require_clean=False):
    if not eras or len(set(eras)) != len(eras) or any(x not in ('ess','digital','packet','cross-era') for x in eras):
        raise ValueError('invalid eras')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty to avoid stale artifact provenance')
    output.mkdir(parents=True, exist_ok=True)
    validations = []
    for era in eras:
        destination = output / era
        module = {'ess':'qec.routing.ess.lifecycle_cli', 'digital':'qec.routing.digital',
                  'packet':'qec.routing.packet', 'cross-era':'qec.routing.cross_era_cli'}[era]
        input_name = {'ess':'ess_lifecycle_input.json','digital':'digital_switch_input.json',
                      'packet':'packet_fabric_input.json','cross-era':'switch_compatibility_corpus_manifest.json'}[era]
        run(module, 'demo', '--output-dir', destination)
        validation = run(module, 'run', '--input', destination/input_name, '--output-dir', destination)
        if validation.get('all_passed') is not True:
            raise ValueError('candidate validation did not pass: '+era)
        validations.append({'era':era, 'validation':validation})
        if era == 'ess':
            value = run(module, 'equivalence-demo', '--output-dir', destination)
            if value.get('all_passed') is not True:
                raise ValueError('ESS equivalence failed')
            validations.append({'era':'ess-fabric-equivalence','validation':value})
        elif era == 'digital':
            value = run(module, 'fault-battery', '--output-dir', destination)
            if value.get('all_passed') is not True:
                raise ValueError('Digital battery failed')
            validations.append({'era':'digital-fault-battery','validation':value})
    artifacts = []
    for path in sorted(output.rglob('*')):
        if path.is_file():
            data = path.read_bytes()
            artifacts.append({'path':path.relative_to(output).as_posix(),'bytes':len(data),
                              'file_sha256':hashlib.sha256(data).hexdigest()})
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain'],text=True)
    if require_clean and dirty:
        raise ValueError('source working tree is dirty after generation')
    environment = {'python':platform.python_version(),'implementation':platform.python_implementation(),
                   'platform':platform.platform(),'package_version':version('qec'),
                   'pytest_version':version('pytest'), 'numpy_version':version('numpy'), 'scipy_version':version('scipy')}
    receipt = {'schema':'qec.switching-candidate-evidence-manifest.v1','source_commit_sha':commit,
               'source_package_version':version('qec'),'implementation_status':'unreleased_candidate',
               'eras':list(eras),'environment':environment,'clean_working_tree':not bool(dirty),
               'native_replay_validations':validations,'artifacts':artifacts,
               'full_repository_test_gate':'separate_exact_head_Build_and_Test_workflow_required',
               'claim_validation':'only_declared_bounded_software_receipts;see_native_machine_claim_boundaries',
               'limitations':['no_release_or_tag_created','no_hardware_performance_evidence',
                              'no_live_network_authority','environment_manifest_is_run_metadata_not_model_determinism',
                              'browser_visual_and_audio_behavior_requires_separate_QA']}
    receipt['sha256']=canonical_sha256(receipt)
    (output/'candidate_evidence_manifest.json').write_text(canonical_json(receipt)+'\n')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--eras',required=True,help='comma-separated explicit era names')
    parser.add_argument('--require-clean',action='store_true',help='fail if source checkout changes during validation')
    args=parser.parse_args(argv)
    try:
        receipt=generate(args.output,tuple(args.eras.split(',')),require_clean=args.require_clean)
        print(canonical_json({'manifest_sha256':receipt['sha256'],'source_commit_sha':receipt['source_commit_sha'],
                              'artifact_count':len(receipt['artifacts']),'clean_working_tree':receipt['clean_working_tree']}))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
