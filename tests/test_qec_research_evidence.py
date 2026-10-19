# SPDX-License-Identifier: MPL-2.0
"""Cross-module evidence replay against actual unchanged reference BP calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from generate_qec_research_evidence import (
    FIXTURE, ROOT, build_evidence, main, reference_repetition_decode,
)
from qec.sonify.canonical import canonical_bytes, canonical_sha256


def test_checked_in_research_fixture_replays_exactly():
    actual = build_evidence()
    assert canonical_bytes(actual) + b"\n" == FIXTURE.read_bytes()
    unsigned = {key: value for key, value in actual.items() if key != "sha256"}
    assert actual["sha256"] == canonical_sha256(unsigned)
    assert actual["claims"]["patchwork_implemented"] is False
    assert main(["--check"]) == 0


@pytest.mark.parametrize("syndrome", [(0, 0), (1, 0), (1, 1), (0, 1)])
def test_native_reference_correction_matches_supplied_syndrome(syndrome):
    check = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    correction = reference_repetition_decode(syndrome)
    assert tuple((check @ np.array(correction)) % 2) == syndrome


def test_evidence_repeat_is_independent_of_caller_mutation():
    first = build_evidence()
    reference = copy.deepcopy(first)
    first["claims"]["decoder_promoted"] = True
    assert build_evidence() == reference


@pytest.mark.parametrize("hash_seed", ["0", "2718"])
def test_research_evidence_fresh_process_determinism(hash_seed):
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts")))
    result = subprocess.run(
        [sys.executable, "-c", "from generate_qec_research_evidence import build_evidence; "
         "from qec.sonify.canonical import canonical_bytes; "
         "import sys; sys.stdout.buffer.write(canonical_bytes(build_evidence()) + b'\\n')"],
        env=env, cwd=ROOT, capture_output=True, check=True, timeout=30,
    )
    assert result.stdout == FIXTURE.read_bytes()


def test_fixture_check_rejects_tampered_bytes(tmp_path, monkeypatch):
    import generate_qec_research_evidence as generator
    forged = json.loads(FIXTURE.read_text())
    forged["claims"]["decoder_promoted"] = True
    forged["sha256"] = canonical_sha256({key: value for key, value in forged.items() if key != "sha256"})
    target = tmp_path / "forged.json"
    target.write_bytes(canonical_bytes(forged) + b"\n")
    monkeypatch.setattr(generator, "FIXTURE", target)
    with pytest.raises(SystemExit) as error:
        main(["--check"])
    assert error.value.code == 1


def test_synthetic_delay_changes_timing_without_changing_native_outputs():
    evidence = build_evidence()
    immediate, delayed = (item["receipt"] for item in evidence["delivery_cases"])
    expected = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert immediate["baseline_corrections"] == expected
    assert immediate["replay_corrections"] == delayed["replay_corrections"] == expected
    assert immediate["replay_deadline_miss_count"] == 0
    assert delayed["baseline_deadline_miss_count"] == 0
    assert delayed["replay_deadline_miss_count"] == 4
    assert delayed["arrivals"][-1]["frame_index"] == 0
    assert delayed["outputs_match"] is True


def test_steane_presentation_retains_one_encoded_qubit():
    witness = build_evidence()["css_witness"]
    invariants = witness["invariants"]
    assert invariants["original"]["logical_qubits"] == 1
    assert invariants["transformed"]["logical_qubits"] == 1
    assert invariants["sample_count"] == 15
    assert all(sample["covariant"] for sample in witness["samples"])
