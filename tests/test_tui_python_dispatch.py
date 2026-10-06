"""Exercise the real Rust binary with controlled Python adapters, without a TTY."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("QEC_TUI_TEST_BIN"), "requires a built Rust TUI")
class PythonDispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.binary = str(Path(os.environ["QEC_TUI_TEST_BIN"]).resolve())
        self.python = self.root / "configured python"
        self.calls = self.root / "calls.jsonl"
        self.env = dict(os.environ, QEC_TUI_DEMO="0", TEST_CALLS=str(self.calls))
        self.env.pop("QEC_PYTHON", None)
        self.env.pop("VIRTUAL_ENV", None)
        self.python.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
module = sys.argv[2]
with open(os.environ['TEST_CALLS'], 'a') as calls:
    calls.write(json.dumps(sys.argv[1:]) + '\\n')
if module == os.environ.get('TEST_ERROR_MODULE'):
    sys.stderr.write('No module named ' + module)
    sys.exit(3)
if module == os.environ.get('TEST_INVALID_MODULE'):
    print('{}')
    sys.exit(0)
fixtures = {
    'qec.cli.diagnostics': {'collapse_score': 0.4, 'trend_state': 'rising', 'adaptive_damping': 0.5, 'healing_mode': 'test', 'history_behavior': 'test'},
    'qec.cli.history': {'timeline': ['real-stub-observation']},
    'qec.cli.invariants': {'determinism': 'PASS', 'bounds': 'PASS', 'stability': 'PASS', 'law_engine': 'PASS'},
    'qec.cli.phase_diagnostics': {'attractor_state': 'test', 'attractor_cycle_length': 2, 'phase_transition_index': 0.5, 'attractor_entry_cycle': 3, 'transition_sharpness_score': 0.3, 'attractor_confidence_score': 0.4, 'detected_cycle_period': 2, 'cycle_spectrum_class': 'test'},
}
print(json.dumps(fixtures[module]))
''')
        self.python.chmod(0o755)

    def check(self):
        result = subprocess.run([self.binary, "--check-engine"], env=self.env,
                                stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=15)
        self.assertNotIn("\x1b", result.stdout, "engine check entered terminal mode")
        return result

    def call_modules(self):
        return [json.loads(line)[1] for line in self.calls.read_text().splitlines()]

    def test_python3_only_path_handles_all_four_panels(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").symlink_to(self.python)
        self.env["PATH"] = str(bin_dir)
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ENGINE: READY", result.stdout)
        self.assertEqual(self.call_modules(), ["qec.cli.diagnostics", "qec.cli.history",
                                              "qec.cli.invariants", "qec.cli.phase_diagnostics"])

    def test_explicit_python_path_with_spaces(self):
        self.env["PATH"] = ""
        self.env["QEC_PYTHON"] = str(self.python)
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.call_modules()), 4)

    def test_python_fallback_when_python3_is_absent(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to(self.python)
        self.env["PATH"] = str(bin_dir)
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.call_modules()), 4)

    def test_module_failure_does_not_switch_system_interpreters(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").symlink_to(self.python)
        (bin_dir / "python").symlink_to(self.python)
        self.env["PATH"] = str(bin_dir)
        self.env["TEST_ERROR_MODULE"] = "qec.cli.invariants"
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(self.call_modules()), 4, "failed module invoked another interpreter")

    def test_active_virtual_environment_precedes_system_python(self):
        venv_bin = self.root / "venv space" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(self.python)
        self.env["VIRTUAL_ENV"] = str(venv_bin.parent)
        self.env["PATH"] = ""
        self.assertEqual(self.check().returncode, 0)
        self.assertEqual(len(self.call_modules()), 4)

    def test_invalid_explicit_python_does_not_fall_back(self):
        self.env["QEC_PYTHON"] = str(self.root / "missing python")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("ENGINE: UNAVAILABLE", result.stdout)
        self.assertIn("missing python", result.stdout)
        self.assertFalse(self.calls.exists())

    def test_missing_interpreters_explain_configuration(self):
        self.env["PATH"] = ""
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("No Python interpreter found", result.stdout)
        self.assertIn("QEC_PYTHON", result.stdout)

    def test_each_module_failure_is_preserved_without_sample_fallback(self):
        self.env["QEC_PYTHON"] = str(self.python)
        for module in ["qec.cli.diagnostics", "qec.cli.history", "qec.cli.invariants", "qec.cli.phase_diagnostics"]:
            with self.subTest(module=module):
                self.env["TEST_ERROR_MODULE"] = module
                self.calls.unlink(missing_ok=True)
                result = self.check()
                self.assertEqual(result.returncode, 1)
                self.assertIn("ENGINE: UNAVAILABLE", result.stdout)
                self.assertIn("No module named " + module, result.stdout)
                self.assertEqual(len(self.call_modules()), 4, "an extra fallback was invoked")

    def test_invalid_adapter_json_marks_engine_unavailable(self):
        self.env["QEC_PYTHON"] = str(self.python)
        self.env["TEST_INVALID_MODULE"] = "qec.cli.invariants"
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("JSON parse error", result.stdout)

    def test_explicit_demo_works_without_python_and_is_labelled(self):
        self.env["QEC_TUI_DEMO"] = "1"
        self.env["QEC_PYTHON"] = str(self.root / "missing python")
        result = self.check()
        self.assertEqual(result.returncode, 0)
        self.assertIn("ENGINE: DEMO", result.stdout)
        self.assertNotIn("ENGINE: READY", result.stdout)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
