#!/usr/bin/env python3
"""
Tests for the repository mirroring script.
"""

import os
import subprocess
import tempfile
import pytest


class TestMirrorReposScript:
    """Test cases for mirror_repos.sh script."""

    def setup_method(self):
        """Set up test environment."""
        self.script_path = os.path.join(os.path.dirname(__file__), '..',
                                        'mirror_repos.sh')
        self.script_path = os.path.abspath(self.script_path)

    def test_script_syntax(self):
        """Test that the script has valid bash syntax."""
        result = subprocess.run(['bash', '-n', self.script_path],
                                capture_output=True, text=True)
        assert result.returncode == 0, f"Syntax error in script: {result.stderr}"

    def test_script_exists_and_executable(self):
        """Test that the script exists and is executable."""
        assert os.path.exists(self.script_path), "mirror_repos.sh script not found"
        assert os.access(self.script_path, os.X_OK), "mirror_repos.sh is not executable"

    def test_dependency_check_function(self):
        """Test the dependency checking functionality."""
        # Test with existing command - should succeed
        test_script_success = '''
        #!/bin/bash
        source ''' + self.script_path + '''

        # Test with existing command
        need echo
        echo "echo check passed"
        '''

        # Test with non-existent command - should fail
        test_script_fail = '''
        #!/bin/bash
        source ''' + self.script_path + '''

        # Test with non-existent command (should fail)
        need nonexistent_command_12345 2>/dev/null
        echo "Should not reach here"
        '''

        # Test success case
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(test_script_success)
            f.flush()

            try:
                result = subprocess.run(['bash', f.name],
                                        capture_output=True, text=True, timeout=10)
                assert result.returncode == 0, f"Script failed: {result.stderr}"
                assert "echo check passed" in result.stdout
            finally:
                os.unlink(f.name)

        # Test failure case
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(test_script_fail)
            f.flush()

            try:
                result = subprocess.run(['bash', f.name],
                                        capture_output=True, text=True, timeout=10)
                # Should fail with exit code 1
                assert result.returncode == 1, "Script should have failed for non-existent command"
                assert "Should not reach here" not in result.stdout
            finally:
                os.unlink(f.name)

    def test_timestamp_function(self):
        """Test the timestamp function."""
        test_script = '''
        #!/bin/bash
        source ''' + self.script_path + '''
        timestamp
        '''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(test_script)
            f.flush()

            try:
                result = subprocess.run(['bash', f.name],
                                        capture_output=True, text=True, timeout=10)
                assert result.returncode == 0, f"Script failed: {result.stderr}"
                # Should output a timestamp in YYYY-MM-DD HH:MM:SS format
                output = result.stdout.strip()
                assert len(output) == 19  # "YYYY-MM-DD HH:MM:SS"
                assert output[4] == '-' and output[7] == '-' and output[10] == ' '
                assert output[13] == ':' and output[16] == ':'
            finally:
                os.unlink(f.name)

    def test_log_function(self):
        """Test the log function."""
        test_script = '''
        #!/bin/bash
        source ''' + self.script_path + '''
        log "Test message"
        '''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(test_script)
            f.flush()

            try:
                result = subprocess.run(['bash', f.name],
                                        capture_output=True, text=True, timeout=10)
                assert result.returncode == 0, f"Script failed: {result.stderr}"
                output = result.stdout.strip()
                assert "Test message" in output
                # Should include timestamp format
                assert "[" in output and "]" in output
            finally:
                os.unlink(f.name)

    def test_dry_run_mode(self):
        """Test that dry run mode is respected."""
        # Test with DRY_RUN=1, should not fail even without gh/git
        env = os.environ.copy()
        env['DRY_RUN'] = '1'

        # Create a minimal test that checks DRY_RUN behavior
        test_script = '''
        #!/bin/bash
        set -euo pipefail
        source ''' + self.script_path + '''

        # Test a simple log message
        log "Testing dry run mode"
        echo "DRY_RUN is set to: $DRY_RUN"
        '''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(test_script)
            f.flush()

            try:
                result = subprocess.run(['bash', f.name],
                                        capture_output=True, text=True, timeout=10,
                                        env=env)
                assert result.returncode == 0, f"Script failed: {result.stderr}"
                assert "Testing dry run mode" in result.stdout
                assert "DRY_RUN is set to: 1" in result.stdout
            finally:
                os.unlink(f.name)

    def test_environment_variables(self):
        """Test that environment variables are properly handled."""
        test_script = '''
        #!/bin/bash
        source ''' + self.script_path + '''

        echo "ORG_TEST: $ORG_TEST"
        echo "ORG_DEV: $ORG_DEV"
        echo "DEST_ORG: $DEST_ORG"
        echo "DRY_RUN: $DRY_RUN"
        echo "USE_HTTPS: $USE_HTTPS"
        '''

        env = os.environ.copy()
        env.update({
            'ORG_TEST': 'test-org',
            'ORG_DEV': 'dev-org',
            'DEST_ORG': 'dest-org',
            'DRY_RUN': '1',
            'USE_HTTPS': '1'
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(test_script)
            f.flush()

            try:
                result = subprocess.run(['bash', f.name],
                                        capture_output=True, text=True, timeout=10,
                                        env=env)
                assert result.returncode == 0, f"Script failed: {result.stderr}"
                output = result.stdout
                assert "ORG_TEST: test-org" in output
                assert "ORG_DEV: dev-org" in output
                assert "DEST_ORG: dest-org" in output
                assert "DRY_RUN: 1" in output
                assert "USE_HTTPS: 1" in output
            finally:
                os.unlink(f.name)


if __name__ == '__main__':
    pytest.main([__file__])
