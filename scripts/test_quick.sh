#!/usr/bin/env bash
# scripts/test_quick.sh — deterministic quick-test bootstrap
# v133.9.1
#
# Installs dev requirements if missing, then runs targeted simulation tests.
# Exit non-zero on any failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV_REQ="${REPO_ROOT}/dev-requirements.txt"
DEFAULT_TARGET="tests/test_phase_timeline_engine.py"

# Install dev requirements if pytest is not available
if ! python -c "import pytest" 2>/dev/null; then
    echo ":: Installing dev requirements ..."
    python -m pip install -r "${DEV_REQ}" --quiet
fi

# Ensure the package itself is importable
if ! python -c "import qec" 2>/dev/null; then
    echo ":: Installing qec in dev mode ..."
    python -m pip install -e "${REPO_ROOT}" --quiet
fi

TARGET="${1:-${DEFAULT_TARGET}}"

echo ":: Running tests: ${TARGET}"
python -m pytest "${TARGET}" -v --tb=short --disable-warnings
