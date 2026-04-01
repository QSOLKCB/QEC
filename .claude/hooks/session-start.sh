#!/bin/bash
set -euo pipefail

# Only run in remote (web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PIP="${CLAUDE_PYTHON:-python} -m pip"

# Install project with dev dependencies only if not already available
if ! "${CLAUDE_PYTHON:-python}" -c "import qec" 2>/dev/null; then
  $PIP install -e "${CLAUDE_PROJECT_DIR}[dev]" --quiet
fi

# Install flake8 only if not already available
if ! "${CLAUDE_PYTHON:-python}" -c "import flake8" 2>/dev/null; then
  $PIP install flake8 --quiet
fi
