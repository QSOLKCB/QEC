#!/bin/bash
set -euo pipefail

# Only run in remote (web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install project with dev dependencies (editable mode)
pip install -e "${CLAUDE_PROJECT_DIR}[dev]" --quiet

# Install flake8 for linting (listed in requirements.txt)
pip install flake8 --quiet
