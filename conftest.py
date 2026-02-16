"""
Root conftest.py — ensures src/qldpc is importable without pulling in
the parent src package's heavy dependencies (qutip, etc.).
"""

import sys
import os

# Add the repo root to sys.path so `src.qldpc` resolves correctly.
sys.path.insert(0, os.path.dirname(__file__))
