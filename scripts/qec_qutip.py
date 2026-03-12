#!/usr/bin/env python3
"""
Helper utility to install or verify QuTiP for QEC experiment modules.
"""

import subprocess
import sys


# Static, repository-defined install command (no user-controlled input).
INSTALL_CMD = (sys.executable, "-m", "pip", "install", "qutip>=5")


def main() -> None:
    print("Checking QuTiP installation...")

    try:
        import qutip
        print(f"QuTiP already installed: {qutip.__version__}")
        return
    except ImportError:
        print("QuTiP not found.")
        print("Installing QuTiP >=5...")

    try:
        subprocess.check_call(INSTALL_CMD)

        import qutip
        print(f"QuTiP successfully installed: {qutip.__version__}")

    except Exception as exc:
        print("Failed to install QuTiP.")
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
