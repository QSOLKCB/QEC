"""
FER plotting utility.  Requires matplotlib (not a core dependency).

This module is intentionally separated from the core package so that
``matplotlib`` is never imported by the decoder or simulation modules.
"""

from __future__ import annotations

import json


def plot_fer(data, title="Frame Error Rate", outfile=None):
    """Plot FER (and optionally BER) vs physical error probability.

    Args:
        data: Dict from :func:`simulate_fer` output, or a path to a
            JSON file containing such a dict.
        title: Plot title string.
        outfile: If given, save figure to this path instead of calling
            ``plt.show()``.

    Returns:
        (fig, ax) matplotlib objects.
    """
    import matplotlib.pyplot as plt

    if isinstance(data, str):
        with open(data, "r") as f:
            data = json.load(f)

    results = data["results"]
    p_grid = results["p"]
    fer = results["FER"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(p_grid, fer, "o-", label="FER")

    if "BER" in results:
        ber = results["BER"]
        ax.semilogy(p_grid, ber, "s--", label="BER")

    ax.set_xlabel("Physical error probability p")
    ax.set_ylabel("Error rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig, ax
