"""Text-based alignment visualization for DFA correction benchmarks (v92.3.0).

Displays per-step syndrome alignment before and after correction
in a readable text table format.

All output is deterministic.  No external dependencies beyond stdlib.
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# PART 1 — FULL ALIGNMENT TABLE
# ---------------------------------------------------------------------------


def print_alignment(
    alignment: List[Dict[str, Any]], max_steps: int = 10
) -> str:
    """Format alignment data as a text table.

    Each row shows: step, DFA state, syndrome before, syndrome after.

    Args:
        alignment: list of dicts with "step", "state", "before", "after".
        max_steps: maximum number of rows to display.

    Returns:
        Formatted table string.
    """
    lines: List[str] = []
    lines.append(
        f"{'step':<6} {'state':<8} {'before':<16} {'after':<16}"
    )
    lines.append("-" * 48)
    for row in alignment[:max_steps]:
        before_str = str(row["before"])
        after_str = str(row["after"])
        lines.append(
            f"{row['step']:<6} {row['state']:<8} "
            f"{before_str:<16} {after_str:<16}"
        )
    if len(alignment) > max_steps:
        lines.append(f"... ({len(alignment) - max_steps} more rows)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PART 2 — CHANGES-ONLY VIEW
# ---------------------------------------------------------------------------


def print_changes_only(alignment: List[Dict[str, Any]]) -> str:
    """Format only rows where syndrome changed after correction.

    Args:
        alignment: list of dicts with "step", "state", "before", "after".

    Returns:
        Formatted table string with only changed rows.
    """
    lines: List[str] = []
    lines.append(
        f"{'step':<6} {'state':<8} {'before':<16} {'after':<16}"
    )
    lines.append("-" * 48)
    found = False
    for row in alignment:
        if row["before"] != row["after"]:
            before_str = str(row["before"])
            after_str = str(row["after"])
            lines.append(
                f"{row['step']:<6} {row['state']:<8} "
                f"{before_str:<16} {after_str:<16}"
            )
            found = True
    if not found:
        lines.append("(no changes)")
    return "\n".join(lines)
