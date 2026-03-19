"""
Interop report generator for v3.1.2 baseline suite results.

Generates a Markdown report that strictly separates:
  (A) Direct comparisons
  (B) Reference baselines (context-only)

Appendices include exact configs, tool versions, and schema record excerpts.
"""

from __future__ import annotations

import json
from typing import Any

from .serialize import canonical_json


def _record_sort_key(rec: dict[str, Any]) -> tuple:
    """Deterministic sort key for interop records."""
    cfg = rec.get("config", {})
    return (
        rec.get("tool", {}).get("name", ""),
        cfg.get("distance", 0),
        cfg.get("p", 0.0),
    )


def generate_report(suite_result: dict[str, Any]) -> str:
    """Generate a Markdown report from a baseline suite result.

    Parameters
    ----------
    suite_result:
        The dict returned by :func:`baselines.run_baseline_suite`.

    Returns
    -------
    Markdown string.
    """
    lines: list[str] = []

    lines.append("# QEC v3.1.2 Industry-Grade Interop Benchmark Report")
    lines.append("")
    lines.append(f"**Created:** {suite_result.get('created_utc', 'N/A')}")
    lines.append(f"**Schema Version:** {suite_result.get('schema_version', 'N/A')}")
    lines.append("")

    env = suite_result.get("environment", {})
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- **Platform:** {env.get('platform', 'N/A')}")
    lines.append(f"- **Python:** {env.get('python_version', 'N/A')}")
    lines.append(f"- **NumPy:** {env.get('numpy_version', 'N/A')}")
    lines.append(f"- **QEC:** {env.get('qec_version', 'N/A')}")
    lines.append(f"- **Git Commit:** {env.get('git_commit', 'N/A')}")
    tool_vers = env.get("tool_versions", {})
    for tool_name in sorted(tool_vers.keys()):
        ver = tool_vers[tool_name]
        status = ver if ver else "not installed"
        lines.append(f"- **{tool_name}:** {status}")
    lines.append("")

    # ── Section A: Direct Comparisons ──
    direct = suite_result.get("direct_comparisons", [])
    lines.append("## A) Direct Comparisons (QEC-Native)")
    lines.append("")
    lines.append("These results compare QEC decoder variants on the same code")
    lines.append("family and representation. Results are directly comparable.")
    lines.append("")

    active_direct = sorted(
        [r for r in direct if r.get("status") != "skipped"],
        key=_record_sort_key,
    )
    if active_direct:
        lines.append(
            "| Tool | Code Family | Distance | p | "
            "Logical Error Rate | Mean Iters |"
        )
        lines.append(
            "|------|------------|----------|---|"
            "-------------------|------------|"
        )
        for rec in active_direct:
            tool_name = rec["tool"]["name"]
            cf = rec["code_family"]
            cfg = rec.get("config", {})
            dist = cfg.get("distance", "N/A")
            p = cfg.get("p", "N/A")
            res = rec.get("results", {})
            ler = res.get("logical_error_rate", "N/A")
            mi = res.get("mean_iters", "N/A")
            if isinstance(ler, float):
                ler = f"{ler:.8f}"
            if isinstance(mi, float):
                mi = f"{mi:.4f}"
            lines.append(f"| {tool_name} | {cf} | {dist} | {p} | {ler} | {mi} |")
        lines.append("")
    else:
        lines.append("*No direct comparison results available.*")
        lines.append("")

    # ── Section B: Reference Baselines ──
    ref = suite_result.get("reference_baselines", [])
    lines.append("## B) Reference Baselines (Context-Only)")
    lines.append("")
    lines.append("These results use different tools and/or representations.")
    lines.append("They provide context but are **NOT** directly comparable")
    lines.append("to QEC-native results.")
    lines.append("")

    active_ref = sorted(
        [r for r in ref if r.get("status") != "skipped"],
        key=_record_sort_key,
    )
    skipped_ref = [r for r in ref if r.get("status") == "skipped"]

    if active_ref:
        lines.append(
            "| Tool | Code Family | Distance | p | "
            "Logical Error Rate | Representation |"
        )
        lines.append(
            "|------|------------|----------|---|"
            "-------------------|----------------|"
        )
        for rec in active_ref:
            tool_name = rec["tool"]["name"]
            cf = rec["code_family"]
            cfg = rec.get("config", {})
            dist = cfg.get("distance", "N/A")
            p = cfg.get("p", "N/A")
            res = rec.get("results", {})
            ler = res.get("logical_error_rate", "N/A")
            if isinstance(ler, float):
                ler = f"{ler:.8f}"
            rep = rec.get("representation", "N/A")
            lines.append(
                f"| {tool_name} | {cf} | {dist} | {p} | {ler} | {rep} |"
            )
        lines.append("")

    if skipped_ref:
        lines.append("### Skipped Reference Baselines")
        lines.append("")
        for rec in skipped_ref:
            tool_name = rec.get("tool", {}).get("name", "unknown")
            reason = rec.get("reason", "unknown reason")
            lines.append(f"- **{tool_name}**: {reason}")
        lines.append("")

    if not active_ref and not skipped_ref:
        lines.append("*No reference baselines configured.*")
        lines.append("")

    # ── Appendix A: Suite Configuration ──
    lines.append("## Appendix A: Suite Configuration")
    lines.append("")
    lines.append("```json")
    suite_cfg = suite_result.get("suite_config", {})
    lines.append(json.dumps(suite_cfg, sort_keys=True, indent=2))
    lines.append("```")
    lines.append("")

    # ── Appendix B: Tool Versions ──
    lines.append("## Appendix B: Tool Versions")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(tool_vers, sort_keys=True, indent=2))
    lines.append("```")
    lines.append("")

    # ── Appendix C: Schema Record Excerpts ──
    lines.append("## Appendix C: Schema Record Excerpts")
    lines.append("")

    all_records = suite_result.get("all_records", [])
    for i, rec in enumerate(all_records):
        status = rec.get("status")
        if status == "skipped":
            lines.append(f"### Record {i + 1} (skipped)")
            lines.append("")
            lines.append(f"- **Reason:** {rec.get('reason', 'N/A')}")
            lines.append("")
            continue

        bk = rec.get("benchmark_kind", "N/A")
        tool_name = rec.get("tool", {}).get("name", "N/A")
        lines.append(f"### Record {i + 1} ({bk} / {tool_name})")
        lines.append("")
        lines.append("```json")
        lines.append(canonical_json(rec))
        lines.append("```")
        lines.append("")

    # ── Appendix D: Usage ──
    lines.append("## Appendix D: Entrypoint Usage")
    lines.append("")
    lines.append("```python")
    lines.append("from bench.interop.baselines import run_baseline_suite")
    lines.append("from bench.interop.report import generate_report")
    lines.append("")
    lines.append("suite = run_baseline_suite(")
    lines.append("    native_distances=[3, 5],")
    lines.append("    native_p_values=[0.001, 0.003],")
    lines.append("    seed=12345,")
    lines.append("    deterministic_metadata=True,")
    lines.append(")")
    lines.append("report_md = generate_report(suite)")
    lines.append("print(report_md)")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)
