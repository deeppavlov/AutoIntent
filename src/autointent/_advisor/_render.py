"""Rendering for the pre-flight report.

Text output is grouped by phase (Resource / Data / Config) plus a Drivers
section and the always-on disclaimer. JSON output dumps the structured
report straight through.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._report import PreflightReport

_SEVERITY_TAG = {"green": "✓", "yellow": "⚠", "red": "✗"}
_PHASE_ORDER = ("resource", "data", "config")
_PHASE_LABEL = {"resource": "Resource", "data": "Data", "config": "Config"}


def render_text(report: PreflightReport) -> str:
    lines: list[str] = []
    title = "Compute feasibility check"
    if report.preset_name:
        title += f" — {report.preset_name}"
    lines.append(title)
    lines.append("─" * len(title))

    hw = report.hardware
    lines.append(
        f"Hardware: {hw.get('accelerator', '?')} ({hw.get('device_name', '?')}),"
        f" {hw.get('vram_gb', 0):.1f} GB VRAM, {hw.get('ram_gb', 0):.0f} GB RAM,"
        f" {hw.get('free_disk_gb', 0):.0f} GB free disk"
    )
    ds = report.dataset
    lines.append(
        f"Dataset: n_samples={ds.get('n_samples')}, n_classes={ds.get('n_classes')},"
        f" avg_tokens={ds.get('avg_tokens')} ({ds.get('source')})"
    )
    lines.append("")

    for phase in _PHASE_ORDER:
        bucket = [f for f in report.findings if f.phase == phase]
        if not bucket:
            continue
        lines.append(f"{_PHASE_LABEL[phase]}:")
        for f in bucket:
            tag = _SEVERITY_TAG.get(f.severity.value, "·")
            lines.append(f"  {tag} {f.message}")
        lines.append("")

    if report.resource.drivers:
        lines.append("Drivers of cost:")
        for d in report.resource.drivers[:8]:
            lines.append(
                f"  {d['node_type']}.{d['module']:<10} {d['model']:<48}"
                f"  {d['mode']:<14}  VRAM ~{d['vram_gb']} GB, time ~{d['time_hours']} h"
                f"  [{d['confidence']}]"
            )
        if len(report.resource.drivers) > 8:
            lines.append(f"  … and {len(report.resource.drivers) - 8} more")
        lines.append("")

    if report.notes:
        lines.append("Notes:")
        for note in report.notes:
            lines.append(f"  • {note}")
        lines.append("")

    summary = f"Verdict: {'feasible' if report.is_feasible else 'INFEASIBLE'} "
    summary += f"(worst severity: {report.worst_severity.value})"
    if report.low_confidence:
        summary += " — low-confidence (heuristic fallback in use)"
    lines.append(summary)
    lines.append("Note: estimates are heuristic upper bounds, not measurements.")
    return "\n".join(lines)


def render_json(report: PreflightReport) -> str:
    return json.dumps(report.to_dict(), indent=2, default=str)


def render_recommendation(
    results: list[tuple[str, PreflightReport]],
    chosen: str | None,
) -> str:
    """Compact table for the ``recommend`` subcommand."""
    lines = ["", "Recommendation:"]
    if chosen:
        lines.append(f"  → {chosen}")
    else:
        lines.append("  → none of the bundled presets fit your hardware as-is.")
    lines.append("")
    lines.append(f"{'Preset':<24} {'Status':<14} {'VRAM':<10} {'Time':<10} {'Worst':<8}")
    lines.append("-" * 68)
    for name, report in results:
        verdict = "feasible" if report.is_feasible else "infeasible"
        lines.append(
            f"{name:<24} {verdict:<14} "
            f"{report.resource.vram_gb:>4.1f} GB   "
            f"{report.resource.time_hours:>4.1f} h    "
            f"{report.worst_severity.value:<8}"
        )
    return "\n".join(lines)
