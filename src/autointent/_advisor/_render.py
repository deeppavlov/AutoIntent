"""Rendering for the pre-flight report.

Text output is grouped by phase (Resource / Data / Config) plus a Drivers
section and the always-on disclaimer. JSON output dumps the structured
report straight through.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._report import PreflightReport

_SEVERITY_TAG = {"ample": "✓", "tight": "⚠", "over": "✗"}
_PHASE_ORDER = ("resource", "data", "config")
_PHASE_LABEL = {"resource": "Resource", "data": "Data", "config": "Config"}


def _batch_hint(driver: dict[str, Any]) -> str:
    """Per-driver batch annotation: '64 → 32', '64', '64 (no fit)', or ''."""
    bs = driver.get("batch_size")
    if bs is None:
        return ""
    mx = driver.get("max_batch_size")
    if mx is None:
        return str(bs)
    if mx == 0:
        return f"{bs} (no fit)"
    if mx == bs:
        return str(bs)
    return f"{bs} → {mx}"


_DRIVERS_LIMIT = 8
_DRIVERS_HEADERS = ("Node", "Model", "Mode", "VRAM", "Time", "Batch", "Source")


def _render_drivers_table(drivers: list[dict[str, Any]]) -> list[str]:
    """Format the Drivers of cost section as an aligned table."""
    visible = drivers[:_DRIVERS_LIMIT]
    rows: list[tuple[str, ...]] = [
        (
            f"{d['node_type']}.{d['module']}",
            str(d["model"]),
            str(d["mode"]),
            f"{d['vram_gb']:.2f} GB",
            f"{d['time_hours']:.2f} h",
            _batch_hint(d),
            f"[{d['confidence']}]",
        )
        for d in visible
    ]

    widths = [len(h) for h in _DRIVERS_HEADERS]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Right-align numeric columns (VRAM @ idx 3, Time @ idx 4); left-align the rest.
    right_align = {3, 4}

    def fmt(row: tuple[str, ...]) -> str:
        cells = []
        for i, cell in enumerate(row):
            if i in right_align:
                cells.append(cell.rjust(widths[i]))
            else:
                cells.append(cell.ljust(widths[i]))
        return "  " + "  ".join(cells).rstrip()

    lines = ["Drivers of cost:", fmt(_DRIVERS_HEADERS), "  " + "  ".join("─" * w for w in widths)]
    lines.extend(fmt(r) for r in rows)
    if len(drivers) > _DRIVERS_LIMIT:
        lines.append(f"  … and {len(drivers) - _DRIVERS_LIMIT} more")
    return lines


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
        lines.extend(_render_drivers_table(report.resource.drivers))
        lines.append("")

    if report.notes:
        lines.append("Notes:")
        lines.extend(f"  • {note}" for note in report.notes)
        lines.append("")

    summary = f"Verdict: {'feasible' if report.is_feasible else 'INFEASIBLE'} "
    summary += f"(headroom: {report.headroom.value})"
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
    lines.append(f"{'Preset':<24} {'Status':<14} {'VRAM':<10} {'Time':<10} {'Headroom':<10}")
    lines.append("-" * 68)
    for name, report in results:
        verdict = "feasible" if report.is_feasible else "infeasible"
        lines.append(
            f"{name:<24} {verdict:<14} "
            f"{report.resource.vram_gb:>4.1f} GB   "
            f"{report.resource.time_hours:>4.1f} h    "
            f"{report.headroom.value:<8}"
        )
    return "\n".join(lines)
