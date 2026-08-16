"""Render the issue-#39 definition-of-done tables straight from the run JSONs.

Table 1 (feasibility) pairs each preset's advisor verdict with what really
happened. Table 2 (accuracy) reports actual/predicted ratios for the presets
that fit. Reading both out of the JSON rather than transcribing by hand keeps
the write-up honest.

Usage:
    uv run --no-sync python scripts/render_issue39_tables.py \
        --preflight calibration_runs/banking77_<ts>.json \
        --fits calibration_runs/phase2_isolated/*/banking77_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(paths: list[str]) -> dict[str, dict[str, Any]]:
    """Merge rows from several calibration JSONs, keyed by preset name."""
    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            out[row["preset"]] = row
    return out


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _ratio(actual: float | None, predicted: float | None) -> str:
    if actual is None or not predicted:
        return "—"
    return f"{actual / predicted:.2f}x"


def _outcome(row: dict[str, Any]) -> str:
    """What actually happened during the fit: OOM / fit / not-run."""
    error = row.get("error")
    if error is None:
        return "fit" if row.get("actual") else "not run"
    if "out of memory" in error.lower():
        return "OOM"
    if row.get("skipped"):
        return "skipped"
    return "error"


def main() -> None:
    parser = argparse.ArgumentParser("render_issue39_tables")
    parser.add_argument("--preflight", required=True, help="Phase 1 (SKIP_FIT) JSON")
    parser.add_argument("--fits", nargs="*", default=[], help="Phase 2 JSONs (one per preset is fine)")
    parser.add_argument("--counterfactual", help="phase1b_counterfactual.json (optional)")
    args = parser.parse_args()

    pre = _load_rows([args.preflight])
    fits = _load_rows(args.fits)
    counter = {}
    if args.counterfactual:
        payload = json.loads(Path(args.counterfactual).read_text(encoding="utf-8"))
        counter = {r["preset"]: r for r in payload["rows"]}

    print("### Table 1 — feasibility\n")
    print("| preset | model | pred VRAM (GB) | advisor verdict | actual | match? |")
    print("| --- | --- | --- | --- | --- | --- |")
    for preset, row in pre.items():
        if row.get("skipped"):
            continue
        vram_sev = row.get("severity_by_metric", {}).get("vram", "—")
        pred_vram = row["predicted"].get("vram_gb")
        fit_row = fits.get(preset)
        actual = _outcome(fit_row) if fit_row else "not run"
        if actual in {"not run", "skipped", "error"}:
            match = "—"
        else:
            predicted_over = vram_sev == "over"
            match = "✅" if predicted_over == (actual == "OOM") else "❌"
        model = ", ".join(sorted(set(row.get("models", {}).values()))) or "—"
        print(
            f"| `{preset}` | {model} | {_fmt(pred_vram)} | **{vram_sev}** | {actual} | {match} |"
        )

    print("\n### Table 2 — accuracy on presets that fit\n")
    print("| preset | actual/pred VRAM | actual/pred RAM | actual/pred time | actual VRAM (GB) | pred VRAM (GB) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for preset, row in fits.items():
        if _outcome(row) != "fit":
            continue
        actual, predicted = row["actual"], row["predicted"]
        print(
            f"| `{preset}` | {_ratio(actual.get('vram_gb'), predicted.get('vram_gb'))} "
            f"| {_ratio(actual.get('ram_gb'), predicted.get('ram_gb'))} "
            f"| {_ratio(actual.get('time_h'), predicted.get('time_h'))} "
            f"| {_fmt(actual.get('vram_gb'))} | {_fmt(predicted.get('vram_gb'))} |"
        )

    if counter:
        print("\n### Metadata counterfactual (deberta has no safetensors -> 350M fallback)\n")
        print("| preset | params assumed | params true | VRAM as-run | VRAM corrected | verdict changes? |")
        print("| --- | --- | --- | --- | --- | --- |")
        for preset, row in counter.items():
            changed = "no" if row["as_run"]["headroom"] == row["corrected"]["headroom"] else "YES"
            print(
                f"| `{preset}` | {row['params_assumed_M']} M | {row['params_true_M']} M "
                f"| {row['as_run']['vram_gb']} GB ({row['as_run']['headroom']}) "
                f"| {row['corrected']['vram_gb']} GB ({row['corrected']['headroom']}) | {changed} |"
            )


if __name__ == "__main__":
    main()
