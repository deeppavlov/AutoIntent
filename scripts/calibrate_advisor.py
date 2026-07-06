"""Calibrate advisor preflight estimates against real Pipeline.fit measurements.

Runs each requested preset twice: first through ``run_preflight`` to capture the
heuristic estimate, then through ``Pipeline.from_preset(...).fit(...)`` while
measuring wall-time, peak RAM (RSS), peak VRAM (CUDA only — MPS has no exact
peak API), and the disk delta in the HF Hub cache.

The output is a JSON file with per-preset predicted vs. actual values plus
ratios, and a side-by-side table on stdout for quick eyeballing.

Usage:
    python scripts/calibrate_advisor.py \\
        --dataset tests/assets/data/clinc_subset.json \\
        --presets classic-light classic-medium \\
        --output calibration.json \\
        --max-trials 3

The ``--skip-fit`` flag runs only the predicted side, useful for sanity-checking
the preflight numbers across presets without paying for fits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

from autointent import Dataset, Pipeline
from autointent._advisor import (
    BUNDLED_PRESETS,
    PreflightReport,
    detect_hardware,
    run_preflight,
    stats_from_dataset_obj,
)
from autointent._callbacks.base import OptimizerCallback
from autointent.configs import HPOConfig, LoggingConfig

logger = logging.getLogger("calibrate_advisor")

_BYTES_PER_GB = 1024**3


@dataclass
class CalibrationRow:
    """One preset's predicted vs. actual numbers."""

    preset: str
    predicted: dict[str, float] = field(default_factory=dict)
    actual: dict[str, float | None] = field(default_factory=dict)
    ratios: dict[str, float | None] = field(default_factory=dict)
    findings: int = 0
    findings_over: int = 0
    # Per-module records from _ModuleTracker: [{module, num, config, duration_s, peak_vram_gb?}, ...]
    modules: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibrate_advisor",
        description="Compare advisor preflight estimates to real Pipeline.fit measurements.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        type=str,
        help=(
            "Either a local JSON path (loaded via ``Dataset.from_json``) or an HF Hub repo id "
            "such as ``DeepPavlov/banking77`` (loaded via ``Dataset.from_hub``)."
        ),
    )
    p.add_argument(
        "--presets",
        nargs="+",
        default=None,
        help="Preset names to run (default: every preset in BUNDLED_PRESETS).",
    )
    p.add_argument("--output", type=Path, default=Path("calibration.json"), help="Where to write the JSON report.")
    p.add_argument("--max-trials", type=int, default=None, help="Override hpo_config.n_trials for faster runs.")
    p.add_argument(
        "--skip-fit",
        action="store_true",
        help="Only run preflight (no fit) — useful for sanity-checking estimates.",
    )
    p.add_argument(
        "--poll-interval-ms",
        type=int,
        default=100,
        help="RSS polling interval during fit (ms). Lower is more accurate but more overhead.",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help=(
            "Attach the W&B reporter so per-step GPU/system metrics land in wandb.ai. "
            "Requires ``wandb`` installed + ``WANDB_API_KEY`` in the environment."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


# === measurement helpers =================================================


def _hf_cache_dir() -> Path:
    """Return the active HF Hub cache directory ($HF_HOME / ~/.cache/huggingface)."""
    return Path(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"))


def _dir_size_gb(path: Path) -> float:
    """Disk usage of ``path`` in GB; 0 when the directory is missing."""
    if not path.exists():
        return 0.0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total / _BYTES_PER_GB


class _PeakSampler:
    """Background thread tracking peak RSS and (on MPS) peak GPU allocation.

    CUDA has an accurate native peak-memory API and doesn't need polling; we
    still read it after the fit. MPS lacks a peak API, so the sampler polls
    ``torch.mps.current_allocated_memory()`` alongside RSS and keeps the max.
    """

    def __init__(self, interval_s: float = 0.1, *, sample_mps: bool = False) -> None:
        self._interval_s = interval_s
        self._proc = psutil.Process()
        self.peak_ram_gb = self._proc.memory_info().rss / _BYTES_PER_GB
        self.peak_mps_gb: float | None = 0.0 if sample_mps else None
        self._sample_mps = sample_mps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _PeakSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            torch = None  # type: ignore[assignment]
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / _BYTES_PER_GB
                if rss > self.peak_ram_gb:
                    self.peak_ram_gb = rss
                if self._sample_mps and torch is not None:
                    mps = float(torch.mps.current_allocated_memory()) / _BYTES_PER_GB
                    if self.peak_mps_gb is None or mps > self.peak_mps_gb:
                        self.peak_mps_gb = mps
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop.wait(self._interval_s)


def _reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _read_vram_peak_gb(accelerator: str) -> float | None:
    """Peak VRAM/GPU in GB. CUDA uses the native peak API; MPS uses the polled sampler value (caller-side)."""
    try:
        import torch
    except ImportError:
        return None
    if accelerator == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / _BYTES_PER_GB
    return None


# === per-module tracking =================================================


class _ModuleTracker(OptimizerCallback):
    """Records per-module wall time and peak VRAM.

    Hooks ``start_module`` / ``end_module`` on the CallbackHandler so we get
    one record per (module_name, trial_num). CUDA peak VRAM is reset per module
    via ``torch.cuda.reset_peak_memory_stats``; MPS is sampled at ``end_module``
    (no per-module peak API, so it's the moment-in-time allocation).
    """

    name = "calibration_tracker"

    def __init__(self) -> None:  # noqa: D401
        self.records: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None

    def start_run(self, run_name: str, dirpath: Path, log_interval_time: float) -> None:  # noqa: ARG002
        pass

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        # Only capture JSON-safe scalars in the config snapshot.
        safe_config = {
            k: v for k, v in module_kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None
        }
        self._current = {
            "module": module_name,
            "num": num,
            "config": safe_config,
            "_start": time.perf_counter(),
        }

    def log_value(self, **kwargs: Any) -> None:  # noqa: ANN401, ARG002
        pass

    def log_metrics(self, metrics: dict[str, Any]) -> None:  # noqa: ARG002
        pass

    def end_module(self) -> None:
        if self._current is None:
            return
        rec = self._current
        rec["duration_s"] = time.perf_counter() - rec.pop("_start")
        try:
            import torch

            if torch.cuda.is_available():
                rec["peak_vram_gb"] = float(torch.cuda.max_memory_allocated()) / _BYTES_PER_GB
            elif torch.backends.mps.is_available():
                # MPS has no per-module peak API — snapshot the current allocation.
                rec["peak_vram_gb"] = float(torch.mps.current_allocated_memory()) / _BYTES_PER_GB
        except (ImportError, AttributeError):
            pass
        self.records.append(rec)
        self._current = None

    def end_run(self) -> None:
        pass

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:  # noqa: ARG002
        pass


def _attach_tracker(pipeline: Pipeline, tracker: _ModuleTracker) -> None:
    """Instance-patch ``pipeline._fit`` so ``tracker`` is appended to the callback chain."""
    original_fit = pipeline._fit  # noqa: SLF001

    def patched(context: Any) -> Any:  # noqa: ANN401
        context.callback_handler.callbacks.append(tracker)
        return original_fit(context)

    pipeline._fit = patched  # type: ignore[method-assign]  # noqa: SLF001


# === per-preset run ======================================================


def _override_trials(pipeline: Pipeline, max_trials: int | None, *, enable_wandb: bool) -> None:
    """Cap n_trials, disable dumping, optionally enable W&B for post-run analysis."""
    updates: dict[str, Any] = {}
    if max_trials is not None:
        updates["n_trials"] = max_trials
    if enable_wandb:
        # Trigger built-in per-run system-metrics collection in W&B.
        updates["report_to"] = ["wandb"]
    if updates:
        pipeline.set_config(pipeline.hpo_config.model_copy(update=updates))
    # We don't want the calibration run to leave dumped module weights on disk.
    pipeline.set_config(LoggingConfig(dump_modules=False, clear_ram=True))


def _calibrate_one(
    *,
    preset: str,
    dataset: Dataset,
    stats: Any,  # noqa: ANN401
    hardware: Any,  # noqa: ANN401
    max_trials: int | None,
    skip_fit: bool,
    poll_interval_ms: int,
    enable_wandb: bool,
) -> CalibrationRow:
    row = CalibrationRow(preset=preset)

    # === predicted ======================================================
    try:
        pipeline = Pipeline.from_preset(preset)
    except Exception as e:  # noqa: BLE001
        row.error = f"from_preset failed: {e}"
        return row

    _override_trials(pipeline, max_trials, enable_wandb=enable_wandb)

    try:
        report: PreflightReport = run_preflight(
            pipeline._build_advisor_config(),  # noqa: SLF001
            stats,
            hardware,
        )
    except Exception as e:  # noqa: BLE001
        row.error = f"preflight failed: {e}"
        return row

    row.predicted = {
        "time_h": report.resource.time_hours,
        "ram_gb": report.resource.ram_gb,
        "vram_gb": report.resource.vram_gb,
        "disk_download_gb": report.resource.disk_download_gb,
        "disk_cached_gb": report.resource.disk_cached_gb,
        "disk_embedding_cache_gb": report.resource.disk_embedding_cache_gb,
    }
    row.findings = len(report.findings)
    row.findings_over = sum(1 for f in report.findings if f.severity.value == "over")
    if report.low_confidence:
        row.notes.append("low-confidence (heuristic fallback in use)")

    if skip_fit:
        return row

    # === actual =========================================================
    hf_cache = _hf_cache_dir()
    cache_before = _dir_size_gb(hf_cache)
    _reset_vram_peak()

    tracker = _ModuleTracker()
    _attach_tracker(pipeline, tracker)

    is_mps = hardware.accelerator == "mps"
    start = time.perf_counter()
    try:
        with _PeakSampler(interval_s=poll_interval_ms / 1000.0, sample_mps=is_mps) as sampler:
            pipeline.fit(dataset, preflight="off")
    except Exception as e:  # noqa: BLE001
        row.error = f"fit failed: {e}"
        row.modules = tracker.records  # keep whatever we collected
        return row
    elapsed_s = time.perf_counter() - start

    cache_after = _dir_size_gb(hf_cache)
    actual_time_h = elapsed_s / 3600.0
    actual_ram_gb = sampler.peak_ram_gb
    actual_vram_gb = _read_vram_peak_gb(hardware.accelerator)
    if actual_vram_gb is None and is_mps:
        actual_vram_gb = sampler.peak_mps_gb
    actual_disk_download_gb = max(0.0, cache_after - cache_before)

    row.actual = {
        "time_h": actual_time_h,
        "ram_gb": actual_ram_gb,
        "vram_gb": actual_vram_gb,
        "disk_download_gb": actual_disk_download_gb,
    }
    row.modules = tracker.records
    if enable_wandb:
        row.notes.append("W&B reporter enabled — inspect wandb.ai run group for per-step GPU/system metrics")

    def _ratio(actual: float | None, predicted: float) -> float | None:
        if actual is None or predicted <= 0:
            return None
        return actual / predicted

    row.ratios = {
        "time": _ratio(actual_time_h, row.predicted["time_h"]),
        "ram": _ratio(actual_ram_gb, row.predicted["ram_gb"]),
        "vram": _ratio(actual_vram_gb, row.predicted["vram_gb"]),
        "disk_download": _ratio(actual_disk_download_gb, row.predicted["disk_download_gb"]),
    }
    return row


# === rendering ===========================================================


_COLS = [
    ("preset", "Preset", 22),
    ("pred_time", "pred_time_h", 12),
    ("act_time", "act_time_h", 12),
    ("r_time", "ratio_t", 8),
    ("pred_ram", "pred_ram_gb", 12),
    ("act_ram", "act_ram_gb", 12),
    ("r_ram", "ratio_r", 8),
    ("pred_vram", "pred_vram_gb", 13),
    ("act_vram", "act_vram_gb", 13),
    ("r_vram", "ratio_v", 8),
]


def _fmt_cell(value: Any) -> str:  # noqa: ANN401
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == 0:
            return "0.00"
        return f"{value:.2f}" if abs(value) >= 0.01 else f"{value:.4f}"
    return str(value)


def _print_summary(rows: list[CalibrationRow]) -> None:
    """Pretty side-by-side table for stdout."""
    header = "  ".join(label.ljust(width) for _, label, width in _COLS)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = {
            "preset": row.preset,
            "pred_time": row.predicted.get("time_h"),
            "act_time": row.actual.get("time_h"),
            "r_time": row.ratios.get("time"),
            "pred_ram": row.predicted.get("ram_gb"),
            "act_ram": row.actual.get("ram_gb"),
            "r_ram": row.ratios.get("ram"),
            "pred_vram": row.predicted.get("vram_gb"),
            "act_vram": row.actual.get("vram_gb"),
            "r_vram": row.ratios.get("vram"),
        }
        print("  ".join(_fmt_cell(cells[key]).ljust(width) for key, _, width in _COLS))
        if row.error:
            print(f"    ! {row.error}")
        for note in row.notes:
            print(f"    * {note}")
        for mod in row.modules:
            duration = mod.get("duration_s")
            vram = mod.get("peak_vram_gb")
            duration_s = f"{duration:.2f}s" if duration is not None else "-"
            vram_s = f"{vram:.2f} GB" if vram is not None else "-"
            print(f"      · {mod.get('module', '?')}#{mod.get('num', '?')}  {duration_s}  vram={vram_s}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    presets = args.presets or list(BUNDLED_PRESETS)
    unknown = [p for p in presets if p not in BUNDLED_PRESETS]
    if unknown:
        parser.error(f"Unknown preset(s): {', '.join(unknown)}. Known: {', '.join(BUNDLED_PRESETS)}")

    dataset_path = Path(args.dataset)
    if dataset_path.is_file():
        logger.info("Loading dataset from local file %s", dataset_path)
        dataset = Dataset.from_json(dataset_path)
        dataset_source = str(dataset_path)
    else:
        logger.info("Loading dataset from HF Hub: %s", args.dataset)
        try:
            dataset = Dataset.from_hub(args.dataset)
        except Exception as e:  # noqa: BLE001
            parser.error(f"Could not load '{args.dataset}' as a local JSON file or as a Hub repo id: {e}")
        dataset_source = f"hub:{args.dataset}"
    stats = stats_from_dataset_obj(dataset)
    hardware = detect_hardware()
    logger.info(
        "Hardware: %s (%s) — %.1f GB VRAM, %.0f GB RAM, %.0f GB free disk",
        hardware.accelerator,
        hardware.device_name,
        hardware.vram_gb,
        hardware.ram_gb,
        hardware.free_disk_gb,
    )

    rows: list[CalibrationRow] = []
    for preset in presets:
        logger.info("=== %s ===", preset)
        row = _calibrate_one(
            preset=preset,
            dataset=dataset,
            stats=stats,
            hardware=hardware,
            max_trials=args.max_trials,
            skip_fit=args.skip_fit,
            poll_interval_ms=args.poll_interval_ms,
            enable_wandb=args.wandb,
        )
        rows.append(row)

    payload = {
        "hardware": {
            "accelerator": hardware.accelerator,
            "device_name": hardware.device_name,
            "vram_gb": hardware.vram_gb,
            "ram_gb": hardware.ram_gb,
            "free_disk_gb": hardware.free_disk_gb,
        },
        "dataset": {
            "path": dataset_source,
            "n_samples": stats.n_samples,
            "n_classes": stats.n_classes,
            "avg_tokens": stats.avg_tokens,
            "multilabel": stats.multilabel,
        },
        "max_trials_override": args.max_trials,
        "skip_fit": args.skip_fit,
        "rows": [r.to_dict() for r in rows],
    }
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote report to %s", args.output)

    print()
    _print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
