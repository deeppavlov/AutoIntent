"""Phase 3 of issue #39: exercise the reduce-to-fit path on real constrained hardware.

The calibrator runs report-only (``preflight="off"``), so the strict gate and
``reduce_to_fit`` have never been driven against a box that actually can't fit
the preset. This script does three things on the live machine:

  A. ``Pipeline.fit(..., preflight="strict")`` on ``transformers-heavy`` —
     must raise ``PreflightError`` *before* any CUDA allocation instead of
     letting the run walk into an OOM.
  B. ``reduce_to_fit`` on ``transformers-heavy`` — a single-scoring-module
     preset, so the only reachable outcome is ``ReduceToFitError``. We check
     the error is the explicit "everything was pruned" one and record whether
     it points the user anywhere useful.
  C. ``reduce_to_fit`` on a mixed infeasible search space (deberta-v3-large +
     knn + linear) — the case where pruning *can* succeed. We then really fit
     the pruned config to prove the survivor is runnable, not just feasible
     on paper.

Usage:
    uv run --no-sync python scripts/phase3_reduce_to_fit.py \
        --dataset DeepPavlov/banking77 --subsample-per-class 30 \
        --output calibration_runs/phase3.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from autointent import Dataset, Pipeline, setup_logging
from autointent._advisor import (
    ReduceToFitError,
    detect_hardware,
    load_config,
    reduce_to_fit,
    run_preflight,
    stats_from_dataset_obj,
)
from autointent._pipeline import PreflightError

setup_logging("WARNING", log_filename="phase3.log")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("phase3")

_BYTES_PER_GB = 1024**3


def _subsample_per_class(dataset: Dataset, cap: int) -> Dataset:
    """Deterministic first-N-per-class slice of the train split.

    Imported from the calibrator rather than reimplemented so Phase 3 sees
    byte-identical dataset stats to the Phase 1 / Phase 2 runs.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from calibrate_advisor import _subsample_per_class as _impl

    return _impl(dataset, cap)


def _vram_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / _BYTES_PER_GB


def _reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _report_digest(report: Any) -> dict[str, Any]:
    return {
        "headroom": report.headroom.value,
        "is_feasible": report.is_feasible,
        "vram_gb": round(report.resource.vram_gb, 3),
        "ram_gb": round(report.resource.ram_gb, 3),
        "time_hours": round(report.resource.time_hours, 3),
        "low_confidence": report.low_confidence,
        "findings": [
            {"severity": f.severity.value, "metric": f.metric, "message": f.message} for f in report.findings
        ],
    }


def _scoring_modules(config: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for node in config.get("search_space", []):
        if node.get("node_type") == "scoring":
            out.extend(str(e.get("module_name")) for e in node.get("search_space", []))
    return out


def _mixed_infeasible_config(base_heavy: dict[str, Any], base_classic: dict[str, Any]) -> dict[str, Any]:
    """deberta-v3-large (infeasible on 6 GB) + knn/linear (cheap) in one scoring node.

    This is the configuration shape reduce_to_fit was actually designed for:
    something expensive to drop, something cheap to keep.
    """
    cfg = copy.deepcopy(base_heavy)
    heavy_scoring = next(n for n in cfg["search_space"] if n["node_type"] == "scoring")
    classic_scoring = next(n for n in base_classic["search_space"] if n["node_type"] == "scoring")
    heavy_scoring["search_space"] = [
        *copy.deepcopy(heavy_scoring["search_space"]),
        *copy.deepcopy([e for e in classic_scoring["search_space"] if e["module_name"] in {"knn", "linear"}]),
    ]
    cfg["embedder_config"] = copy.deepcopy(base_classic.get("embedder_config", {}))
    cfg["hpo_config"]["n_trials"] = 2
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser("phase3_reduce_to_fit")
    parser.add_argument("--dataset", default="DeepPavlov/banking77")
    parser.add_argument("--subsample-per-class", type=int, default=30)
    parser.add_argument("--output", default="calibration_runs/phase3.json")
    parser.add_argument(
        "--skip-real-fit",
        action="store_true",
        help="Skip step C's real fit of the pruned config (paper-feasibility only).",
    )
    args = parser.parse_args()

    hardware = detect_hardware()
    print(
        f"Hardware: {hardware.accelerator} ({hardware.device_name}) — "
        f"{hardware.vram_gb:.2f} GB VRAM, {hardware.ram_gb:.1f} GB RAM, class={hardware.device_class}"
    )

    dataset = Dataset.from_hub(args.dataset)
    if args.subsample_per_class:
        dataset = _subsample_per_class(dataset, args.subsample_per_class)
    stats = stats_from_dataset_obj(dataset)
    print(f"Dataset: n_samples={stats.n_samples} n_classes={stats.n_classes}")

    heavy_cfg, _ = load_config("transformers-heavy")
    classic_cfg, _ = load_config("classic-light")

    results: dict[str, Any] = {
        "hardware": {
            "accelerator": hardware.accelerator,
            "device_name": hardware.device_name,
            "vram_gb": hardware.vram_gb,
            "ram_gb": hardware.ram_gb,
            "device_class": hardware.device_class,
        },
        "dataset": {"source": args.dataset, "n_samples": stats.n_samples, "n_classes": stats.n_classes},
    }

    # === A: strict gate must fire before any CUDA allocation ==============
    print("\n=== A. Pipeline.fit(preflight='strict') on transformers-heavy ===")
    _reset_vram_peak()
    step_a: dict[str, Any] = {}
    pipeline = Pipeline.from_preset("transformers-heavy")
    t0 = time.perf_counter()
    try:
        pipeline.fit(dataset, preflight="strict")
    except PreflightError as e:
        step_a = {
            "outcome": "PreflightError",
            "raised_before_alloc": _vram_peak_gb() < 0.1,  # noqa: PLR2004
            "vram_peak_gb": round(_vram_peak_gb(), 4),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "message": str(e)[:1000],
        }
    except Exception as e:  # noqa: BLE001
        step_a = {
            "outcome": type(e).__name__,
            "vram_peak_gb": round(_vram_peak_gb(), 4),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "message": str(e)[:1000],
            "traceback": traceback.format_exc()[-2000:],
        }
    else:
        step_a = {
            "outcome": "fit-completed",
            "vram_peak_gb": round(_vram_peak_gb(), 4),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
    results["A_strict_gate"] = step_a
    print(json.dumps(step_a, indent=2)[:1200])

    # === B: reduce_to_fit on the single-module heavy preset ===============
    print("\n=== B. reduce_to_fit(transformers-heavy) ===")
    step_b: dict[str, Any] = {"scoring_modules_before": _scoring_modules(heavy_cfg)}
    try:
        pruned, report = reduce_to_fit(copy.deepcopy(heavy_cfg), stats, hardware)
    except ReduceToFitError as e:
        step_b.update(
            {
                "outcome": "ReduceToFitError",
                "message": str(e),
                "scoring_modules_after": _scoring_modules(e.pruned_config),
                "last_report": _report_digest(e.last_report),
                # The issue asks whether the error "points at a lighter preset".
                "names_a_lighter_preset": any(
                    p in str(e) for p in ("classic", "nn-", "zero-shot", "transformers-light", "preset")
                ),
            }
        )
    except Exception as e:  # noqa: BLE001
        step_b.update({"outcome": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()[-2000:]})
    else:
        step_b.update(
            {
                "outcome": "pruned-to-feasible",
                "scoring_modules_after": _scoring_modules(pruned),
                "report": _report_digest(report),
            }
        )
    results["B_reduce_heavy"] = step_b
    print(json.dumps(step_b, indent=2)[:1500])

    # === C: reduce_to_fit on a mixed search space + real fit ==============
    print("\n=== C. reduce_to_fit(deberta-v3-large + knn + linear) ===")
    mixed = _mixed_infeasible_config(heavy_cfg, classic_cfg)
    step_c: dict[str, Any] = {"scoring_modules_before": _scoring_modules(mixed)}
    before = run_preflight(copy.deepcopy(mixed), stats, hardware, preset_name="mixed")
    step_c["report_before"] = _report_digest(before)
    print(f"  before: headroom={before.headroom.value} vram={before.resource.vram_gb:.2f} GB")

    try:
        pruned_mixed, report_mixed = reduce_to_fit(copy.deepcopy(mixed), stats, hardware)
    except ReduceToFitError as e:
        step_c.update({"outcome": "ReduceToFitError", "message": str(e), "last_report": _report_digest(e.last_report)})
    except Exception as e:  # noqa: BLE001
        step_c.update({"outcome": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()[-2000:]})
    else:
        step_c.update(
            {
                "outcome": "pruned-to-feasible",
                "scoring_modules_after": _scoring_modules(pruned_mixed),
                "report_after": _report_digest(report_mixed),
            }
        )
        print(
            f"  after: modules={_scoring_modules(pruned_mixed)} "
            f"headroom={report_mixed.headroom.value} vram={report_mixed.resource.vram_gb:.2f} GB"
        )

        # The whole point: is the survivor actually runnable on this box?
        if not args.skip_real_fit:
            print("  fitting the pruned config for real ...")
            _reset_vram_peak()
            t0 = time.perf_counter()
            try:
                pruned_pipeline = Pipeline.from_optimization_config(pruned_mixed)
                pruned_pipeline.fit(dataset, preflight="off")
            except Exception as e:  # noqa: BLE001
                step_c["real_fit"] = {
                    "outcome": "failed",
                    "error": f"{type(e).__name__}: {e}"[:600],
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                    "vram_peak_gb": round(_vram_peak_gb(), 3),
                    "traceback": traceback.format_exc()[-2000:],
                }
            else:
                step_c["real_fit"] = {
                    "outcome": "ok",
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                    "vram_peak_gb": round(_vram_peak_gb(), 3),
                }
            print(f"  real fit: {json.dumps(step_c['real_fit'])[:500]}")

    results["C_reduce_mixed"] = step_c

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
