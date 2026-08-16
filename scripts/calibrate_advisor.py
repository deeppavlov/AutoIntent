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
from autointent.configs import LoggingConfig
from autointent import setup_logging

setup_logging("INFO", log_filename="logs.log")
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("calibrate_advisor")

_BYTES_PER_GB = 1024**3
# Anything above this still allocated after a preset finishes means references
# outlived the run and the next preset's measurement can't be trusted.
_LEAK_WARN_GB = 0.25


@dataclass
class CalibrationRow:
    """One preset's predicted vs. actual numbers."""

    preset: str
    predicted: dict[str, float] = field(default_factory=dict)
    actual: dict[str, float | None] = field(default_factory=dict)
    findings: int = 0
    findings_over: int = 0
    # Top-line advisor verdict. ``headroom`` is the worst severity across all
    # findings ("ample" / "tight" / "over"); ``is_feasible`` is ``headroom !=
    # over``. Both live on ``PreflightReport`` but were previously dropped on
    # the floor here, so a calibration JSON could not answer the one question
    # the advisor exists to answer. ``severity_by_metric`` keeps the per-metric
    # breakdown (vram / ram / disk / time) so a RED can be attributed.
    headroom: str | None = None
    is_feasible: bool | None = None
    severity_by_metric: dict[str, str] = field(default_factory=dict)
    # Resolved model name per driver, e.g. {"scoring/bert": "microsoft/deberta-v3-large"}.
    # Recorded so a local preset swap can never masquerade as "transformers-heavy".
    models: dict[str, str] = field(default_factory=dict)
    # Per-module records from _ModuleTracker: [{module, num, config, duration_s, peak_vram_gb?}, ...]
    modules: list[dict[str, Any]] = field(default_factory=list)
    cache_policy: str = "unknown"  # "cold" (embeddings cache cleared) | "warm" (kept as-is)
    low_confidence: bool = False  # advisor fell back to heuristic HF-metadata for one+ models
    repeat_idx: int = 0  # 0-based index within a (preset, dataset) repeat group
    # ``skipped`` is set when the preset needs an optional extra that isn't
    # installed (peft / catboost / openai / ...). We still populate ``error``
    # for the summary, but callers analysing the JSON should treat
    # ``skipped=True`` rows separately from ``error != None && skipped=False``
    # rows (real crashes) — the former are expected and shouldn't count as
    # advisor failures.
    skipped: bool = False
    # Snapshot of ``autointent-advisor inspect <preset> --json`` run in-process
    # under the same stats + budget as the direct-API preflight. Lets us catch
    # a CLI-wrapper regression (JSON schema drift, feasibility verdict flip)
    # without a separate subprocess round-trip. None means we didn't run it
    # (e.g. skipped row, or CLI itself crashed — see notes for the reason).
    cli_smoke: dict[str, Any] | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize with ratios + role-decomposed timings computed at read time.

        Storing ratios in the row is a bug magnet: any late edit to
        ``predicted``/``actual`` gets missed. We compute them from the current
        row values at serialization time so consumers can trust ``row["ratios"]``.
        ``time_by_role_s`` splits the measured wall-time across embedder /
        scorer / decision so classic-preset time can be interpreted (embedder
        forward vs sklearn fit) without re-walking ``modules``.
        """
        payload = asdict(self)
        payload["ratios"] = self._ratios()
        payload["time_by_role_s"] = _sum_time_by_role(self.modules)
        return payload

    def _ratios(self) -> dict[str, float | None]:
        keys = ("time_h", "ram_gb", "vram_gb", "disk_download_gb", "disk_embedding_cache_gb")
        out: dict[str, float | None] = {}
        for key in keys:
            actual = self.actual.get(key)
            predicted = self.predicted.get(key)
            if actual is None or predicted is None or predicted <= 0:
                out[key] = None
            else:
                out[key] = actual / predicted
        return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibrate_advisor",
        description="Compare advisor preflight estimates to real Pipeline.fit measurements.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        type=str,
        help=(
            "One or more datasets — each is either a local JSON path (loaded via "
            "``Dataset.from_json``) or an HF Hub repo id such as ``DeepPavlov/banking77`` "
            "(loaded via ``Dataset.from_hub``). Every preset runs against every dataset, "
            "so pairing a multilabel + long-token + small + large dataset exercises the "
            "n_samples / n_classes / avg_tokens surfaces of the advisor's formulas."
        ),
    )
    p.add_argument(
        "--subsample-per-class",
        type=int,
        default=None,
        help=(
            "Cap each class to at most N training samples (deterministic first-N slice) "
            "before running. Lets one big dataset stand in as a 'small' shape — enough to "
            "exercise ``LogisticRegressionCV cv=3`` split-readiness and rare-class findings."
        ),
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Run each (preset, dataset) N times so ratio gaps have variance bars. "
            "The summary prints mean ± stdev for the actual measurements across "
            "repeats; individual repeat rows are still written to the JSON with "
            "``repeat_idx`` so consumers can compute their own aggregates. Default: 1."
        ),
    )
    p.add_argument(
        "--presets",
        nargs="+",
        default=None,
        help=(
            "Preset names to run (default: every preset in BUNDLED_PRESETS). "
            "Items ending in .yaml/.yml are treated as paths to a preset file — "
            "used to run e.g. ``scripts/coverage_preset.yaml`` which packs "
            "lora/ptuning/dnnc/gcn/cross-encoder into one small run for module "
            "coverage without touching the shipped presets."
        ),
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
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Suffix appended to each preset's LoggingConfig.run_name — the resulting "
            "value is ``{preset}_{run_name}`` and is used by the LoggingHandler as the "
            "W&B run group / on-disk dump directory name."
        ),
    )
    p.add_argument(
        "--clear-embedding-cache",
        action="store_true",
        help=(
            "Wipe ``<user_cache_dir>/autointent/embeddings/`` before each preset so every "
            "measurement reflects a COLD run (embedder forward not skipped). Without this "
            "flag, the cross-run cache silently makes later runs look artificially cheap."
        ),
    )
    p.add_argument(
        "--budget-vram-gb",
        type=float,
        default=None,
        help=(
            "Override the detected VRAM budget passed to run_preflight, e.g. ``--budget-vram-gb 8`` "
            "to exercise the constrained-hardware / severity paths on a big box without needing "
            "a small GPU. Does NOT affect the real fit — only the predicted-side estimate."
        ),
    )
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help=(
            "Fail fast if PyTorch can't initialize CUDA (guards against the silent "
            "'2 GPUs detected, but torch runs on CPU' driver-mismatch trap)."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


# === measurement helpers =================================================


def _hf_cache_dir() -> Path:
    """Return the active HF Hub cache directory ($HF_HOME / ~/.cache/huggingface)."""
    return Path(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"))


def _embeddings_cache_dir() -> Path:
    """Return autointent's embeddings-cache dir (``<user_cache_dir>/autointent/embeddings/``).

    Uses the same ``appdirs.user_cache_dir("autointent")`` path as
    :func:`autointent._wrappers.embedder.utils.get_embeddings_path` so the
    harness reads/clears the same directory the runtime writes to.
    """
    from autointent._wrappers.embedder.utils import get_embeddings_path

    return get_embeddings_path("_probe").parent


def _clear_embeddings_cache() -> int:
    """Delete every ``*.npy`` file in the embeddings cache. Returns count removed."""
    cache = _embeddings_cache_dir()
    if not cache.exists():
        return 0
    removed = 0
    for path in cache.glob("*.npy"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


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
    """Background thread polling peak RSS + (optionally) MPS / CUDA current
    allocation. CUDA polling catches allocations that fall outside any
    module bracket — the per-module tracker's peak counter gets reset at each
    start_module, losing anything allocated before it (e.g. the embedder
    forward during pipeline setup). Best-effort: sub-poll-interval spikes
    can be missed."""

    def __init__(
        self, interval_s: float = 0.1, *, sample_mps: bool = False, sample_cuda: bool = False,
    ) -> None:
        self._interval_s = interval_s
        self._proc = psutil.Process()
        self.peak_ram_gb = self._proc.memory_info().rss / _BYTES_PER_GB
        self.peak_mps_gb: float | None = 0.0 if sample_mps else None
        self.peak_cuda_gb: float | None = 0.0 if sample_cuda else None
        self._sample_mps = sample_mps
        self._sample_cuda = sample_cuda
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
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / _BYTES_PER_GB
                self.peak_ram_gb = max(self.peak_ram_gb, rss)
                if self._sample_mps and torch is not None:
                    mps = float(torch.mps.current_allocated_memory()) / _BYTES_PER_GB
                    if self.peak_mps_gb is None or mps > self.peak_mps_gb:
                        self.peak_mps_gb = mps
                if self._sample_cuda and torch is not None and torch.cuda.is_available():
                    cuda = float(torch.cuda.memory_allocated()) / _BYTES_PER_GB
                    if self.peak_cuda_gb is None or cuda > self.peak_cuda_gb:
                        self.peak_cuda_gb = cuda
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


# Static module_name → role classification. Used to tag tracker records so
# downstream analysis can decompose classic-preset wall-time into
# embedder-forward vs scorer-fit vs decision-search — the follow-up review's
# R4-P1 #30 asked for this because today classic wall-time conflates the two.
_EMBEDDER_MODULE_NAMES = frozenset(
    {"sentence_transformer", "openai_embedder", "vllm_embedder", "hashing_vectorizer"},
)
_DECISION_MODULE_NAMES = frozenset({"threshold", "argmax", "jinoos", "tunable", "adaptive"})


def _classify_module_role(module_name: str) -> str:
    """Bucket module_name into ``embedder`` / ``decision`` / ``scorer``.

    Everything not in the known embedder or decision sets is treated as a
    scorer — so newly added scorer modules land in the right bucket by default
    and only new decision/embedder modules would need to update the sets.
    """
    if module_name in _EMBEDDER_MODULE_NAMES:
        return "embedder"
    if module_name in _DECISION_MODULE_NAMES:
        return "decision"
    return "scorer"


# Base class for _StepTimingCallback. We inherit from HF's real
# ``TrainerCallback`` when transformers is installed — that gives us the
# correct no-op default for every ``on_*`` hook (on_train_begin/on_log/
# on_save/...) automatically, so we only override the two we time. HF's
# ``CallbackHandler.call_event`` dispatches with a bare ``getattr`` (no
# hasattr probe), so a plain class missing hooks would ``AttributeError``
# the moment a real trial calls e.g. ``on_train_begin``.
#
# When transformers isn't installed we fall back to ``object`` so the
# harness still imports on classic-only runs. In that case the callback is
# never actually instantiated (``_patch_trainer_for_step_timing`` bails out
# in the same ``ImportError`` branch), so the fallback base is only needed
# to make the class definition itself succeed.
try:
    from transformers import TrainerCallback as _StepTimingBase  # type: ignore[import-not-found]
except ImportError:
    _StepTimingBase = object  # type: ignore[assignment,misc]


class _StepTimingCallback(_StepTimingBase):  # type: ignore[misc,valid-type]
    """HF ``TrainerCallback`` that appends the wall-time of each optimizer step
    to a caller-owned list.

    Injected into every ``transformers.Trainer`` for the duration of a fit via
    :func:`_patch_trainer_for_step_timing`. The sink is the current module's
    step buffer on :class:`_ModuleTracker`, so the transformer's per-step
    latency lands in that module's record automatically — no plumbing across
    module boundaries.
    """

    def __init__(self, sink: list[float]) -> None:
        # TrainerCallback.__init__ takes (*args, **kwargs); calling super is
        # safe both when the base is the real HF class and when it's ``object``.
        super().__init__()
        self._sink = sink
        self._t0: float | None = None

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:  # noqa: ANN401, ARG002
        self._t0 = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:  # noqa: ANN401, ARG002
        if self._t0 is not None:
            self._sink.append(time.perf_counter() - self._t0)
            self._t0 = None


def _summarize_step_times(step_times: list[float]) -> dict[str, float]:
    """Fold a list of per-step wall-times into summary stats for the row.

    ``seconds_per_step`` is what the advisor's transformer-time baseline
    encodes (currently a flat ~1 s constant); logging measured ``mean`` and
    ``p95`` lets the baseline be recalibrated directly from row data instead
    of eyeballed off a wandb dashboard.
    """
    import statistics as _stats

    if not step_times:
        return {}
    if len(step_times) == 1:
        return {"n_steps": 1, "mean_step_s": step_times[0], "p95_step_s": step_times[0]}
    sorted_st = sorted(step_times)
    p95_idx = min(len(sorted_st) - 1, int(round(0.95 * (len(sorted_st) - 1))))
    return {
        "n_steps": len(step_times),
        "mean_step_s": _stats.fmean(step_times),
        "p95_step_s": sorted_st[p95_idx],
        "total_step_s": sum(step_times),
    }


def _patch_trainer_for_step_timing(tracker: _ModuleTracker) -> Any:  # noqa: ANN401
    """Monkey-patch ``transformers.Trainer.__init__`` to inject a step-timing
    callback bound to ``tracker._current_step_buffer`` — the list on the
    module record currently being tracked.

    Returns a callable that undoes the patch. No-ops (returns a no-op undoer)
    when transformers isn't importable, so classic-only presets aren't blocked.
    """
    try:
        from transformers import Trainer  # type: ignore[import-not-found]
    except ImportError:
        return lambda: None

    original_init = Trainer.__init__

    def patched(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        original_init(self, *args, **kwargs)
        buffer = tracker.current_step_buffer()
        if buffer is not None:
            self.add_callback(_StepTimingCallback(buffer))

    Trainer.__init__ = patched  # type: ignore[method-assign]

    def _undo() -> None:
        Trainer.__init__ = original_init  # type: ignore[method-assign]

    return _undo


class _ModuleTracker(OptimizerCallback):
    """Records per-module wall time and peak VRAM.

    Hooks ``start_module`` / ``end_module`` on the CallbackHandler so we get
    one record per (module_name, trial_num). CUDA peak VRAM is reset per module
    via ``torch.cuda.reset_peak_memory_stats``; MPS is sampled at ``end_module``
    (no per-module peak API, so it's the moment-in-time allocation).

    Because per-module CUDA resets clobber the global ``max_memory_allocated``
    counter, the tracker also keeps ``self.peak_vram_gb_overall`` — the max
    across every recorded module. The calibration script reads this instead of
    the post-fit ``torch.cuda.max_memory_allocated()`` value, which by then
    reflects only the last (usually CPU-only decision) module.
    """

    name = "calibration_tracker"

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._current_step_buffer: list[float] | None = None
        self.peak_vram_gb_overall: float = 0.0

    def current_step_buffer(self) -> list[float] | None:
        """Return the per-step wall-time list the ``_StepTimingCallback``
        should append to. ``None`` when no module is currently being tracked
        (e.g. between modules) — the callback then skips."""
        return self._current_step_buffer

    def start_run(self, run_name: str, dirpath: Path, log_interval_time: float) -> None:
        pass

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        # Only capture JSON-safe scalars in the config snapshot.
        safe_config = {k: v for k, v in module_kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
        self._current_step_buffer = []
        self._current = {
            "module": module_name,
            "role": _classify_module_role(module_name),
            "num": num,
            "config": safe_config,
            "_start": time.perf_counter(),
        }

    def log_value(self, **kwargs: Any) -> None:  # noqa: ANN401
        pass

    def log_metrics(self, metrics: dict[str, Any]) -> None:
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
        peak = rec.get("peak_vram_gb")
        if peak is not None and peak > self.peak_vram_gb_overall:
            self.peak_vram_gb_overall = peak
        # Fold per-step timings into the module record so a transformer's
        # trial exposes ``mean_step_s`` / ``p95_step_s`` next to its total
        # duration — the advisor's flat 1 s/step baseline can then be
        # recalibrated per device_class directly from row data.
        step_times = self._current_step_buffer or []
        step_summary = _summarize_step_times(step_times)
        if step_summary:
            rec["step_timings"] = step_summary
        self._current_step_buffer = None
        self.records.append(rec)
        self._current = None

    def end_run(self) -> None:
        pass

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        pass


def _attach_callbacks(pipeline: Pipeline, callbacks: list[OptimizerCallback]) -> None:
    """Instance-patch ``pipeline._fit`` to append ``callbacks`` to the callback chain."""
    original_fit = pipeline._fit  # noqa: SLF001

    def patched(context: Any) -> Any:  # noqa: ANN401
        context.callback_handler.callbacks.extend(callbacks)
        return original_fit(context)

    pipeline._fit = patched  # type: ignore[method-assign]  # noqa: SLF001


# === preset resolution & optional-extras skip ============================


# module_name → the ``autointent[extra]`` that must be installed for the
# module's __init__ to succeed. Sourced from ``require(...)`` calls in
# ``src/autointent/modules/scoring/`` — keep in sync.
_MODULE_TO_EXTRA: dict[str, str] = {
    "bert": "transformers",
    "catboost": "catboost",
    "lora": "peft",
    "ptuning": "peft",
    "description_llm": "openai",
}


def _missing_extras_for_config(cfg: dict[str, Any]) -> list[str]:
    """Return every optional extra that ``cfg``'s search_space needs but isn't installed.

    Uses the same ``_deps.require`` validator the modules use at runtime, so
    what the harness pre-checks matches what would fail inside ``fit``.
    A missing extra returns an ``ImportError``; anything else (e.g. unknown
    extra) propagates.
    """
    from autointent._deps import require  # type: ignore[import-not-found]

    needed: set[str] = set()
    for node in cfg.get("search_space") or []:
        for entry in node.get("search_space") or []:
            name = entry.get("module_name") if isinstance(entry, dict) else None
            extra = _MODULE_TO_EXTRA.get(name) if isinstance(name, str) else None
            if extra:
                needed.add(extra)

    missing: list[str] = []
    for extra in sorted(needed):
        try:
            require(extra)  # type: ignore[arg-type]
        except ImportError:
            missing.append(extra)
    return missing


def _load_pipeline_from_preset_ref(ref: str) -> tuple[Pipeline, str]:
    """Resolve ``ref`` as either a bundled preset name or a YAML file path.

    Returns ``(pipeline, display_name)`` where ``display_name`` is the bundled
    name for name refs or the file stem for path refs. Path refs let the
    harness exercise modules not in any bundled preset (LoRA / ptuning /
    dnnc / gcn / cross-encoder scorer) without polluting the shipped
    ``SearchSpacePreset`` literal.
    """
    if ref.endswith((".yaml", ".yml")):
        path = Path(ref).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Preset file not found: {path}")
        pipeline = Pipeline.from_optimization_config(path)
        return pipeline, path.stem
    return Pipeline.from_preset(ref), ref  # type: ignore[arg-type]


def _run_cli_smoke(
    preset_ref: str,
    stats: Any,  # noqa: ANN401
    budget_vram_gb: float | None,
) -> dict[str, Any]:
    """Invoke ``autointent-advisor inspect <preset> --json`` in-process.

    Fed the same stats (as placeholder args) and budget the direct-API path
    saw, so a divergence in ``is_feasible`` / predicted numbers points at the
    CLI wrapper or the JSON renderer, not at differing inputs.

    Returns a dict with:
      * ``payload`` — the parsed JSON body from the CLI (or ``None`` on crash)
      * ``rc`` — the CLI return code
      * ``error`` — traceback string when the CLI or JSON parse failed
      * ``divergence`` — dict of |cli - direct| deltas populated by the caller

    Runs in-process (no subprocess) so we don't pay the interpreter-startup
    cost on every preset — the review only asked for a wrapper smoke, not a
    full subprocess isolation test.
    """
    import contextlib
    import io as _io
    import traceback

    from autointent._advisor._cli import main as cli_main

    argv = [
        "inspect",
        preset_ref,
        "--n-samples",
        str(int(stats.n_samples)),
        "--n-classes",
        str(int(stats.n_classes)),
        "--avg-tokens",
        str(int(stats.avg_tokens)),
        "--task",
        "multilabel" if getattr(stats, "multilabel", False) else "multiclass",
        "--json",
    ]
    if budget_vram_gb is not None:
        argv += ["--budget-vram-gb", str(budget_vram_gb)]

    buf = _io.StringIO()
    result: dict[str, Any] = {"payload": None, "rc": None, "error": None}
    try:
        with contextlib.redirect_stdout(buf):
            result["rc"] = cli_main(argv)
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc(limit=3)
        return result

    raw = buf.getvalue().strip()
    if not raw:
        result["error"] = "CLI produced empty stdout"
        return result
    try:
        result["payload"] = json.loads(raw)
    except json.JSONDecodeError as e:
        result["error"] = f"CLI --json output not parseable: {e}"
    return result


def _load_config_from_preset_ref(ref: str) -> dict[str, Any]:
    """Load the raw preset config dict without instantiating a Pipeline.

    Used for pre-fit extras checks so a preset whose modules would need a
    missing extra (e.g. ``lora`` → ``peft``) never reaches ``from_preset``.
    """
    if ref.endswith((".yaml", ".yml")):
        import yaml

        with Path(ref).expanduser().open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    from autointent.utils import load_preset  # local import to avoid top-level cost

    return load_preset(ref)  # type: ignore[arg-type]


# === per-preset run ======================================================


def _override_trials(pipeline: Pipeline, max_trials: int | None, *, run_name: str | None = None) -> None:
    """Cap n_trials, force ``n_jobs=1`` (serial HPO to keep wall-time measurements clean
    and to prevent CPU oversubscription with sklearn's own ``n_jobs``), disable dumping.
    When ``run_name`` is set, tag ``LoggingConfig.run_name`` (used as the W&B group /
    dump-dir name).
    """
    updates: dict[str, Any] = {"n_jobs": 1}
    if max_trials is not None:
        updates["n_trials"] = max_trials
    pipeline.set_config(pipeline.hpo_config.model_copy(update=updates))
    logging_config = LoggingConfig(dump_modules=False, clear_ram=True, run_name=run_name)
    pipeline.set_config(logging_config)


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
    run_name: str | None,
    budget_vram_gb: float | None,
    clear_embedding_cache: bool,
) -> CalibrationRow:
    # ``preset`` may be a bundled name OR a path to a YAML file (the coverage
    # preset). Resolve early so we can pre-check extras against the raw config
    # before touching the Pipeline machinery.
    try:
        raw_cfg = _load_config_from_preset_ref(preset)
    except Exception as e:  # noqa: BLE001
        display_name = Path(preset).stem if preset.endswith((".yaml", ".yml")) else preset
        row = CalibrationRow(preset=display_name)
        row.error = f"load-preset failed: {e}"
        return row
    display_name = Path(preset).stem if preset.endswith((".yaml", ".yml")) else preset

    row = CalibrationRow(preset=display_name)
    row.cache_policy = "cold" if clear_embedding_cache else "warm"

    # Detect missing optional extras BEFORE the fit — otherwise the trial
    # would raise ImportError deep inside HPO, producing a fit-failed row
    # indistinguishable from a real bug.
    missing = _missing_extras_for_config(raw_cfg)
    if missing:
        row.skipped = True
        row.error = f"skipped: missing extras {sorted(missing)}"
        row.notes.append(
            "install with: uv pip install " + " ".join(f"'autointent[{e}]'" for e in sorted(missing))
        )
        return row

    if clear_embedding_cache:
        removed = _clear_embeddings_cache()
        logger.info("Cleared %d embedding cache files for cold-cache measurement", removed)

    # === predicted ======================================================
    try:
        pipeline, _ = _load_pipeline_from_preset_ref(preset)
    except Exception as e:  # noqa: BLE001
        row.error = f"from_preset failed: {e}"
        return row

    tagged_run_name = f"{display_name}_{run_name}" if run_name else None
    _override_trials(pipeline, max_trials, run_name=tagged_run_name)

    try:
        # Optionally override hardware.vram_gb to exercise severity paths without a small GPU.
        preflight_hw = hardware
        if budget_vram_gb is not None:
            from dataclasses import replace

            preflight_hw = replace(hardware, vram_gb=budget_vram_gb)
        report: PreflightReport = run_preflight(
            pipeline._build_advisor_config(),  # noqa: SLF001
            stats,
            preflight_hw,
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
    row.headroom = report.headroom.value
    row.is_feasible = report.is_feasible
    # Last writer wins per metric; the resource phase emits at most one finding
    # per metric so there is nothing to collapse in practice.
    row.severity_by_metric = {f.metric: f.severity.value for f in report.findings if f.metric}
    for driver in report.resource.drivers:
        model = driver.get("model")
        if model:
            row.models[f"{driver.get('node_type', '?')}/{driver.get('module', '?')}"] = str(model)
    row.low_confidence = report.low_confidence
    if report.low_confidence:
        row.notes.append("low-confidence (heuristic HF metadata fallback in use)")

    # CLI wrapper smoke — same preset, same stats, same budget. Any divergence
    # in ``is_feasible`` or the top-line predicted numbers means the CLI /
    # JSON renderer drifted from the direct API. Runs unconditionally so a
    # regression shows up on every calibration run.
    smoke = _run_cli_smoke(preset, stats, budget_vram_gb)
    if smoke["error"]:
        row.notes.append(f"cli-smoke FAILED: {smoke['error'].splitlines()[-1] if smoke['error'] else '?'}")
    elif smoke["payload"]:
        cli_pred = smoke["payload"].get("resource") or {}
        divergence: dict[str, float] = {}
        for cli_key, direct_val in (
            ("time_hours", report.resource.time_hours),
            ("ram_gb", report.resource.ram_gb),
            ("vram_gb", report.resource.vram_gb),
            ("disk_download_gb", report.resource.disk_download_gb),
        ):
            cli_val = cli_pred.get(cli_key)
            if cli_val is None or direct_val is None:
                continue
            delta = abs(float(cli_val) - float(direct_val))
            if delta > 1e-6:
                divergence[cli_key] = delta
        smoke["divergence"] = divergence
        cli_feasible = smoke["payload"].get("is_feasible")
        if cli_feasible is not None and cli_feasible != report.is_feasible:
            row.notes.append(
                f"cli-smoke VERDICT MISMATCH: cli.is_feasible={cli_feasible} vs direct={report.is_feasible}"
            )
        elif divergence:
            # ``autointent-advisor inspect`` has no n_trials flag, so under
            # --max-trials the CLI necessarily costs the preset's bundled
            # n_trials while the direct path costs the override. That is an
            # apples-to-oranges comparison, not a wrapper regression — the
            # historical "the two paths differ ~10x" reading of this field was
            # this artifact. Only time_hours scales with n_trials, so a drift
            # confined to that key under an override is expected.
            expected_trials_artifact = max_trials is not None and set(divergence) == {"time_hours"}
            smoke["divergence_expected"] = expected_trials_artifact
            if expected_trials_artifact:
                row.notes.append(
                    f"cli-smoke time differs (cli n_trials={_preset_n_trials(raw_cfg)} vs "
                    f"--max-trials {max_trials}); expected, not a wrapper regression"
                )
            else:
                row.notes.append(f"cli-smoke numeric drift on {sorted(divergence)} (see cli_smoke.divergence)")
    row.cli_smoke = smoke

    if skip_fit:
        return row

    # === actual =========================================================
    hf_cache = _hf_cache_dir()
    embed_cache = _embeddings_cache_dir()
    hf_before = _dir_size_gb(hf_cache)
    embed_before = _dir_size_gb(embed_cache)
    _reset_vram_peak()

    tracker = _ModuleTracker()
    callbacks: list[OptimizerCallback] = [tracker]
    if enable_wandb:
        try:
            from autointent._callbacks.wandb import WandbCallback

            callbacks.append(WandbCallback())
        except ImportError as e:
            row.notes.append(f"W&B requested but not available: {e}")
    _attach_callbacks(pipeline, callbacks)

    is_mps = hardware.accelerator == "mps"
    is_cuda = hardware.accelerator == "cuda"
    undo_step_patch = _patch_trainer_for_step_timing(tracker)
    start = time.perf_counter()
    try:
        with _PeakSampler(
            interval_s=poll_interval_ms / 1000.0, sample_mps=is_mps, sample_cuda=is_cuda,
        ) as sampler:
            pipeline.fit(dataset, preflight="off")
    except Exception as e:  # noqa: BLE001
        row.error = f"fit failed: {e}"
        row.modules = tracker.records  # keep whatever we collected
        return row
    finally:
        undo_step_patch()
    elapsed_s = time.perf_counter() - start

    hf_after = _dir_size_gb(hf_cache)
    embed_after = _dir_size_gb(embed_cache)
    actual_time_h = elapsed_s / 3600.0
    actual_ram_gb = sampler.peak_ram_gb
    # VRAM: take max of per-module tracker (inside brackets) and background
    # sampler (outside brackets, e.g. classic-preset embedder forward).
    # Fallback to a raw peak read only if both are zero.
    actual_vram_gb: float | None = None
    tracker_peak = tracker.peak_vram_gb_overall if tracker.peak_vram_gb_overall > 0 else None
    sampler_peak = sampler.peak_cuda_gb if sampler.peak_cuda_gb and sampler.peak_cuda_gb > 0 else None
    candidates = [x for x in (tracker_peak, sampler_peak) if x is not None]
    if candidates:
        actual_vram_gb = max(candidates)
    else:
        actual_vram_gb = _read_vram_peak_gb(hardware.accelerator)
    if actual_vram_gb is None and is_mps:
        actual_vram_gb = sampler.peak_mps_gb

    row.actual = {
        "time_h": actual_time_h,
        "ram_gb": actual_ram_gb,
        "vram_gb": actual_vram_gb,
        "disk_download_gb": max(0.0, hf_after - hf_before),
        "disk_embedding_cache_gb": max(0.0, embed_after - embed_before),
        # Per-signal breakdown; classic presets expect sampler > tracker.
        "vram_gb_tracker": tracker_peak,
        "vram_gb_sampler": sampler_peak,
    }
    row.modules = tracker.records
    if enable_wandb and not any("W&B requested but not available" in n for n in row.notes):
        row.notes.append("W&B reporter attached — inspect wandb.ai run group for per-step GPU/system metrics")
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
    _print_repeat_aggregates(rows)
    for row in rows:
        # Ratios are always computed at read time (see CalibrationRow.to_dict).
        ratios = row._ratios()  # noqa: SLF001
        cells = {
            "preset": row.preset,
            "pred_time": row.predicted.get("time_h"),
            "act_time": row.actual.get("time_h"),
            "r_time": ratios.get("time_h"),
            "pred_ram": row.predicted.get("ram_gb"),
            "act_ram": row.actual.get("ram_gb"),
            "r_ram": ratios.get("ram_gb"),
            "pred_vram": row.predicted.get("vram_gb"),
            "act_vram": row.actual.get("vram_gb"),
            "r_vram": ratios.get("vram_gb"),
        }
        print("  ".join(_fmt_cell(cells[key]).ljust(width) for key, _, width in _COLS))
        if row.error:
            marker = "~" if row.skipped else "!"
            print(f"    {marker} {row.error}")
        if row.low_confidence:
            print(f"    ! LOW-CONFIDENCE — advisor used heuristic HF metadata (exclude from prediction-accuracy stats)")
        if row.headroom is not None:
            verdict = "FEASIBLE" if row.is_feasible else "INFEASIBLE"
            by_metric = " ".join(f"{m}={s}" for m, s in sorted(row.severity_by_metric.items()))
            print(f"    · verdict={verdict} headroom={row.headroom} over={row.findings_over}  [{by_metric}]")
        if row.models:
            print(f"    · models: {', '.join(f'{k}={v}' for k, v in sorted(row.models.items()))}")
        print(f"    · cache-policy={row.cache_policy}")
        role_totals = _sum_time_by_role(row.modules)
        if role_totals:
            breakdown = "  ".join(f"{role}={total:.2f}s" for role, total in role_totals.items())
            print(f"    · time-by-role: {breakdown}")
        for note in row.notes:
            print(f"    * {note}")
        for mod in row.modules:
            duration = mod.get("duration_s")
            vram = mod.get("peak_vram_gb")
            role = mod.get("role", "?")
            duration_s = f"{duration:.2f}s" if duration is not None else "-"
            vram_s = f"{vram:.2f} GB" if vram is not None else "-"
            line = (
                f"      · [{role}] {mod.get('module', '?')}#{mod.get('num', '?')}  {duration_s}  vram={vram_s}"
            )
            step = mod.get("step_timings")
            if step:
                line += (
                    f"  n_steps={step['n_steps']} mean_step_s={step['mean_step_s']:.3f} "
                    f"p95_step_s={step['p95_step_s']:.3f}"
                )
            print(line)


def _sum_time_by_role(modules: list[dict[str, Any]]) -> dict[str, float]:
    """Fold per-module durations into ``{role: total_seconds}`` — used both for
    the printed breakdown and for the top-level ``time_by_role`` row field."""
    totals: dict[str, float] = {}
    for mod in modules:
        role = mod.get("role", "?")
        duration = mod.get("duration_s")
        if duration is None:
            continue
        totals[role] = totals.get(role, 0.0) + float(duration)
    return totals


def _print_repeat_aggregates(rows: list[CalibrationRow]) -> None:
    """When any (preset, dataset) has more than one repeat, print a mean±stdev
    block up-front so small ratio gaps are judgeable at a glance.

    Groups by ``(preset, first-note)`` — the dataset marker is inserted as the
    first note in main() so this key is stable across repeats.
    """
    import statistics

    groups: dict[tuple[str, str], list[CalibrationRow]] = {}
    for row in rows:
        dataset_note = row.notes[0] if row.notes else "dataset=?"
        groups.setdefault((row.preset, dataset_note), []).append(row)

    multi_groups = [(k, v) for k, v in groups.items() if len(v) > 1]
    if not multi_groups:
        return
    print(">>> repeats aggregation (mean ± stdev, successful runs only):")
    for (preset, dataset_note), group in multi_groups:
        # Skipped rows are expected — separate them from real failures so the
        # aggregate isn't polluted by "all repeats failed" when the actual
        # cause is a missing optional extra.
        skipped = [r for r in group if r.skipped]
        successful = [r for r in group if r.error is None]
        real_failures = len(group) - len(successful) - len(skipped)
        n = len(successful)
        if n == 0:
            reason = f"{real_failures} failed"
            if skipped:
                reason += f", {len(skipped)} skipped"
            print(f"  {preset}  [{dataset_note}]  {reason} (no successful repeats)")
            continue
        parts = [f"  {preset}  [{dataset_note}]  n={n}"]
        for metric in ("time_h", "ram_gb", "vram_gb"):
            values = [r.actual.get(metric) for r in successful if r.actual.get(metric) is not None]
            if not values:
                continue
            mean = statistics.fmean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            parts.append(f"{metric}={mean:.2f}±{stdev:.2f}")
        print("  " + "  ".join(parts))
    print()


def _apply_thread_cap() -> None:
    """Cap torch intra-op threads to the same value as OMP_NUM_THREADS.

    Env vars (OMP/MKL/OpenBLAS) MUST be set before Python starts to be effective —
    that's the bash wrapper's job. This function is belt-and-braces: torch reads
    OMP_NUM_THREADS on init, but ``set_num_threads`` also caps its C++ intra-op
    pool if a caller forgets the env var.
    """
    n = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if n <= 0:
        return
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass


def _guard_cuda_init(*, required: bool) -> None:
    """When ``required`` is True, fail fast if PyTorch can't initialize CUDA.

    Guards against the silent 'nvidia-smi shows 2 GPUs but torch runs on CPU'
    trap that happens when the CUDA driver is older than what the installed
    torch wheel was built against.
    """
    if not required:
        return
    try:
        import torch
    except ImportError:
        msg = "--require-cuda passed but torch isn't installed"
        raise SystemExit(msg) from None
    if not torch.cuda.is_available():
        # Get the underlying reason if we can — usually a warning on import time.
        msg = (
            "--require-cuda passed but torch.cuda.is_available() is False. "
            "Check `nvidia-smi` vs `python -c 'import torch; print(torch.version.cuda)'` — "
            "you likely need a torch wheel built against a matching CUDA runtime."
        )
        raise SystemExit(msg)


def _load_dataset(dataset_arg: str, parser: argparse.ArgumentParser) -> tuple[Dataset, str]:
    """Load one dataset from a local JSON path or an HF Hub repo id, returning
    ``(dataset, source_label)`` — the label mirrors what the calibrator writes
    to the report so different sources are distinguishable in aggregate output.
    """
    dataset_path = Path(dataset_arg)
    if dataset_path.is_file():
        logger.info("Loading dataset from local file %s", dataset_path)
        return Dataset.from_json(dataset_path), str(dataset_path)
    logger.info("Loading dataset from HF Hub: %s", dataset_arg)
    try:
        dataset = Dataset.from_hub(dataset_arg)
    except Exception as e:  # noqa: BLE001
        parser.error(f"Could not load '{dataset_arg}' as a local JSON file or as a Hub repo id: {e}")
    return dataset, f"hub:{dataset_arg}"


def _preset_n_trials(raw_cfg: dict[str, Any]) -> int | None:
    """``hpo_config.n_trials`` as written in the preset, before any override."""
    hpo = raw_cfg.get("hpo_config")
    return hpo.get("n_trials") if isinstance(hpo, dict) else None


def _release_accelerator_memory() -> float:
    """Drop cached accelerator memory between presets; return GB still allocated.

    Without this the sweep is not measuring what it thinks it is on a small
    GPU: a preset that OOMs leaves its model, optimizer state and HPO trial
    objects alive, so the *next* preset starts with several GB already gone and
    OOMs too — an AMPLE preset then gets recorded as a failure it would never
    hit on its own. A non-zero return value means references survived the
    collection and the remaining presets in this process are suspect.
    """
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()
    still_allocated = torch.cuda.memory_allocated() / _BYTES_PER_GB
    torch.cuda.reset_peak_memory_stats()
    return still_allocated


def _subsample_per_class(dataset: Dataset, cap: int) -> Dataset:
    """Cap each class in the train split to at most ``cap`` samples (first-N slice).

    Uses a deterministic first-N slice per class — reproducible across runs
    without seeding, and keeps class-ordering intuitive when inspecting the
    subset. Only rewrites the train split; validation/test are left as-is so
    the metric baselines remain comparable.
    """
    from autointent.custom_types import Split

    train_key = Split.TRAIN if Split.TRAIN in dataset else next(
        (k for k in dataset if str(k).startswith(str(Split.TRAIN))), None,
    )
    if train_key is None:
        return dataset
    train = dataset[train_key]
    label_feature = dataset.label_feature
    seen: dict[Any, int] = {}
    keep: list[int] = []
    for idx, row in enumerate(train):
        label = row[label_feature]
        # For multilabel, key on the tuple so a sample with a rare-class tag
        # still contributes toward that class's cap.
        key = tuple(label) if isinstance(label, list) else label
        count = seen.get(key, 0)
        if count < cap:
            keep.append(idx)
            seen[key] = count + 1
    dataset[train_key] = train.select(keep)
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _apply_thread_cap()
    _guard_cuda_init(required=args.require_cuda)

    presets = args.presets or list(BUNDLED_PRESETS)
    unknown = [
        p
        for p in presets
        if not p.endswith((".yaml", ".yml")) and p not in BUNDLED_PRESETS
    ]
    if unknown:
        parser.error(
            f"Unknown preset(s): {', '.join(unknown)}. Known: {', '.join(BUNDLED_PRESETS)}, "
            "or pass a path to a .yaml file (e.g. scripts/coverage_preset.yaml)."
        )
    for p in presets:
        if p.endswith((".yaml", ".yml")) and not Path(p).expanduser().exists():
            parser.error(f"Preset file not found: {p}")

    hardware = detect_hardware()
    logger.info(
        "Hardware: %s (%s) — %.1f GB VRAM, %.0f GB RAM, %.0f GB free disk",
        hardware.accelerator,
        hardware.device_name,
        hardware.vram_gb,
        hardware.ram_gb,
        hardware.free_disk_gb,
    )
    logger.info(
        "Thread caps: OMP=%s MKL=%s OPENBLAS=%s TOKENIZERS_PARALLELISM=%s",
        os.environ.get("OMP_NUM_THREADS", "<unset>"),
        os.environ.get("MKL_NUM_THREADS", "<unset>"),
        os.environ.get("OPENBLAS_NUM_THREADS", "<unset>"),
        os.environ.get("TOKENIZERS_PARALLELISM", "<unset>"),
    )

    rows: list[CalibrationRow] = []
    datasets_meta: list[dict[str, Any]] = []

    def _write_payload() -> None:
        """Serialize the current in-memory rows to ``args.output``. Called
        after each preset finishes so a mid-sweep crash / Broken pipe leaves
        a valid partial report behind rather than losing everything.
        """
        payload_now = {
            "hardware": {
                "accelerator": hardware.accelerator,
                "device_name": hardware.device_name,
                "vram_gb": hardware.vram_gb,
                "ram_gb": hardware.ram_gb,
                "free_disk_gb": hardware.free_disk_gb,
            },
            "datasets": datasets_meta,
            "max_trials_override": args.max_trials,
            "skip_fit": args.skip_fit,
            "cache_policy": "cold" if args.clear_embedding_cache else "warm",
            "budget_vram_gb_override": args.budget_vram_gb,
            "subsample_per_class": args.subsample_per_class,
            "thread_caps": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            },
            "in_progress": True,
            "rows": [r.to_dict() for r in rows],
        }
        # Atomic write: dump to a sibling file, then rename. Prevents readers
        # from seeing a half-written JSON if the run is killed mid-serialize.
        tmp = args.output.with_suffix(args.output.suffix + ".partial")
        tmp.write_text(json.dumps(payload_now, indent=2, default=str))
        tmp.replace(args.output)

    for dataset_arg in args.dataset:
        dataset, dataset_source = _load_dataset(dataset_arg, parser)
        if args.subsample_per_class is not None:
            dataset = _subsample_per_class(dataset, args.subsample_per_class)
            dataset_source += f"|subsample-per-class={args.subsample_per_class}"
        stats = stats_from_dataset_obj(dataset)
        datasets_meta.append(
            {
                "path": dataset_source,
                "n_samples": stats.n_samples,
                "n_classes": stats.n_classes,
                "avg_tokens": stats.avg_tokens,
                "multilabel": stats.multilabel,
            },
        )
        logger.info(
            "Dataset %s: n_samples=%d n_classes=%d avg_tokens=%.1f multilabel=%s",
            dataset_source,
            stats.n_samples,
            stats.n_classes,
            stats.avg_tokens,
            stats.multilabel,
        )
        for preset in presets:
            for repeat_idx in range(max(1, args.repeats)):
                header = f"=== {preset} @ {dataset_source}"
                if args.repeats > 1:
                    header += f" (repeat {repeat_idx + 1}/{args.repeats})"
                header += " ==="
                logger.info(header)
                row = _calibrate_one(
                    preset=preset,
                    dataset=dataset,
                    stats=stats,
                    hardware=hardware,
                    max_trials=args.max_trials,
                    skip_fit=args.skip_fit,
                    poll_interval_ms=args.poll_interval_ms,
                    enable_wandb=args.wandb,
                    run_name=(
                        f"{args.run_name}_r{repeat_idx}" if args.run_name and args.repeats > 1 else args.run_name
                    ),
                    budget_vram_gb=args.budget_vram_gb,
                    clear_embedding_cache=args.clear_embedding_cache,
                )
                row.repeat_idx = repeat_idx
                row.notes.insert(0, f"dataset={dataset_source}")
                leaked_gb = _release_accelerator_memory()
                if leaked_gb > _LEAK_WARN_GB:
                    row.notes.append(
                        f"accelerator memory still held after cleanup: {leaked_gb:.2f} GB — "
                        f"later presets in this sweep may report a contaminated OOM"
                    )
                    logger.warning(
                        "%s left %.2f GB of VRAM allocated after cleanup; "
                        "run presets in separate processes for trustworthy numbers",
                        preset,
                        leaked_gb,
                    )
                rows.append(row)
                _write_payload()

    payload = {
        "hardware": {
            "accelerator": hardware.accelerator,
            "device_name": hardware.device_name,
            "vram_gb": hardware.vram_gb,
            "ram_gb": hardware.ram_gb,
            "free_disk_gb": hardware.free_disk_gb,
        },
        "datasets": datasets_meta,
        "max_trials_override": args.max_trials,
        "skip_fit": args.skip_fit,
        "cache_policy": "cold" if args.clear_embedding_cache else "warm",
        "budget_vram_gb_override": args.budget_vram_gb,
        "subsample_per_class": args.subsample_per_class,
        "thread_caps": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
        "in_progress": False,
        "rows": [r.to_dict() for r in rows],
    }
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote report to %s", args.output)

    print()
    _print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
