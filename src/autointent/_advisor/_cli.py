"""Console-script entry point for the pre-flight advisor.

Two subcommands:

* ``inspect`` — show what a given preset / config will cost on this machine.
* ``recommend`` — pick the best-fitting bundled preset for this machine.

Both subcommands accept either a real ``--dataset`` (path to load with
``Dataset.from_*`` constructors) or ``--n-samples / --n-classes / --avg-tokens``
placeholders so the script is useful before the user has built a dataset.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from autointent import Dataset
from autointent.utils import load_preset

from ._estimates import run_preflight
from ._hardware import detect_hardware
from ._render import render_json, render_recommendation, render_text
from ._report import DatasetStats, PreflightReport, Severity

logger = logging.getLogger("autointent.advisor")

BUNDLED_PRESETS = [
    "transformers-heavy",
    "transformers-light",
    "transformers-no-hpo",
    "nn-heavy",
    "nn-medium",
    "classic-heavy",
    "classic-medium",
    "classic-light",
    "zero-shot-encoders",
    "zero-shot-llm",
]

# rough quality tiering used by `recommend`
_QUALITY_TIER = {
    "transformers-heavy": 5,
    "nn-heavy": 4,
    "transformers-light": 4,
    "nn-medium": 3,
    "classic-heavy": 3,
    "transformers-no-hpo": 3,
    "classic-medium": 2,
    "classic-light": 1,
    "zero-shot-encoders": 2,
    "zero-shot-llm": 4,
}


def _load_config(target: str) -> tuple[dict[str, Any], str]:
    """Return (config_dict, friendly_name) for either a preset or a path."""
    path = Path(target)
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f), path.stem
    # treat as a bundled preset name
    return load_preset(target), target  # type: ignore[arg-type]


def _stats_from_args(args: argparse.Namespace) -> DatasetStats:
    if args.dataset:
        return _stats_from_dataset(args.dataset, multilabel=args.task == "multilabel")
    return DatasetStats.placeholder(
        n_samples=args.n_samples,
        n_classes=args.n_classes,
        avg_tokens=args.avg_tokens,
        multilabel=args.task == "multilabel",
    )


def _stats_from_dataset(path: str, *, multilabel: bool) -> DatasetStats:
    """Best-effort: load a dataset from disk via the existing Dataset constructor."""
    try:
        ds = Dataset.from_json(path) if path.endswith(".json") else Dataset.from_hub(path)
    except (OSError, ValueError) as e:
        logger.warning("Failed to load dataset %s: %s", path, e)
        return DatasetStats.placeholder(multilabel=multilabel)

    train = ds.get("train") or next(iter(ds.values()), None)
    if train is None:
        return DatasetStats.placeholder(multilabel=multilabel)

    utt_col = getattr(ds, "utterance_feature", "utterance")
    sample = train[:1000] if len(train) > 1000 else train[:]
    lengths = [len(str(s).split()) for s in sample.get(utt_col, [])]
    avg_tokens = int(sum(lengths) / max(1, len(lengths))) if lengths else 32
    p95 = sorted(lengths)[int(len(lengths) * 0.95)] if lengths else avg_tokens * 2

    return DatasetStats(
        n_samples=len(train),
        n_classes=getattr(ds, "n_classes", 0) or 0,
        avg_tokens=avg_tokens,
        p95_tokens=p95,
        multilabel=getattr(ds, "multilabel", multilabel),
        has_descriptions=getattr(ds, "has_descriptions", None),
        source=f"dataset:{path}",
    )


def _add_common_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", help="Path or hub id of a dataset; overrides placeholders.")
    p.add_argument("--n-samples", type=int, default=1_000, help="Placeholder training set size.")
    p.add_argument("--n-classes", type=int, default=10, help="Placeholder class count.")
    p.add_argument("--avg-tokens", type=int, default=32, help="Placeholder average token length.")
    p.add_argument(
        "--task",
        choices=("multiclass", "multilabel"),
        default="multiclass",
        help="Placeholder task type when --dataset isn't given.",
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    config, name = _load_config(args.target)
    hardware = detect_hardware(
        vram_budget_gb=args.budget_vram_gb,
    )
    stats = _stats_from_args(args)
    report = run_preflight(config, stats, hardware, preset_name=name)
    if args.json:
        sys.stdout.write(render_json(report))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(report))
        sys.stdout.write("\n")
    return 0 if report.is_feasible else 1


def cmd_recommend(args: argparse.Namespace) -> int:
    hardware = detect_hardware(vram_budget_gb=args.budget_vram_gb)
    stats = _stats_from_args(args)

    results: list[tuple[str, PreflightReport]] = []

    for preset in BUNDLED_PRESETS:
        try:
            cfg = load_preset(preset)  # type: ignore[arg-type]
        except (OSError, ValueError, KeyError) as e:
            logger.debug("Skipping preset %s: %s", preset, e)
            continue
        report = run_preflight(cfg, stats, hardware, preset_name=preset)
        if args.budget_time_h is not None and report.resource.time_hours > args.budget_time_h:
            report.add(
                "resource",
                Severity.OVER,
                f"Estimated time {report.resource.time_hours:.1f} h exceeds budget {args.budget_time_h} h.",
            )
        results.append((preset, report))

    feasible = [(name, r) for name, r in results if r.is_feasible]
    feasible.sort(key=lambda pair: (-_QUALITY_TIER.get(pair[0], 0), pair[1].resource.time_hours, pair[0]))
    chosen = feasible[0][0] if feasible else None

    if args.json:
        import json

        out = {
            "chosen": chosen,
            "results": [{"preset": name, "report": r.to_dict()} for name, r in results],
        }
        sys.stdout.write(json.dumps(out, indent=2, default=str))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_recommendation(results, chosen))
        sys.stdout.write("\n")
        if chosen:
            sys.stdout.write("\n")
            sys.stdout.write(render_text(dict(results)[chosen]))
            sys.stdout.write("\n")
    return 0 if chosen else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autointent-advisor",
        description="Pre-flight feasibility advisor for AutoIntent search-space optimization.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser(
        "inspect",
        help="Inspect a preset or OptimizationConfig and print a feasibility report.",
    )
    p_inspect.add_argument("target", help="Preset name (e.g. transformers-light) or path to a YAML config.")
    p_inspect.add_argument("--json", action="store_true", help="Emit a structured JSON report.")
    p_inspect.add_argument("--budget-vram-gb", type=float, default=None, help="Override detected VRAM budget.")
    _add_common_dataset_args(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_rec = sub.add_parser(
        "recommend",
        help="Detect hardware and recommend the best-fitting bundled preset.",
    )
    p_rec.add_argument("--json", action="store_true", help="Emit a structured JSON report.")
    p_rec.add_argument("--budget-vram-gb", type=float, default=None, help="Override detected VRAM budget.")
    p_rec.add_argument("--budget-time-h", type=float, default=None, help="Optional wall-time ceiling in hours.")
    _add_common_dataset_args(p_rec)
    p_rec.set_defaults(func=cmd_recommend)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
