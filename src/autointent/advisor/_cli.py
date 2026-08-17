"""Console-script entry point for the pre-flight advisor.

Two subcommands:

* ``inspect`` — show what a given preset / config will cost on this machine.
* ``recommend`` — pick the best-fitting bundled preset for this machine.

Both subcommands accept either a real ``--dataset`` (Hub id or local
csv/json/jsonl/parquet path loaded via ``datasets.load_dataset``) or
``--n-samples / --n-classes / --avg-tokens`` placeholders so the script is
useful before the user has built a dataset.

The CLI is a thin wrapper around :func:`autointent.advisor.inspect` and
:func:`autointent.advisor.recommend`; callers that don't need argparse can
import those helpers directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from autointent.advisor import inspect, recommend, stats_from_dataset

from ._render import render_json, render_recommendation, render_text
from ._report import DatasetStats

logger = logging.getLogger("autointent.advisor")


def _stats_from_args(args: argparse.Namespace) -> DatasetStats:
    multilabel = args.task == "multilabel"
    if args.dataset:
        return stats_from_dataset(args.dataset, multilabel=multilabel)
    return DatasetStats.placeholder(
        n_samples=args.n_samples,
        n_classes=args.n_classes,
        avg_tokens=args.avg_tokens,
        multilabel=multilabel,
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
    report = inspect(
        args.target,
        stats=_stats_from_args(args),
        budget_vram_gb=args.budget_vram_gb,
    )
    if args.json:
        sys.stdout.write(render_json(report))
    else:
        sys.stdout.write(render_text(report))
    sys.stdout.write("\n")
    return 0 if report.is_feasible else 1


def cmd_recommend(args: argparse.Namespace) -> int:
    result = recommend(
        stats=_stats_from_args(args),
        budget_vram_gb=args.budget_vram_gb,
        budget_time_h=args.budget_time_h,
    )
    if args.json:
        sys.stdout.write(json.dumps(result.to_dict(), indent=2, default=str))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_recommendation(result.results, result.chosen))
        sys.stdout.write("\n")
        if result.chosen:
            sys.stdout.write("\n")
            sys.stdout.write(render_text(dict(result.results)[result.chosen]))
            sys.stdout.write("\n")
    return 0 if result.chosen else 1


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
    return int(args.func(args))


if __name__ == "__main__":
    main()
