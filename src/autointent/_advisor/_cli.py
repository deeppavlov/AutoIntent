"""Console-script entry point for the pre-flight advisor.

Two subcommands:

* ``inspect`` — show what a given preset / config will cost on this machine.
* ``recommend`` — pick the best-fitting bundled preset for this machine.

Both subcommands accept either a real ``--dataset`` (Hub id or local
csv/json/jsonl/parquet path loaded via ``datasets.load_dataset``) or
``--n-samples / --n-classes / --avg-tokens`` placeholders so the script is
useful before the user has built a dataset.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from datasets import ClassLabel, Sequence, load_dataset

from autointent.utils import load_preset

from ._estimates import run_preflight
from ._hardware import detect_hardware
from ._render import render_json, render_recommendation, render_text
from ._report import DatasetStats, Severity

if TYPE_CHECKING:
    from ._report import PreflightReport

logger = logging.getLogger("autointent.advisor")

_SAMPLE_LIMIT = 1000
_P95_PERCENTILE = 0.95

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


_UTTERANCE_COLS = ("utterance", "text", "sentence", "query", "input")
_LABEL_COLS = ("label", "labels", "intent", "target")
# Map file extension → datasets builder name. Anything else is treated as a Hub
# repo id or a directory and passed to load_dataset directly.
_FILE_BUILDERS = {".csv": "csv", ".tsv": "csv", ".json": "json", ".jsonl": "json", ".parquet": "parquet"}


def _stats_from_dataset(path: str, *, multilabel: bool) -> DatasetStats:
    """Best-effort: load via HF ``datasets.load_dataset``.

    Accepts a Hub repo id ('DeepPavlov/clinc150') or a local file path
    (.csv / .json / .jsonl / .parquet) / dataset directory. Falls back to a
    placeholder on any loader error so the advisor stays best-effort.
    """
    builder = _FILE_BUILDERS.get(Path(path).suffix.lower())
    try:
        ds = load_dataset(builder, data_files=path) if builder else load_dataset(path)
    except (OSError, ValueError, FileNotFoundError) as e:
        logger.warning("Failed to load dataset %s: %s", path, e)
        return DatasetStats.placeholder(multilabel=multilabel)

    train = ds["train"] if "train" in ds else next(iter(ds.values()), None)
    if train is None:
        return DatasetStats.placeholder(multilabel=multilabel)

    cols = train.column_names
    utt_col = next((c for c in _UTTERANCE_COLS if c in cols), cols[0] if cols else None)
    label_col = next((c for c in _LABEL_COLS if c in cols), None)

    detected_multilabel, n_classes = _label_shape(train, label_col, fallback_multilabel=multilabel)

    sample = train[:_SAMPLE_LIMIT] if len(train) > _SAMPLE_LIMIT else train[:]
    lengths = [len(str(s).split()) for s in (sample.get(utt_col, []) if utt_col else [])]
    avg_tokens = int(sum(lengths) / max(1, len(lengths))) if lengths else 32
    if lengths:
        sorted_lengths = sorted(lengths)
        idx = max(0, min(len(sorted_lengths) - 1, round((len(sorted_lengths) - 1) * _P95_PERCENTILE)))
        p95 = sorted_lengths[idx]
    else:
        p95 = avg_tokens * 2

    return DatasetStats(
        n_samples=len(train),
        n_classes=n_classes,
        avg_tokens=avg_tokens,
        p95_tokens=p95,
        multilabel=detected_multilabel,
        has_descriptions=None,
        rare_classes=_rare_classes(train, label_col, detected_multilabel, n_classes) if label_col else [],
        source=f"dataset:{path}",
    )


def _label_shape(train: Any, label_col: str | None, *, fallback_multilabel: bool) -> tuple[bool, int]:  # noqa: ANN401
    """Derive (multilabel, n_classes) from the HF feature schema, with a value-based fallback."""
    if label_col is None:
        return fallback_multilabel, 0
    feature = train.features.get(label_col)
    if isinstance(feature, Sequence):
        inner = feature.feature
        if isinstance(inner, ClassLabel):
            return True, inner.num_classes
        # Sequence of plain ints — n_classes = max label index + 1.
        max_idx = max((max(row) for row in train[label_col] if row), default=-1)
        return True, max_idx + 1
    if isinstance(feature, ClassLabel):
        return False, feature.num_classes
    # Plain int/string column. Detect multilabel from the first non-empty row, then count uniques.
    is_multi = len(train) > 0 and isinstance(train[0][label_col], (list, tuple))
    if is_multi:
        max_idx = max((max(row) for row in train[label_col] if row), default=-1)
        return True, max_idx + 1
    return False, len({label for label in train[label_col] if label is not None})


def _rare_classes(
    train: Any,  # noqa: ANN401
    label_col: str,
    multilabel: bool,
    n_classes: int,
    min_count: int = 3,
) -> list[str]:
    """Return labels with fewer than ``min_count`` samples in the train split.

    Used to surface the LogisticRegressionCV(cv=3) failure case before fit.
    Returns an empty list on any error so the advisor stays best-effort.
    """
    try:
        labels = train[label_col]
    except (KeyError, AttributeError, TypeError):
        return []
    counts: dict[str, int] = {}
    if multilabel:
        for row in labels:
            if not row:
                continue
            for i, v in enumerate(row):
                if v:
                    counts[str(i)] = counts.get(str(i), 0) + 1
        for i in range(n_classes):
            counts.setdefault(str(i), 0)
    else:
        for label in labels:
            counts[str(label)] = counts.get(str(label), 0) + 1
    return sorted(name for name, c in counts.items() if c < min_count)


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
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
