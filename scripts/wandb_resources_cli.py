"""CLI tool for processing wandb runs and computing system metrics."""

import argparse
import re
from collections.abc import Sequence
from typing import Any

from scipy.integrate import trapezoid

import wandb


def calculate_area(metrics: Sequence[float], timestamps: Sequence[float]) -> Any:
    """Calculate the area under the curve using the trapezoidal rule.

    Args:
        metrics (iterable): Array of metric values.
        timestamps (iterable): Array of timestamps corresponding to the metric values.

    Returns:
        float: The computed area under the curve.

    Raises:
        ValueError: If the lengths of metrics and timestamps do not match.
    """
    if len(metrics) != len(timestamps):
        error = "Metrics and timestamps dimensions do not match!"
        raise ValueError(error)
    return trapezoid(metrics, timestamps)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by replacing invalid characters with underscores.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: A sanitized filename containing only alphanumeric characters, underscores, or hyphens.
    """
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", filename)


def process_run(run: wandb.Api.Run) -> dict[str, dict[str, float]]:
    """Process a wandb run to extract system metrics and compute statistics.

    Args:
        run (Any): A wandb run object containing history data.

    Returns:
        dict: A dictionary mapping metric columns to their computed statistics,
            including max, min, avg, area, and median.
    """
    system_metrics = run.history(stream="system_metrics")
    results = {}
    for column in system_metrics.columns:
        if "system" not in column:
            continue
        metrics_values = system_metrics[column].dropna()
        if len(metrics_values) > 0:
            timestamps = system_metrics.index[: len(metrics_values)]
            results[column] = {
                "max": metrics_values.max(),
                "min": metrics_values.min(),
                "avg": metrics_values.mean(),
                "area": calculate_area(metrics_values, timestamps),
                "median": metrics_values.median(),
            }
    return results


def main() -> None:
    """CLI endpoint."""
    parser = argparse.ArgumentParser(description="Processing wandb resource data")
    parser.add_argument("--project", type=str, required=True, help="Wandb project name")
    parser.add_argument("--group", type=str, required=True, help="Wandb group name")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["max", "area", "avg"],
        help="Metrics to compute (e.g. min, max, area, avg, median)",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(args.project, filters={"group": args.group})

    if not runs:
        error = f"No runs found for group {args.group} in project {args.project}."
        raise ValueError(error)

    for run in runs:
        if "final_metrics" not in run.name:
            wandb.init(project=args.project, group=args.group, name=f"system_resources_{run.name}")
            results = process_run(run)

            for column, metrics in results.items():
                column_process = column.replace("/", "-")
                log_data = {}

                for m in args.metrics:
                    if m in metrics.keys():
                        log_data[f"system_resources/{column_process}_{m}"] = metrics[m]

                wandb.log(log_data)

            wandb.finish()


if __name__ == "__main__":
    main()
