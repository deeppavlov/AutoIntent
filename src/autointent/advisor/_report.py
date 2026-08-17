"""Dataclasses for the pre-flight advisor's structured report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class Severity(str, Enum):
    """How much headroom a finding leaves against the detected budget.

    * ``AMPLE`` — comfortably within budget; informational only.
    * ``TIGHT`` — expected to fit, but with little margin; the estimate is
      heuristic, so treat this as "may not fit".
    * ``OVER`` — the budget is expected to be exceeded.

    A single ``OVER`` finding makes the whole report infeasible
    (:attr:`PreflightReport.is_feasible` is ``False``) and is what
    ``Pipeline.fit(preflight="strict")`` raises on.
    """

    AMPLE = "ample"
    TIGHT = "tight"
    OVER = "over"


Phase = Literal["resource", "data", "config"]


@dataclass(frozen=True)
class Finding:
    """A single advisor finding rendered as one line in the summary."""

    phase: Phase
    severity: Severity
    message: str
    metric: str | None = None


@dataclass
class ResourceEstimate:
    """Aggregated resource numbers across the search space."""

    disk_download_gb: float = 0.0
    disk_cached_gb: float = 0.0
    disk_dump_gb: float = 0.0
    disk_embedding_cache_gb: float = 0.0
    ram_gb: float = 0.0
    vram_gb: float = 0.0
    time_hours: float = 0.0
    parallel_factor: int = 1
    drivers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_disk_gb(self) -> float:
        return self.disk_download_gb + self.disk_dump_gb + self.disk_embedding_cache_gb


@dataclass
class DatasetStats:
    """Minimal stats the advisor needs about the user's dataset.

    Built either from a real ``Dataset`` or from CLI placeholder flags.
    """

    n_samples: int
    n_classes: int
    avg_tokens: int
    p95_tokens: int | None = None
    multilabel: bool = False
    has_descriptions: bool | None = None
    # Per-class train-split sample counts; empty when no real dataset was provided.
    class_counts: dict[str, int] = field(default_factory=dict)
    source: str = "placeholder"

    @classmethod
    def placeholder(
        cls,
        n_samples: int = 1_000,
        n_classes: int = 10,
        avg_tokens: int = 32,
        multilabel: bool = False,
    ) -> DatasetStats:
        """Build stats for a hypothetical dataset, for sizing a search space without data.

        ``p95_tokens`` is derived as ``avg_tokens * 2.5`` and ``class_counts`` is
        left empty. Use :func:`~autointent.advisor.dataset_stats` instead when a
        real ``Dataset`` is available.

        Args:
            n_samples: number of training utterances to assume.
            n_classes: number of intent classes to assume.
            avg_tokens: average utterance length in whitespace-separated tokens.
            multilabel: whether to assume a multilabel task.

        Returns:
            Stats with ``source="placeholder"``.
        """
        return cls(
            n_samples=n_samples,
            n_classes=n_classes,
            avg_tokens=avg_tokens,
            p95_tokens=int(avg_tokens * 2.5),
            multilabel=multilabel,
        )


@dataclass
class PreflightReport:
    """One report covering all three phases."""

    findings: list[Finding] = field(default_factory=list)
    resource: ResourceEstimate = field(default_factory=ResourceEstimate)
    hardware: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    preset_name: str | None = None
    low_confidence: bool = False
    notes: list[str] = field(default_factory=list)

    def add(self, phase: Phase, severity: Severity, message: str, metric: str | None = None) -> None:
        """Append a :class:`Finding` to this report.

        Args:
            phase: which check produced it — ``"resource"``, ``"data"`` or ``"config"``.
            severity: headroom level; a single ``OVER`` makes the report infeasible.
            message: one-line human-readable explanation, shown as-is in reports.
            metric: short budget name the finding is about (``"vram"``, ``"ram"``,
                ``"disk"``, ``"time"``), or ``None`` for findings not tied to a budget.
        """
        self.findings.append(Finding(phase=phase, severity=severity, message=message, metric=metric))

    @property
    def headroom(self) -> Severity:
        """Worst headroom level across all findings — the column shown in CLI reports."""
        order = {Severity.AMPLE: 0, Severity.TIGHT: 1, Severity.OVER: 2}
        if not self.findings:
            return Severity.AMPLE
        return max((f.severity for f in self.findings), key=lambda s: order[s])

    @property
    def is_feasible(self) -> bool:
        """Whether the run is expected to fit: True unless some finding is OVER."""
        return self.headroom != Severity.OVER

    def to_dict(self) -> dict[str, Any]:
        """Render the report as JSON-serializable data.

        Severities become their string values, and the derived ``headroom`` and
        ``is_feasible`` properties are included as keys. This is what the CLI's
        ``--json`` output emits.

        Returns:
            A plain dict of the report, its findings, resource estimate,
            hardware and dataset summaries.
        """
        d = asdict(self)
        d["findings"] = [{**asdict(f), "severity": f.severity.value} for f in self.findings]
        d["headroom"] = self.headroom.value
        d["is_feasible"] = self.is_feasible
        return d


@dataclass
class RecommendationResult:
    """Output of the recommend workflow: ranked per-preset reports plus the pick.

    ``chosen`` is the best feasible preset name, or ``None`` if none fit.
    ``results`` is the full per-preset report list in evaluation order.
    """

    chosen: str | None
    results: list[tuple[str, PreflightReport]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "results": [{"preset": name, "report": r.to_dict()} for name, r in self.results],
        }
