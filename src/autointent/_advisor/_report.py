"""Dataclasses for the pre-flight advisor's structured report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class Severity(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


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
    ram_gb: float = 0.0
    vram_gb: float = 0.0
    time_hours: float = 0.0
    parallel_factor: int = 1
    drivers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_disk_gb(self) -> float:
        return self.disk_download_gb + self.disk_dump_gb


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
    rare_classes: list[str] = field(default_factory=list)
    source: str = "placeholder"

    @classmethod
    def placeholder(
        cls,
        n_samples: int = 1_000,
        n_classes: int = 10,
        avg_tokens: int = 32,
        multilabel: bool = False,
    ) -> DatasetStats:
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
        self.findings.append(Finding(phase=phase, severity=severity, message=message, metric=metric))

    @property
    def worst_severity(self) -> Severity:
        order = {Severity.GREEN: 0, Severity.YELLOW: 1, Severity.RED: 2}
        if not self.findings:
            return Severity.GREEN
        return max((f.severity for f in self.findings), key=lambda s: order[s])

    @property
    def is_feasible(self) -> bool:
        return self.worst_severity != Severity.RED

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["findings"] = [{**asdict(f), "severity": f.severity.value} for f in self.findings]
        d["worst_severity"] = self.worst_severity.value
        d["is_feasible"] = self.is_feasible
        return d
