"""Advisor exceptions raised across the package / Pipeline boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._report import Finding


class PreflightError(RuntimeError):
    """Raised when ``Pipeline.fit(preflight="strict")`` finds OVER-budget resources."""

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings
        lines = "\n".join(f"  [{f.phase}] {f.message}" for f in findings)
        msg = f"Preflight check failed with {len(findings)} OVER finding(s):\n{lines}"
        super().__init__(msg)
