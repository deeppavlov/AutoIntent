"""Tests for OptimizationInfo bookkeeping around module dumps."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from autointent.context.optimization_info import OptimizationInfo, ScorerArtifact

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _log_trial(info: OptimizationInfo, metric_value: float, dump_dir: str, module: Any) -> None:
    info.log_module_optimization(
        node_type="scoring",
        module_name="knn",
        module_params={},
        metric_value=metric_value,
        metric_name="accuracy",
        metrics={"accuracy": metric_value},
        module_dump_dir=dump_dir,
        module=module,
    )


def test_new_best_removes_previous_dump_remote_aware(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Replacing the previous best must go through remove_module_dump (cluster-aware),
    not a bare rmtree — a remote_manifest.json inside the old dump must be processed."""
    info = OptimizationInfo()
    module = MagicMock()
    module.get_assets.return_value = ScorerArtifact()

    first_dump = tmp_path / "trial0"
    first_dump.mkdir()
    (first_dump / "remote_manifest.json").write_text(
        json.dumps({"engine": "some-future-engine", "index": "x", "dump_id": "y"}),
        encoding="utf-8",
    )
    _log_trial(info, metric_value=0.5, dump_dir=str(first_dump), module=module)

    second_dump = tmp_path / "trial1"
    second_dump.mkdir()
    with caplog.at_level(logging.WARNING):
        _log_trial(info, metric_value=0.9, dump_dir=str(second_dump), module=module)

    assert not first_dump.exists()
    assert "some-future-engine" in caplog.text  # proves remove_module_dump ran, not bare rmtree
