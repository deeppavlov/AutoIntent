"""Tests for OptimizationInfo bookkeeping around module dumps."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from autointent.context.optimization_info import OptimizationInfo, ScorerArtifact

if TYPE_CHECKING:
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

    def _fake_dump(dump_dir: str) -> None:
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        (dump_path / "marker.json").write_text("{}", encoding="utf-8")

    module.dump.side_effect = _fake_dump

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


def test_new_best_keeps_previous_dump_when_new_dump_not_produced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`Dumper.dump()` swallows failures (raise_errors=False): a new-best trial's dump can
    silently produce nothing. The previous best dump must survive in that case — otherwise
    the pipeline is left referencing a best trial with no artifact at all."""
    info = OptimizationInfo()

    first_dump = tmp_path / "trial0"
    first_module = MagicMock()
    first_module.get_assets.return_value = ScorerArtifact()
    first_module.dump.side_effect = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
    _log_trial(info, metric_value=0.5, dump_dir=str(first_dump), module=first_module)
    assert first_dump.exists()

    second_dump = tmp_path / "trial1"
    second_module = MagicMock()  # dump() is a no-op by default: writes nothing
    second_module.get_assets.return_value = ScorerArtifact()

    with caplog.at_level(logging.WARNING):
        _log_trial(info, metric_value=0.9, dump_dir=str(second_dump), module=second_module)

    assert first_dump.exists()  # kept: the new best trial's dump was not actually produced
    assert "keeping previous best dump" in caplog.text
