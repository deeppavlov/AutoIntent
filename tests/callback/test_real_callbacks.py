"""Exercise the concrete logging callbacks with mocked external services.

These drive the real callback classes; the heavy third-party clients are either
replaced with mocks (wandb, codecarbon) or used for real against a temp dir
(tensorboard). They run in the unit-tests CI job, which installs the wandb and
codecarbon extras and the tensorboardx test dependency.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from autointent._callbacks.emissions_tracker import EmissionsTrackerCallback
from autointent._callbacks.tensorboard import TensorBoardCallback
from autointent._callbacks.wandb import WandbCallback

if TYPE_CHECKING:
    from pathlib import Path


def test_tensorboard_callback_writes_events(tmp_path: Path) -> None:
    pytest.importorskip("tensorboardX")
    callback = TensorBoardCallback()
    callback.start_run("run", tmp_path, 1.0)
    callback.start_module("scoring", 0, {"k": "v", "n": 1})
    callback.log_value(loss=0.5, note="text")  # type: ignore[arg-type]  # source annotates **kwargs as dict
    callback.log_metrics({"accuracy": 0.9, "label": "scoring"})
    callback.end_module()
    callback.log_final_metrics({"f1": 0.8})
    callback.end_run()

    assert any(tmp_path.iterdir())  # event files were written


def test_wandb_callback_logs_through_client(tmp_path: Path) -> None:
    pytest.importorskip("wandb")
    callback = WandbCallback()
    callback.wandb = MagicMock()

    callback.start_run("run", tmp_path, 1.0)
    callback.start_module("scoring", 0, {"k": "v"})
    callback.log_value(loss=0.5)  # type: ignore[arg-type]  # source annotates **kwargs as dict
    callback.log_metrics({"accuracy": 0.9})
    callback.end_module()
    callback.log_final_metrics({"configs": {"a": 1}, "pipeline_metrics": {"f1": 0.8}})
    callback.end_run()

    assert callback.wandb.init.called
    assert callback.wandb.log.called
    assert callback.wandb.finish.called


def test_codecarbon_callback_merges_emissions(tmp_path: Path) -> None:
    pytest.importorskip("codecarbon")
    tracker_cls = MagicMock()
    tracker = tracker_cls.return_value
    tracker.stop_task.return_value.toJSON.return_value = json.dumps({"emissions": 0.5, "name": "ignored"})
    tracker.final_emissions_data.toJSON.return_value = json.dumps({"emissions": 0.7})

    callback = EmissionsTrackerCallback()
    callback.emission_tracker = tracker_cls

    callback.start_run("run", tmp_path, 1.0)
    callback.start_module("scoring", 0, {})
    per_module = callback.update_metrics({"accuracy": 0.9})
    final = callback.update_final_metrics({"f1": 0.8})

    assert per_module["accuracy"] == 0.9
    assert per_module["emissions/emissions"] == 0.5  # numeric kept, "name" dropped
    assert final["f1"] == 0.8
    assert final["emissions"] == {"emissions/emissions": 0.7}
