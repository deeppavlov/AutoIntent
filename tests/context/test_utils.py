"""Unit tests for autointent.context._utils (NumpyEncoder, load_dataset)."""

from __future__ import annotations

import importlib.resources as ires
import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from autointent import Dataset
from autointent.context._utils import NumpyEncoder, load_dataset

if TYPE_CHECKING:
    from pathlib import Path


def test_numpy_encoder_serializes_numpy_types() -> None:
    payload = {"i": np.int64(3), "f": np.float64(1.5), "a": np.array([1, 2, 3])}
    decoded = json.loads(json.dumps(payload, cls=NumpyEncoder))
    assert decoded == {"i": 3, "f": 1.5, "a": [1, 2, 3]}


def test_numpy_encoder_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, cls=NumpyEncoder)


def test_load_dataset_from_local_json() -> None:
    path = cast("Path", ires.files("tests.assets.data").joinpath("clinc_subset.json"))
    assert isinstance(load_dataset(path), Dataset)
