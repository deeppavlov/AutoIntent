import importlib.resources as ires
from pathlib import Path
from uuid import uuid4

import pytest

from autointent import Dataset


def setup_environment() -> tuple[Path, Path, Path]:
    logs_dir = ires.files("tests").joinpath("logs") / str(uuid4())
    dump_dir = logs_dir / "modules_dump"
    return dump_dir, logs_dir


def get_dataset_path():
    return ires.files("tests.assets.data").joinpath("clinc_subset.json")


@pytest.fixture
def dataset():
    return Dataset.from_json(get_dataset_path())
