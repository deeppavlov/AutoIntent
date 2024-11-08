import importlib.resources as ires
from uuid import uuid4

import pytest

from autointent.context.utils import load_data


def setup_environment() -> tuple[str, str]:
    logs_dir = ires.files("tests").joinpath("logs") / str(uuid4())
    db_dir = logs_dir / "db"
    dump_dir = logs_dir / "modules_dump"
    return db_dir, dump_dir, logs_dir


@pytest.fixture
def dataset_path():
    return ires.files("tests.assets.data").joinpath("clinc_subset.json")


@pytest.fixture
def dataset(dataset_path):
    return load_data(dataset_path)
