import importlib.resources as ires
from uuid import uuid4

import pytest

from autointent.pipeline.optimization.utils import load_data


@pytest.fixture
def setup_environment() -> tuple[str, str]:
    logs_dir = ires.files("tests").joinpath("logs")

    def get_db_dir():
        return logs_dir / "db" / str(uuid4())

    dump_dir = logs_dir / "modules_dump"
    return get_db_dir, dump_dir, logs_dir


@pytest.fixture
def dataset():
    dataset_path = ires.files("tests.assets.data").joinpath("clinc_subset.json")
    return load_data(dataset_path)
