import importlib.resources as ires
from typing import Literal
from uuid import uuid4

import pytest

from autointent import Context
from autointent.context.data_handler import Dataset
from autointent.pipeline.optimization.utils import load_config, load_data

DATASET_TYPE = Literal["multiclass", "multilabel"]


@pytest.fixture
def setup_environment() -> tuple[str, str]:
    logs_dir = ires.files("tests").joinpath("logs")

    def get_db_dir():
        return logs_dir / "db" / str(uuid4())

    dump_dir = logs_dir / "modules_dump"
    return get_db_dir, dump_dir, logs_dir


def load_clinc_subset(dataset_type):
    dataset_path = ires.files("tests.assets.data").joinpath(f"clinc_subset_{dataset_type}.json")
    return load_data(dataset_path)


@pytest.fixture
def context(setup_environment):
    db_dir, dump_dir, logs_dir = setup_environment

    def _get_context(dataset_type: DATASET_TYPE) -> Context:
        return Context(
            dataset=load_clinc_subset(dataset_type),
            test_dataset=None,
            device="cpu",
            multilabel_generation_config="",
            regex_sampling=0,
            seed=0,
            db_dir=db_dir(),
            dump_dir=dump_dir,
        )

    return _get_context


@pytest.fixture
def get_config():
    def _get_config(dataset_type: DATASET_TYPE):
        config_path = ires.files("tests.assets.configs").joinpath(f"{dataset_type}.yaml")
        return load_config(str(config_path), multilabel=dataset_type == "multilabel")

    return _get_config
