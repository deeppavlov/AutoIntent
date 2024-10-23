import importlib.resources as ires
from typing import Literal

import pytest

from autointent import Context
from autointent.context.data_handler import Dataset
from autointent.pipeline.optimization.utils import load_config, load_data, setup_logging

DATASET_TYPE = Literal["multiclass", "multilabel"]


@pytest.fixture
def setup_environment() -> tuple[str, str]:
    setup_logging("DEBUG")
    logs_dir = ires.files("tests").joinpath("logs")
    db_dir = logs_dir / "db"
    dump_dir = logs_dir / "modules_dump"
    return db_dir, dump_dir, logs_dir


@pytest.fixture
def load_clinc_subset():
    def _load_data(dataset_type: DATASET_TYPE) -> Dataset:
        dataset_path = ires.files("tests.assets.data").joinpath(f"clinc_subset_{dataset_type}.json")
        return load_data(dataset_path)

    return _load_data


@pytest.fixture
def context(load_clinc_subset, setup_environment):
    db_dir, dump_dir, logs_dir = setup_environment

    def _get_context(dataset_type: DATASET_TYPE) -> Context:
        return Context(
            dataset=load_clinc_subset(dataset_type),
            test_dataset=None,
            device="cpu",
            multilabel_generation_config="",
            regex_sampling=0,
            seed=0,
            db_dir=db_dir,
            dump_dir=dump_dir,
        )

    return _get_context


@pytest.fixture
def get_config():
    def _get_config(dataset_type: DATASET_TYPE):
        config_path = ires.files("tests.assets.configs").joinpath(f"{dataset_type}.yaml")
        return load_config(str(config_path), multilabel=dataset_type == "multilabel")

    return _get_config
