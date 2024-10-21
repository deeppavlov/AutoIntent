import importlib.resources as ires
import pathlib
from typing import Literal
from uuid import uuid4

import pytest

from autointent import Context
from autointent.context.data_handler import Dataset
from autointent.pipeline.optimization.utils import get_db_dir, get_run_name, load_config, load_data, setup_logging

DATASET_TYPE = Literal["multiclass", "multilabel"]


@pytest.fixture
def setup_environment() -> tuple[str, str]:
    setup_logging("DEBUG")
    uuid = uuid4()
    run_name = get_run_name(str(uuid))
    db_dir = get_db_dir("", run_name)
    return run_name, db_dir


@pytest.fixture
def load_clinc_subset():
    def _load_data(dataset_type: DATASET_TYPE) -> Dataset:
        dataset_path = ires.files("tests.assets.data").joinpath(f"clinc_subset_{dataset_type}.json")
        return load_data(dataset_path, multilabel=False)

    return _load_data


@pytest.fixture
def context(load_clinc_subset, dump_dir, setup_environment):
    run_name, db_dir = setup_environment
    def _get_context(dataset_type: DATASET_TYPE) -> Context:
        return Context(
            dataset=load_clinc_subset(dataset_type),
            test_dataset=None,
            device="cpu",
            multilabel_generation_config="",
            regex_sampling=0,
            seed=0,
            db_dir=db_dir,
            dump_dir=dump_dir
        )

    return _get_context


@pytest.fixture
def get_config():
    def _get_config(dataset_type: DATASET_TYPE):
        config_path = ires.files("tests.assets.configs").joinpath(f"{dataset_type}.yaml")
        return load_config(str(config_path), multilabel=dataset_type == "multilabel")
    return _get_config

@pytest.fixture
def logs_dir() -> pathlib.Path:
    return pathlib.Path.cwd() / "tests_logs"


@pytest.fixture
def dump_dir(logs_dir) -> str:
    return str(logs_dir / "module_dumps")
