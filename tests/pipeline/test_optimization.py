import importlib.resources as ires
from typing import Literal

import pytest

from autointent.configs.optimization_cli import (
    DataConfig,
    LoggingConfig,
    OptimizationConfig,
    TaskConfig,
    VectorIndexConfig,
)
from autointent.pipeline.optimization.cli_endpoint import main as optimize_pipeline
from autointent.pipeline.optimization.utils import load_config
from tests.conftest import setup_environment

ConfigType = Literal["multiclass", "multilabel"]


@pytest.fixture
def get_config():
    def _get_config(config_type: ConfigType):
        config_path = ires.files("tests.assets.configs").joinpath(f"{config_type}.yaml")
        return load_config(str(config_path), multilabel=config_type == "multilabel")

    return _get_config


@pytest.mark.parametrize(
    "dataset_type",
    ["multiclass", "multilabel", "description"],
)
def test_optimization_pipeline_cli(dataset_type):
    db_dir, dump_dir, logs_dir = setup_environment()
    config = OptimizationConfig(
        data=DataConfig(
            train_path=ires.files("tests.assets.data").joinpath("clinc_subset.json"),
            force_multilabel=(dataset_type == "multilabel"),
        ),
        task=TaskConfig(
            search_space_path=ires.files("tests.assets.configs").joinpath(f"{dataset_type}.yaml"),
        ),
        vector_index=VectorIndexConfig(
            device="cpu",
        ),
        logs=LoggingConfig(
            dirpath=logs_dir,
        ),
    )
    optimize_pipeline(config)
