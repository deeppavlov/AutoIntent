import importlib.resources as ires
from typing import Literal

import pytest

from autointent import Context
from autointent.configs.optimization_cli import (
    DataConfig,
    LoggingConfig,
    OptimizationConfig,
    TaskConfig,
    VectorIndexConfig,
)
from autointent.pipeline import PipelineOptimizer
from autointent.pipeline.optimization.cli_endpoint import main as optimize_pipeline
from autointent.pipeline.optimization.utils import load_config

ConfigType = Literal["multiclass", "multilabel"]


@pytest.fixture
def get_config():
    def _get_config(config_type: ConfigType):
        config_path = ires.files("tests.assets.configs").joinpath(f"{config_type}.yaml")
        return load_config(str(config_path), multilabel=config_type == "multilabel")

    return _get_config


@pytest.mark.parametrize(
    "config_type",
    ["multiclass", "multilabel"],
)
def test_full_pipeline(setup_environment, get_config, dataset, config_type: ConfigType):
    db_dir, dump_dir, logs_dir = setup_environment

    context = Context(dataset=dataset, db_dir=db_dir(), dump_dir=dump_dir, force_multilabel=config_type == "multilabel")

    # run optimization
    search_space_config = get_config(config_type)
    pipeline = PipelineOptimizer.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir=logs_dir)


@pytest.mark.parametrize(
    "dataset_type",
    ["multiclass", "multilabel", "description"],
)
def test_optimization_pipeline_cli(dataset_type, setup_environment):
    db_dir, dump_dir, logs_dir = setup_environment
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
