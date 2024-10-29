import importlib.resources as ires

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


@pytest.mark.parametrize(
    "dataset_type",
    ["multiclass", "multilabel"],
)
def test_full_pipeline(setup_environment, load_clinc_subset, get_config, dataset_type):
    db_dir, dump_dir, logs_dir = setup_environment

    dataset = load_clinc_subset(dataset_type)

    context = Context(
        dataset=dataset,
        db_dir=db_dir(),
        dump_dir=dump_dir,
    )

    # run optimization
    search_space_config = get_config(dataset_type)
    pipeline = PipelineOptimizer.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir=logs_dir)


@pytest.mark.parametrize(
    "dataset_type",
    [
        "multiclass",
        "multilabel", "description"
    ],
)
def test_optimization_pipeline_cli(dataset_type, setup_environment):
    db_dir, dump_dir, logs_dir = setup_environment
    config = OptimizationConfig(
        data=DataConfig(train_path=ires.files("tests.assets.data").joinpath(f"clinc_subset_{dataset_type}.json")),
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
