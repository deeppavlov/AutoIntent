import importlib.resources as ires

import pytest

from autointent import Context
from autointent.pipeline import PipelineOptimizer
from autointent.pipeline.optimization.cli_endpoint import OptimizationConfig
from autointent.pipeline.optimization.cli_endpoint import main as optimize_pipeline


@pytest.mark.parametrize(
    "dataset_type",
    ["multiclass", "multilabel"],
)
def test_full_pipeline(setup_environment, load_clinc_subset, get_config, dataset_type, dump_dir, logs_dir):
    run_name, db_dir = setup_environment

    dataset = load_clinc_subset(dataset_type)

    context = Context(
        dataset=dataset,
        test_dataset=None,
        device="cpu",
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
        db_dir=db_dir,
        dump_dir=dump_dir,
    )

    # run optimization
    search_space_config = get_config(dataset_type)
    pipeline = PipelineOptimizer.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir=logs_dir)


@pytest.mark.parametrize(
    ("dataset_type"),
    [
        "multiclass",
        "multilabel",
    ],
)
def test_optimization_pipeline_cli(dataset_type, logs_dir):
    config = OptimizationConfig(
        search_space_path=ires.files("tests.assets.configs").joinpath(f"{dataset_type}.yaml"),
        dataset_path=ires.files("tests.assets.data").joinpath(f"clinc_subset_{dataset_type}.json"),
        device="cpu",
        logs_dir=logs_dir,
    )
    optimize_pipeline(config)
