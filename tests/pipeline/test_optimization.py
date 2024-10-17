import importlib.resources as ires

import pytest

from autointent import Context
from autointent.configs.optimization_cli import ClassificationMode
from autointent.pipeline import PipelineOptimizer
from autointent.pipeline.optimization.cli_endpoint import OptimizationConfig
from autointent.pipeline.optimization.cli_endpoint import main as optimize_pipeline
from autointent.pipeline.optimization.utils import load_config


@pytest.mark.parametrize(
    ("mode", "config_name"), [("multiclass", "multiclass.yaml"), ("multiclass_as_multilabel", "multilabel.yaml")]
)
def test_optimization_pipeline_python_api(setup_environment, mode, load_clinic_subset, config_name, logs_dir, dump_dir):
    run_name, db_dir = setup_environment

    context = Context(
        multiclass_intent_records=load_clinic_subset,
        multilabel_utterance_records=[],
        test_utterance_records=[],
        device="cpu",
        mode=mode,
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
        db_dir=db_dir,
        dump_dir=dump_dir,
    )

    # run optimization
    config_path = ires.files("tests.assets.configs").joinpath(config_name)
    search_space_config = load_config(str(config_path), multilabel=mode != "multiclass")
    pipeline = PipelineOptimizer.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir=logs_dir)


@pytest.mark.parametrize(
    ("mode", "config_name"),
    [
        (ClassificationMode.multiclass, "multiclass.yaml"),
        (ClassificationMode.multiclass_as_multilabel, "multilabel.yaml"),
    ],
)
def test_optimization_pipeline_cli(mode, config_name, logs_dir):
    config = OptimizationConfig(
        search_space_path=ires.files("tests.assets.configs").joinpath(config_name),
        multiclass_path=ires.files("tests.assets.data").joinpath("clinc_subset.json"),
        device="cpu",
        mode=mode,
        logs_dir=logs_dir,
    )
    optimize_pipeline(config)
