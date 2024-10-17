import importlib.resources as ires
import pathlib

import pytest

from autointent import Context
from autointent.pipeline import Pipeline
from autointent.pipeline.optimization.utils import load_config


@pytest.mark.parametrize(
    ("mode", "config_name"), [("multiclass", "multiclass.yaml"), ("multiclass_as_multilabel", "multilabel.yaml")]
)
def test_full_pipeline(setup_environment, load_clinic_subset, mode, config_name):
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
    )

    # run optimization
    config_path = ires.files("tests.assets.configs").joinpath(config_name)
    search_space_config = load_config(str(config_path), multilabel=mode != "multiclass")

    pipeline = Pipeline.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    logs_dir = pathlib.Path.cwd() / "tests_logs"
    pipeline.dump(logs_dir=str(logs_dir), run_name=run_name)
