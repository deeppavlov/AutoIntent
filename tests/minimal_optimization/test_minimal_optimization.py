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

    cur_path = pathlib.Path(__file__).parent.resolve()
    context = Context(multiclass_intent_records=load_clinic_subset, multilabel_utterance_records=[],
                      test_utterance_records=[], device="cpu", mode=mode, multilabel_generation_config="",
                      regex_sampling=0, seed=0)

    # run optimization
    search_space_config = load_config(str(cur_path / "configs" / config_name), multilabel=mode != "multiclass")
    pipeline = Pipeline.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir=str(cur_path / "logs"), run_name=run_name)
