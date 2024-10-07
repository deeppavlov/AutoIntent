import pathlib

import pytest

from autointent import Context
from autointent.pipeline import Pipeline


@pytest.mark.parametrize("mode, config_name", [
    ("multiclass", "multiclass.yaml"),
    ("multiclass_as_multilabel", "multilabel.yaml")
])
def test_full_pipeline(setup_environment, load_clinic_subset, mode, config_name):
    run_name, db_dir = setup_environment

    cur_path = pathlib.Path(__file__).parent.resolve()
    context = Context(
        multiclass_intent_records=load_clinic_subset,
        multilabel_utterance_records=[],
        test_utterance_records=[],
        device="cpu",
        mode=mode,  # type: ignore
        multilabel_generation_config="",
        db_dir=db_dir,
        regex_sampling=0,
        seed=0,
    )

    # run optimization
    pipeline = Pipeline(
        config_path=str(cur_path / "configs" / config_name),
        mode=mode,
    )
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir=str(cur_path / "logs"), run_name=run_name)