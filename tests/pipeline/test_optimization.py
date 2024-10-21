import pathlib

import pytest

from autointent import Context
from autointent.pipeline import Pipeline


@pytest.mark.parametrize(
    ("dataset_type"),
    ["multiclass", "multilabel"],
)
def test_full_pipeline(setup_environment, load_clinc_subset, get_config, dataset_type):
    run_name, db_dir = setup_environment

    dataset = load_clinc_subset(dataset_type)

    context = Context(
        dataset=dataset,
        test_dataset=None,
        device="cpu",
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
    )

    # run optimization
    search_space_config = get_config(dataset_type)
    pipeline = Pipeline.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    logs_dir = pathlib.Path.cwd() / "tests_logs"
    pipeline.dump(logs_dir=str(logs_dir), run_name=run_name)
