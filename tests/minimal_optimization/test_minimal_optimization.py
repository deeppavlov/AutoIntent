import pathlib

import pytest

from autointent import Context
from autointent.context.data_handler.schemas import DatasetType
from autointent.pipeline import Pipeline
from autointent.pipeline.optimization.utils import load_config


@pytest.mark.parametrize(
    ("dataset_filename", "config_name"),
    [("clinc_subset_multiclass.json", "multiclass.yaml"), ("clinc_subset_multilabel.json", "multilabel.yaml")],
)
def test_full_pipeline(setup_environment, load_clinc_subset, dataset_filename, config_name):
    run_name, db_dir = setup_environment

    cur_path = pathlib.Path(__file__).parent.resolve()

    dataset = load_clinc_subset(
        pathlib.Path("tests/minimal_optimization/data/") / dataset_filename,
    )

    context = Context(
        dataset=dataset,
        test_dataset=None,
        device="cpu",
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
    )

    # run optimization
    search_space_config = load_config(
        str(cur_path / "configs" / config_name),
        dataset.type == DatasetType.multilabel,
    )
    pipeline = Pipeline.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir=str(cur_path / "logs"), run_name=run_name)
