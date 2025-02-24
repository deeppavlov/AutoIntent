import pytest

from autointent import Pipeline
from autointent.configs import LoggingConfig
from tests.conftest import get_search_space, setup_environment


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_inference_config(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    inference_config = context.optimization_info.get_inference_nodes_config()

    inference_pipeline = Pipeline.from_config(inference_config)
    utterances = ["123", "hello world"]
    prediction = inference_pipeline.predict(utterances)
    assert len(prediction) == 2

    rich_outputs = inference_pipeline.predict_with_metadata(utterances)
    assert len(rich_outputs.predictions) == len(utterances)

    context.dump()


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_inference_context(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline = Pipeline.from_search_space(search_space)

    pipeline.set_config(LoggingConfig(project_dir=project_dir, dump_modules=False, clear_ram=False))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline.fit(dataset)
    utterances = ["123", "hello world"]
    prediction = pipeline.predict(utterances)

    assert len(prediction) == 2

    rich_outputs = pipeline.predict_with_metadata(utterances)
    assert len(rich_outputs.predictions) == len(utterances)

    context.dump()
