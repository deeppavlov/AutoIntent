import pytest

from autointent import Pipeline
from autointent.configs import EmbedderConfig, LoggingConfig
from autointent.custom_types import NodeType
from tests.conftest import get_search_space, setup_environment


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_inference_from_config(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    logging_config = LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True)
    pipeline_optimizer.set_config(logging_config)

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    context.dump()

    inference_pipeline = Pipeline.load(logging_config.dirpath)
    utterances = ["123", "hello world"]
    prediction = inference_pipeline.predict(utterances)
    assert len(prediction) == 2

    rich_outputs = inference_pipeline.predict_with_metadata(utterances)
    assert len(rich_outputs.predictions) == len(utterances)


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_inference_on_the_fly(dataset, task_type):
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


def test_load_with_overrided_params(dataset):
    project_dir = setup_environment()
    search_space = get_search_space("light")

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    logging_config = LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True)
    pipeline_optimizer.set_config(logging_config)

    context = pipeline_optimizer.fit(dataset)
    context.dump()

    inference_pipeline = Pipeline.load(logging_config.dirpath, embedder_config=EmbedderConfig(max_length=8))
    utterances = ["123", "hello world"]
    prediction = inference_pipeline.predict(utterances)
    assert len(prediction) == 2

    rich_outputs = inference_pipeline.predict_with_metadata(utterances)
    assert len(rich_outputs.predictions) == len(utterances)

    assert inference_pipeline.nodes[NodeType.scoring].module._embedder.max_length == 8


# TODO Pipeline.dump()
