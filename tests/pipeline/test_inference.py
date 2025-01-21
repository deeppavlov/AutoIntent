import importlib.resources as ires
from typing import Literal

import pytest

from autointent import Pipeline
from autointent.configs import CrossEncoderConfig, EmbedderConfig, LoggingConfig, VectorIndexConfig
from autointent.utils import load_search_space
from tests.conftest import setup_environment

TaskType = Literal["multiclass", "multilabel", "description"]


def get_search_space_path(task_type: TaskType):
    return ires.files("tests.assets.configs").joinpath(f"{task_type}.yaml")


def get_search_space(task_type: TaskType):
    path = get_search_space_path(task_type)
    return load_search_space(path)


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_inference_config(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(VectorIndexConfig(save_db=True))
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32, device="cpu"))
    pipeline_optimizer.set_config(CrossEncoderConfig())

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    inference_config = context.optimization_info.get_inference_nodes_config()

    inference_pipeline = Pipeline.from_config(inference_config)
    utterances = ["123", "hello world"]
    prediction = inference_pipeline.predict(utterances)
    if task_type == "multilabel":
        assert prediction.shape == (2, len(dataset.intents))
    else:
        assert prediction.shape == (2,)

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
    pipeline.set_config(VectorIndexConfig(save_db=True))
    pipeline.set_config(EmbedderConfig(batch_size=16, max_length=32, device="cpu"))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline.fit(dataset)
    utterances = ["123", "hello world"]
    prediction = pipeline.predict(utterances)

    if task_type == "multilabel":
        assert prediction.shape == (2, len(dataset.intents))
    else:
        assert prediction.shape == (2,)

    rich_outputs = pipeline.predict_with_metadata(utterances)
    assert len(rich_outputs.predictions) == len(utterances)

    context.dump()
