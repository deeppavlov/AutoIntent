import os
from typing import Literal

import pytest

from autointent import Pipeline
from autointent.configs import (
    EmbedderConfig,
    LoggingConfig,
    VectorIndexConfig,
)
from tests.conftest import get_search_space, setup_environment

TaskType = Literal["multiclass", "multilabel", "description"]


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_no_context_optimization(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=False))
    pipeline_optimizer.set_config(VectorIndexConfig())
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32, device="cpu"))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    context.dump()


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_save_db(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=False))
    pipeline_optimizer.set_config(VectorIndexConfig(save_db=True))
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32, device="cpu"))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    context.dump()


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_dump_modules(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True))
    pipeline_optimizer.set_config(VectorIndexConfig())
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32, device="cpu"))

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    context = pipeline_optimizer.fit(dataset)
    context.dump()

    assert os.listdir(pipeline_optimizer.logging_config.dump_dir)
