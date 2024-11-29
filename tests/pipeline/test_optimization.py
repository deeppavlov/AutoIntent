import importlib.resources as ires
import os
from pathlib import Path
from typing import Literal

import pytest

from autointent.configs._optimization_cli import (
    DataConfig,
    EmbedderConfig,
    LoggingConfig,
    OptimizationConfig,
    TaskConfig,
    VectorIndexConfig,
)
from autointent.pipeline.optimization import PipelineOptimizer
from autointent.pipeline.optimization._cli_endpoint import main as optimize_pipeline
from autointent.pipeline.optimization._utils import load_config
from tests.conftest import setup_environment

TaskType = Literal["multiclass", "multilabel", "description"]


def get_search_space_path(task_type: TaskType):
    return ires.files("tests.assets.configs").joinpath(f"{task_type}.yaml")


def get_search_space(task_type: TaskType):
    path = get_search_space_path(task_type)
    return load_config(str(path), multilabel=task_type == "multilabel")


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_no_context_optimization(dataset, task_type):
    db_dir, dump_dir, logs_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = PipelineOptimizer.from_dict(search_space)

    pipeline_optimizer.set_config(LoggingConfig(dirpath=Path(logs_dir).resolve(), dump_modules=False))
    pipeline_optimizer.set_config(VectorIndexConfig(db_dir=Path(db_dir).resolve(), device="cpu"))
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32))

    context = pipeline_optimizer.fit(dataset, force_multilabel=(task_type == "multilabel"))
    context.dump()


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_save_db(dataset, task_type):
    db_dir, dump_dir, logs_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = PipelineOptimizer.from_dict(search_space)

    pipeline_optimizer.set_config(LoggingConfig(dirpath=Path(logs_dir).resolve(), dump_modules=False))
    pipeline_optimizer.set_config(VectorIndexConfig(db_dir=Path(db_dir).resolve(), save_db=True, device="cpu"))
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32))

    context = pipeline_optimizer.fit(dataset, force_multilabel=(task_type == "multilabel"))
    context.dump()

    assert os.listdir(db_dir)


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_dump_modules(dataset, task_type):
    db_dir, dump_dir, logs_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = PipelineOptimizer.from_dict(search_space)

    pipeline_optimizer.set_config(LoggingConfig(dirpath=Path(logs_dir).resolve(), dump_dir=dump_dir, dump_modules=True))
    pipeline_optimizer.set_config(VectorIndexConfig(db_dir=Path(db_dir).resolve(), device="cpu"))
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32))

    context = pipeline_optimizer.fit(dataset, force_multilabel=(task_type == "multilabel"))
    context.dump()

    assert os.listdir(dump_dir)


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_optimization_pipeline_cli(task_type):
    db_dir, dump_dir, logs_dir = setup_environment()
    config = OptimizationConfig(
        data=DataConfig(
            train_path=ires.files("tests.assets.data").joinpath("clinc_subset.json"),
            force_multilabel=(task_type == "multilabel"),
        ),
        task=TaskConfig(
            search_space_path=get_search_space_path(task_type),
        ),
        vector_index=VectorIndexConfig(
            db_dir=db_dir,
            device="cpu",
        ),
        logs=LoggingConfig(
            dirpath=Path(logs_dir),
        ),
    )
    optimize_pipeline(config)
