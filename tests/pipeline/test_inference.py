import importlib.resources as ires
from pathlib import Path
from typing import Literal

import pytest

from autointent.configs.inference_cli import InferenceConfig
from autointent.configs.optimization_cli import (
    EmbedderConfig,
    LoggingConfig,
    VectorIndexConfig,
)
from autointent.pipeline.inference import InferencePipeline
from autointent.pipeline.inference.cli_endpoint import main as inference_pipeline
from autointent.pipeline.optimization import PipelineOptimizer
from autointent.pipeline.optimization.utils import load_config
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
def test_inference_config(dataset, task_type):
    db_dir, dump_dir, logs_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = PipelineOptimizer.from_dict_config(search_space)

    pipeline_optimizer.set_config(LoggingConfig(dirpath=Path(logs_dir).resolve(), dump_modules=True))
    pipeline_optimizer.set_config(VectorIndexConfig(db_dir=Path(db_dir).resolve(), device="cpu", save_db=True))
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32))

    context = pipeline_optimizer.optimize_from_dataset(dataset, force_multilabel=(task_type == "multilabel"))
    inference_config = context.optimization_info.get_inference_nodes_config()

    inference_pipeline = InferencePipeline.from_config(inference_config)
    prediction = inference_pipeline.predict(["123", "hello world"])
    if task_type == "multilabel":
        assert prediction.shape == (2, len(dataset.intents))
    else:
        assert prediction.shape == (2,)

    context.dump()


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_inference_context(dataset, task_type):
    db_dir, dump_dir, logs_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = PipelineOptimizer.from_dict_config(search_space)

    pipeline_optimizer.set_config(LoggingConfig(dirpath=Path(logs_dir).resolve(), dump_modules=True))
    pipeline_optimizer.set_config(VectorIndexConfig(db_dir=Path(db_dir).resolve(), device="cpu", save_db=True))
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32))

    context = pipeline_optimizer.optimize_from_dataset(dataset, force_multilabel=(task_type == "multilabel"))
    inference_pipeline = InferencePipeline.from_context(context)
    prediction = inference_pipeline.predict(["123", "hello world"])

    if task_type == "multilabel":
        assert prediction.shape == (2, len(dataset.intents))
    else:
        assert prediction.shape == (2,)

    context.dump()


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_inference_pipeline_cli(dataset, task_type):
    db_dir, dump_dir, logs_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = PipelineOptimizer.from_dict_config(search_space)

    pipeline_optimizer.set_config(LoggingConfig(dirpath=Path(logs_dir).resolve(), dump_dir=dump_dir, dump_modules=True))
    pipeline_optimizer.set_config(VectorIndexConfig(db_dir=Path(db_dir).resolve(), device="cuda", save_db=True))
    pipeline_optimizer.set_config(EmbedderConfig(batch_size=16, max_length=32))
    context = pipeline_optimizer.optimize_from_dataset(dataset, force_multilabel=(task_type == "multilabel"))

    context.dump()

    config = InferenceConfig(
        data_path=ires.files("tests.assets.data").joinpath("clinc_subset.json"),
        source_dir=dump_dir,
        output_path=dump_dir,
        log_level="CRITICAL",
    )
    inference_pipeline(config)
