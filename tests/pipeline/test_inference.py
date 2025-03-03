import pytest

from autointent import Pipeline
from autointent.configs import EmbedderConfig, LoggingConfig
from autointent.custom_types import NodeType
from tests.conftest import get_search_space, setup_environment


@pytest.mark.parametrize(
    "task_type",
    ["regex", "multiclass", "multilabel", "description"],
)
def test_inference_from_config(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    logging_config = LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True)
    pipeline_optimizer.set_config(logging_config)

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    # case 1: inference from file system
    context = pipeline_optimizer.fit(dataset)
    context.dump()

    inference_pipeline = Pipeline.load(logging_config.dirpath)
    utterances = ["123", "hello world"]
    prediction = inference_pipeline.predict(utterances)
    assert len(prediction) == 2

    # case 2: rich inference from file system
    rich_outputs = inference_pipeline.predict_with_metadata(utterances)
    assert len(rich_outputs.predictions) == len(utterances)

    if task_type == "regex":
        assert rich_outputs.regex_predictions is not None

    # case 3: dump and then load pipeline
    dump_dir = project_dir / "dumped_pipeline"
    pipeline_optimizer.dump(dump_dir)
    del pipeline_optimizer

    loaded_pipe = Pipeline.load(dump_dir)
    prediction_v2 = loaded_pipe.predict(utterances)
    assert prediction == prediction_v2


@pytest.mark.parametrize(
    "task_type",
    ["regex", "multiclass", "multilabel", "description"],
)
def test_inference_on_the_fly(dataset, task_type):
    project_dir = setup_environment()
    search_space = get_search_space(task_type)

    pipeline = Pipeline.from_search_space(search_space)

    logging_config = LoggingConfig(project_dir=project_dir, dump_modules=False, clear_ram=False)
    pipeline.set_config(logging_config)

    if task_type == "multilabel":
        dataset = dataset.to_multilabel()

    # case 1: simple inference on the fly
    pipeline.fit(dataset)
    utterances = ["123", "hello world"]
    prediction = pipeline.predict(utterances)
    assert len(prediction) == 2

    # case 2: rich inference on the fly
    rich_outputs = pipeline.predict_with_metadata(utterances)
    assert len(rich_outputs.predictions) == len(utterances)

    if task_type == "regex":
        assert rich_outputs.regex_predictions is not None

    # case 3: dump and then load pipeline
    pipeline.dump()
    del pipeline

    loaded_pipe = Pipeline.load(logging_config.dirpath)
    prediction_v2 = loaded_pipe.predict(utterances)
    assert prediction == prediction_v2


def test_load_with_overrided_params(dataset):
    project_dir = setup_environment()
    search_space = get_search_space("light")

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    logging_config = LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True)
    pipeline_optimizer.set_config(logging_config)

    context = pipeline_optimizer.fit(dataset)
    context.dump()

    # case 1: simple inference from file system
    inference_pipeline = Pipeline.load(logging_config.dirpath, embedder_config=EmbedderConfig(max_length=8))
    utterances = ["123", "hello world"]
    prediction = inference_pipeline.predict(utterances)
    assert len(prediction) == 2

    # case 2: rich inference from file system
    rich_outputs = inference_pipeline.predict_with_metadata(utterances)
    assert len(rich_outputs.predictions) == len(utterances)
    assert inference_pipeline.nodes[NodeType.scoring].module._embedder.max_length == 8
    del inference_pipeline

    # case 3: dump and then load pipeline
    pipeline_optimizer.dump()
    del pipeline_optimizer

    loaded_pipe = Pipeline.load(logging_config.dirpath, embedder_config=EmbedderConfig(max_length=8))
    prediction_v2 = loaded_pipe.predict(utterances)
    assert prediction == prediction_v2
    assert loaded_pipe.nodes[NodeType.scoring].module._embedder.max_length == 8


def test_no_saving(dataset):
    project_dir = setup_environment()
    search_space = get_search_space("light")

    pipeline_optimizer = Pipeline.from_search_space(search_space)

    logging_config = LoggingConfig(project_dir=project_dir, dump_modules=False, clear_ram=True)
    pipeline_optimizer.set_config(logging_config)

    pipeline_optimizer.fit(dataset)
    with pytest.raises(RuntimeError):
        pipeline_optimizer.dump()
