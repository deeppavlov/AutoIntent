import pathlib
from typing import Any
from uuid import uuid4

import pytest

from autointent import Context
from autointent.pipeline.optimization.utils import get_db_dir, get_run_name, load_data, setup_logging

cur_path = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def setup_environment() -> tuple[str, str]:
    setup_logging("DEBUG")
    uuid = uuid4()
    run_name = get_run_name(str(uuid))
    db_dir = get_db_dir("", run_name)
    return run_name, db_dir


@pytest.fixture
def load_clinic_subset() -> list[dict[str, Any]]:
    return load_data(str(cur_path / "minimal_optimization" / "data" / "clinc_subset.json"), multilabel=False)


@pytest.fixture
def context_multiclass(load_clinic_subset, setup_environment, dump_dir):
    run_name, db_dir = setup_environment
    return Context(
        multiclass_intent_records=load_clinic_subset,
        multilabel_utterance_records=[],
        test_utterance_records=[],
        device="cpu",
        mode="multiclass",
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
        db_dir=db_dir,
        dump_dir=dump_dir
    )


@pytest.fixture
def context_multilabel(load_clinic_subset, setup_environment, dump_dir):
    run_name, db_dir = setup_environment
    return Context(
        multiclass_intent_records=load_clinic_subset,
        multilabel_utterance_records=[],
        test_utterance_records=[],
        device="cpu",
        mode="multiclass_as_multilabel",
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
        db_dir=db_dir,
        dump_dir=dump_dir
    )

@pytest.fixture
def logs_dir() -> str:
    cur_path = pathlib.Path(__file__).parent.resolve()
    return cur_path / "logs"


@pytest.fixture
def dump_dir(logs_dir) -> str:
    return str(logs_dir / "module_dumps")
