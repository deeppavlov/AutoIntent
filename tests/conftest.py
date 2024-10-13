import pathlib
from typing import Any
from uuid import uuid4

import pytest

from autointent import Context
from autointent.pipeline.optimization.utils import get_run_name, load_data, setup_logging
from autointent.pipeline.utils import get_db_dir

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
def context(load_clinic_subset):
    return Context(multiclass_intent_records=load_clinic_subset, multilabel_utterance_records=[],
                   test_utterance_records=[], device="cpu", mode="multiclass", multilabel_generation_config="",
                   regex_sampling=0, seed=0)
