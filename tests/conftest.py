import pathlib
from typing import Any
from uuid import uuid4

import pytest

from autointent.pipeline.main import get_run_name, load_data, setup_logging
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
    return load_data(str(cur_path / "minimal_optimization" / "data" / "clinc_subset.json"))
