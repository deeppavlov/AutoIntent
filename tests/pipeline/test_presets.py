from typing import get_args

import pytest

from autointent import Pipeline
from autointent.configs import DataConfig, LoggingConfig
from autointent.custom_types import SearchSpacePreset
from tests.conftest import setup_environment


@pytest.mark.parametrize("preset", get_args(SearchSpacePreset))
def test_presets(dataset, preset):
    project_dir = setup_environment()

    pipeline_optimizer = Pipeline.from_preset(preset)

    if preset in ["heavy_extra", "light", "heavy"]:
        return

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="ho"))

    pipeline_optimizer.fit(dataset, refit_after=False)
