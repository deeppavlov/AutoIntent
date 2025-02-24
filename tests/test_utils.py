from typing import get_args

import pytest

from autointent.custom_types import SearchSpacePresets
from autointent.nodes import OptimizationConfig
from autointent.utils import load_preset


@pytest.mark.parametrize("preset", get_args(SearchSpacePresets))
def test_load_default_configs(preset):
    search_space = load_preset(preset)
    OptimizationConfig(search_space).model_dump()
