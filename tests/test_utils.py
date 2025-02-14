import pytest

from autointent.nodes import OptimizationConfig
from autointent.utils import load_default_search_space


@pytest.mark.parametrize("multilabel", [True, False])
def test_load_default_configs(multilabel):
    search_space = load_default_search_space(multilabel=multilabel)
    OptimizationConfig(search_space).model_dump()
