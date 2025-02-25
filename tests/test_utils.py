import pytest

from autointent.nodes import OptimizationSearchSpaceConfig
from autointent.utils import load_default_search_space


@pytest.mark.parametrize("multilabel", [True, False])
def test_load_default_configs(multilabel):
    search_space = load_default_search_space(multilabel=multilabel)
    OptimizationSearchSpaceConfig(search_space).model_dump()
