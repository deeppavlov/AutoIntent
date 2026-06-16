from typing import Any, cast

import pytest
from pydantic import ValidationError

from autointent import OptimizationConfig
from tests.conftest import get_search_space


def test_validate_full_config() -> None:
    # full_training.yaml is a top-level dict (not a list like the other YAMLs);
    # get_search_space's declared return type is list[dict[str, Any]] but yaml.safe_load
    # returns whatever the file structure dictates. Cast at the boundary; this is a
    # known src/ signature gap on load_search_space (see issue #315).
    config = cast("dict[str, Any]", get_search_space("full_training"))
    validated_config = OptimizationConfig(**config)
    assert isinstance(validated_config, OptimizationConfig)


def test_not_valid_reporting() -> None:
    config = cast("dict[str, Any]", get_search_space("full_training"))
    config["logging_config"]["report_to"] = "test"

    with pytest.raises(ValidationError):
        OptimizationConfig(**config)
