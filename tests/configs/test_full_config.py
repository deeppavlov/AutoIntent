from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError

from autointent import OptimizationConfig
from tests.conftest import get_search_space

if TYPE_CHECKING:
    from tests.conftest import TaskType


def test_validate_full_config() -> None:
    # full_training.yaml is a dict (not a list like the other YAMLs); get_search_space's
    # declared signature is list[dict[str, Any]] but yaml.safe_load returns whatever the
    # file structure dictates. Cast at the boundary; this is a known src/ signature gap.
    # reason: TaskType literal in conftest does not include 'full_training', but the
    # asset YAML by that name exists and is the contract this test pins.
    config = cast(
        "dict[str, Any]",
        get_search_space(cast("TaskType", "full_training")),
    )
    validated_config = OptimizationConfig(**config)
    assert isinstance(validated_config, OptimizationConfig)


def test_not_valid_reporting() -> None:
    config = cast(
        "dict[str, Any]",
        get_search_space(cast("TaskType", "full_training")),
    )
    config["logging_config"]["report_to"] = "test"

    with pytest.raises(ValidationError):
        OptimizationConfig(**config)
