"""Tests for OptimizationSearchSpaceConfig.validate_nodes error branches."""

from __future__ import annotations

from typing import Any

import pytest

from autointent.schemas.node_validation import OptimizationSearchSpaceConfig


def test_item_must_be_dict() -> None:
    data: list[Any] = ["not-a-dict"]
    with pytest.raises(TypeError, match="must be a dictionary"):
        OptimizationSearchSpaceConfig(data)


def test_item_requires_node_type() -> None:
    data: list[Any] = [{}]
    with pytest.raises(TypeError, match="must have a 'node_type' key"):
        OptimizationSearchSpaceConfig(data)


def test_search_space_must_be_list() -> None:
    data: list[Any] = [{"node_type": "scoring", "search_space": "nope"}]
    with pytest.raises(TypeError, match="'search_space' key of type list"):
        OptimizationSearchSpaceConfig(data)


def test_search_space_item_requires_module_name() -> None:
    data: list[Any] = [{"node_type": "scoring", "search_space": [{}]}]
    with pytest.raises(TypeError, match="missing 'module_name'"):
        OptimizationSearchSpaceConfig(data)


def test_unknown_node_type() -> None:
    data: list[Any] = [{"node_type": "bogus", "search_space": [{"module_name": "knn"}]}]
    with pytest.raises(TypeError, match="Unknown node type"):
        OptimizationSearchSpaceConfig(data)


def test_invalid_module_params() -> None:
    data: list[Any] = [{"node_type": "scoring", "search_space": [{"module_name": "knn", "k": "not-an-int"}]}]
    with pytest.raises(TypeError, match="knn is invalid"):
        OptimizationSearchSpaceConfig(data)
