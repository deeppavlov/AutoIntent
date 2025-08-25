"""Base node info class."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import METRIC_FN
from autointent.modules.base import BaseModule


class NodeInfo:
    """Base node info class."""

    metrics_available: ClassVar[Mapping[str, METRIC_FN]]
    """Available metrics for the node."""
    modules_available: ClassVar[Mapping[str, type[BaseModule]]]
    """Available modules for the node."""
    node_type: NodeType
    """Node type."""
    multiclass_available_metrics: ClassVar[Mapping[str, METRIC_FN]]
    """Available metrics for multiclass classification."""
    multilabel_available_metrics: ClassVar[Mapping[str, METRIC_FN]]
    """Available metrics for multilabel classification."""
