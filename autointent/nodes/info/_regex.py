"""Regex node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import REGEX_METRICS
from autointent.metrics.regex import RegexMetricFn
from autointent.modules.base import BaseRegex
from autointent.modules.regex import Regex

from ._base import NodeInfo


class RegexNodeInfo(NodeInfo):
    """Regex node info."""

    metrics_available: ClassVar[Mapping[str, RegexMetricFn]] = REGEX_METRICS

    modules_available: ClassVar[Mapping[str, type[BaseRegex]]] = {NodeType.regex: Regex}

    node_type = NodeType.regex
