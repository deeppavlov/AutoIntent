"""Regex node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import REGEX_METRICS
from autointent.metrics.regex import RegexMetricFn
from autointent.modules import REGEX_MODULES
from autointent.modules.base import BaseRegex

from ._base import NodeInfo


class RegexNodeInfo(NodeInfo):
    """Regex node info."""

    metrics_available: ClassVar[Mapping[str, RegexMetricFn]] = REGEX_METRICS

    modules_available: ClassVar[Mapping[str, type[BaseRegex]]] = REGEX_MODULES

    node_type = NodeType.regex

    multiclass_available_metrics: ClassVar[Mapping[str, RegexMetricFn]] = REGEX_METRICS

    multilabel_available_metrics: ClassVar[Mapping[str, RegexMetricFn]] = REGEX_METRICS
