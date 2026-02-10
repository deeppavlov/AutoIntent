"""Regex node info."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import REGEX_METRICS
from autointent.modules import REGEX_MODULES

from ._base import NodeInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from autointent.metrics.regex import RegexMetricFn
    from autointent.modules.base import BaseRegex


class RegexNodeInfo(NodeInfo):
    """Regex node info."""

    metrics_available: ClassVar[Mapping[str, RegexMetricFn]] = REGEX_METRICS

    modules_available: ClassVar[Mapping[str, type[BaseRegex]]] = REGEX_MODULES

    node_type = NodeType.regex

    multiclass_available_metrics: ClassVar[Mapping[str, RegexMetricFn]] = REGEX_METRICS

    multilabel_available_metrics: ClassVar[Mapping[str, RegexMetricFn]] = REGEX_METRICS
