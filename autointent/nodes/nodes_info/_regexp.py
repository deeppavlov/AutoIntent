"""Regexp node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import REGEXP_METRICS
from autointent.metrics.regexp import RegexpMetricFn
from autointent.modules.abc import Module
from autointent.modules.regexp import RegExp

from ._base import NodeInfo


class RegExpNodeInfo(NodeInfo):
    """Regexp node info."""

    metrics_available: ClassVar[Mapping[str, RegexpMetricFn]] = REGEXP_METRICS

    modules_available: ClassVar[Mapping[str, type[Module]]] = {NodeType.regexp: RegExp}

    node_type = NodeType.regexp
