"""Regexp node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import regexp_partial_accuracy, regexp_partial_precision
from autointent.metrics.regexp import RegexpMetricFn
from autointent.modules import Module, RegExp
from autointent.utils import _funcs_to_dict

from .base import NodeInfo


class RegExpNodeInfo(NodeInfo):
    """Regexp node info."""

    metrics_available: ClassVar[Mapping[str, RegexpMetricFn]] = _funcs_to_dict(
        regexp_partial_accuracy,
        regexp_partial_precision,
    )

    modules_available: ClassVar[Mapping[str, type[Module]]] = {NodeType.regexp: RegExp}

    node_type = NodeType.regexp
