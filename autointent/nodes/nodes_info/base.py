from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import METRIC_FN
from autointent.modules import Module


class NodeInfo:
    metrics_available: ClassVar[Mapping[str, METRIC_FN]]
    modules_available: ClassVar[Mapping[str, type[Module]]]
    node_type: NodeType
